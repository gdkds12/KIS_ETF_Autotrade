# Slash 명령·세션 관리자 

import discord
from discord import app_commands, Interaction, ButtonStyle, Embed, ui, Message, Thread
from discord.ext import commands
import logging
import asyncio
from datetime import datetime, timedelta, timezone
import uuid
from openai import OpenAI, AsyncOpenAI, APIError, RateLimitError # OpenAI Library
import json # For order parsing
import os
from enum import Enum, auto
import aiohttp # Added for making HTTP requests to FastAPI
import traceback # Import traceback module

# --- Registry and Utility imports ---
from src.utils.registry import COMMANDS, set_orchestrator # Keep COMMANDS, set_orchestrator
from src.utils import registry # Import the module itself for accessing ORCHESTRATOR
from src.utils.discord_utils import DiscordRequestType # Import the enum from utils
# ----------------------------------

# --- Configuration and DB imports ---
from src.config import settings
from src.db.models import SessionLocal, TradingSession, SessionLog # DB Models
from sqlalchemy import select
# ----------------------------------

# Remove direct top-level imports of components initialized in setup_hook
# from src.agents.orchestrator import Orchestrator # Moved to setup_hook
# from src.brokers.kis import KisBroker, KisBrokerError # Moved to setup_hook
# from qdrant_client import QdrantClient # Moved to setup_hook

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Configured in main execution block

# --- Configuration Check --- 
if not settings.DISCORD_TOKEN:
     raise ValueError("DISCORD_TOKEN must be set in the environment variables or .env file.")
if not settings.DISCORD_ORDER_CONFIRMATION_CHANNEL_ID:
     logger.warning("DISCORD_ORDER_CONFIRMATION_CHANNEL_ID is not set. Order confirmation feature will not work.")
     # Or raise an error if this channel is mandatory
     # raise ValueError("DISCORD_ORDER_CONFIRMATION_CHANNEL_ID must be set.")

# --- Configuration --- 
DISCORD_TOKEN = settings.DISCORD_TOKEN
GUILD_ID = 1363088557517967582 # Optional: Specify guild ID for faster command registration
INTENTS = discord.Intents.default()
INTENTS.message_content = True # Needs to be enabled in Developer Portal
INTENTS.members = True # Optional, if member info is needed

# --- OpenAI Client --- 
openai_client = None
if settings.OPENAI_API_KEY:
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    logger.info("OpenAI client initialized.")
else:
    logger.warning("OPENAI_API_KEY not set. OpenAI features will be disabled.")

# --- Constants & Enums ---
# Channel ID where order confirmations should be sent (Loaded from settings)
ORDER_CONFIRMATION_CHANNEL_ID = settings.DISCORD_ORDER_CONFIRMATION_CHANNEL_ID

class TradingBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=INTENTS) # Prefix not really used for slash commands
        self.active_sessions = {} # thread_id: {user_id, last_interaction_time, llm_session_id, etc.}
        self.db_session_factory = SessionLocal # Store factory
        # self.backend_client = BackendClient() # Placeholder for backend communication
        # Add placeholders for broker and qdrant client if needed elsewhere
        # self.broker: KisBroker | None = None 
        # self.qdrant_client: QdrantClient | None = None
        # self.orchestrator: Orchestrator | None = None

    async def setup_hook(self):
        # --- Lazy Imports and Component Initialization --- 
        # ① 필요한 모듈 lazy import
        # NOTE: settings and SessionLocal are already imported at top-level
        from src.brokers.kis import KisBroker # Lazy import
        from qdrant_client import QdrantClient # Lazy import
        from src.agents.orchestrator import Orchestrator # Lazy import
        # set_orchestrator is already imported at top-level
        
        broker_instance = None
        qdrant_instance = None
        orchestrator_instance = None

        try:
            # ② KisBroker 세팅
            if not all([settings.APP_KEY, settings.APP_SECRET, settings.CANO, settings.ACNT_PRDT]):
                logger.error("Missing required KIS API credentials or account info in settings for bot initialization.")
            else:
                is_virtual = settings.KIS_VIRTUAL_ACCOUNT
                base_url = settings.KIS_VIRTUAL_URL if is_virtual else settings.BASE_URL
                broker_instance = KisBroker(
                    app_key=settings.APP_KEY,
                    app_secret=settings.APP_SECRET,
                    base_url=base_url,
                    cano=settings.CANO,
                    acnt_prdt_cd=settings.ACNT_PRDT,
                    virtual_account=is_virtual
                )
                if broker_instance.check_token(): # Initial token check
                    logger.info("KIS Broker initialized and token validated within Discord Bot.")
                else:
                     logger.warning("KIS Broker initialized but failed initial token validation.")
            
            # ③ Qdrant 클라이언트 세팅
            try:
                qdrant_instance = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=10
                )
                _ = qdrant_instance.get_collections() # Test connection
                logger.info("Qdrant client initialized within Discord Bot.")
            except Exception as q_e:
                 logger.error(f"Failed to initialize Qdrant client in bot: {q_e}. RAG features might fail.", exc_info=True)
                 qdrant_instance = None # Ensure it's None on failure

            # ④ Orchestrator 인스턴스 생성 및 registry 등록
            if broker_instance: # Only initialize if broker is ready
                orchestrator_instance = Orchestrator(
                    broker=broker_instance,
                    db_session_factory=self.db_session_factory,
                    qdrant_client=qdrant_instance # Pass potentially None qdrant client
                )
                set_orchestrator(orchestrator_instance) # Register the instance
                logger.info("✅ Orchestrator initialized and registry ORCHESTRATOR has been set in Discord bot")
            else:
                 logger.error("Orchestrator cannot be initialized because Broker failed to initialize.")
                 set_orchestrator(None) # Explicitly set registry to None

        except Exception as init_e:
            logger.error(f"Critical error during bot component initialization: {init_e}", exc_info=True)
            set_orchestrator(None) # Ensure registry is None on critical failure
            # Consider closing the bot or preventing full startup
            # await self.close()
            # return

        # Sync commands (globally or to a specific guild)
        if GUILD_ID:
            guild = discord.Object(id=GUILD_ID)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info(f"Synced commands to guild {GUILD_ID}")
        else:
            await self.tree.sync()
            logger.info("Synced commands globally")
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info("------")
        self.loop.create_task(self.check_inactive_sessions()) # Start background task

    async def on_ready(self):
        logger.info(f'{self.user} has connected to Discord!')

    async def log_message_to_db(self, session_uuid: str, actor: str, message: str, suggested_order_json: str = None, order_confirmed: bool = None):
         """Log a message to the database associated with the session."""
         db = self.db_session_factory()
         try:
             # Find the session ID from the UUID
             stmt = select(TradingSession).where(TradingSession.session_uuid == session_uuid)
             session_obj = db.scalars(stmt).first()
             if not session_obj:
                  logger.error(f"Could not find session with UUID {session_uuid} to log message.")
                  return
                  
             log_entry = SessionLog(
                 session_id=session_obj.id,
                 actor=actor,
                 message=message,
                 suggested_order_json=suggested_order_json,
                 order_confirmed=order_confirmed
             )
             db.add(log_entry)
             db.commit()
             logger.info(f"Logged message for session {session_uuid} to DB.")
         except Exception as e:
              logger.error(f"Failed to log message to DB for session {session_uuid}: {e}", exc_info=True)
              db.rollback()
         finally:
              db.close()
              
    async def get_openai_response(self, session_info: dict, user_message: str) -> tuple[str, dict | None, str]:
        """Get response from OpenAI main tier model, managing conversation history and function calls."""
        if not openai_client:
            return "(죄송합니다, OpenAI API가 설정되지 않아 응답할 수 없습니다.)", None, "(Debug info not generated)"

        llm_session_id = session_info['llm_session_id']
        message_history = session_info.get('message_history', [])

        # Add user message to history
        message_history.append({"role": "user", "content": user_message})
        
        system_prompt = (
            "당신은 한국 ETF 시장에 대한 금융 도우미 AI입니다. 사용자의 질문에 답변하고, 필요시 내부 명령(get_balance, get_positions, get_market_summary 등)을 호출하여 정보를 얻습니다."
            " 시장 분석, 종목 정보 제공, 간단한 계산 등을 수행합니다. "
            "투자 관련 조언은 제공하지만, 최종 결정은 사용자의 책임임을 명시해야 합니다."
            "만약 매수 또는 매도 주문을 제안해야 한다면, 반드시 다음 JSON 형식으로 제안 내용을 응답 끝에 포함시키세요: "
            '\n{\n  "suggested_order": {\n    "symbol": "종목코드 (예: 069500)",\n    "action": "buy 또는 sell",\n    "quantity": 주문수량 (정수),\n    "price": 주문가격 (지정가=실제가격, 시장가=0)\n  }\n}'
            "주문 제안이 없다면, JSON 부분을 포함하지 마세요."
            "모든 답변은 한국어로 제공하세요."
        )

        messages = [
            {"role": "system", "content": system_prompt}
        ] + message_history[-10:] # Keep last 10 messages

        # -------- Function calling set‑up --------
        functions = [fn._oas for fn in COMMANDS.values()]
        
        response_text = "(오류 발생)"
        suggested_order = None
        debug_info = "(Debug info not generated)" # Default debug info

        try:
            logger.info(f"Sending request to OpenAI model: {settings.LLM_MAIN_TIER_MODEL} for session {llm_session_id} (with function calling)")
            
            # -----------------------------------------
            #   1st completion - Allow function call or direct answer
            # -----------------------------------------
            completion = await openai_client.chat.completions.create(
                model=settings.LLM_MAIN_TIER_MODEL,
                messages=messages,
                functions=functions, # Pass function specs
                function_call="auto", # Let the model decide
                temperature=0.7,
                max_tokens=1000,
            )

            choice = completion.choices[0]
            message_from_llm = choice.message

            # --- DEBUG INFO GENERATION --- 
            fc = None
            if choice.finish_reason == 'function_call':
                fc = {
                    "name": choice.message.function_call.name,
                    "arguments": choice.message.function_call.arguments
                }
            debug_info = json.dumps({
                "finish_reason": choice.finish_reason,
                "function_call": fc,
                "raw_content": choice.message.content or "",
                "orchestrator_set": registry.ORCHESTRATOR is not None # Include orchestrator status
            }, ensure_ascii=False, indent=2)
            logger.debug(f"Generated Debug Info: {debug_info}")
            # ---------------------------

            # 4️⃣ Check if a function call was requested
            if choice.finish_reason != 'function_call':
                # No function call, use the direct response
                logger.info(f"LLM responded directly for session {llm_session_id}.")
                response_text = message_from_llm.content
            else:
                # Function call requested, proceed with execution
                logger.info(f"LLM requested function call: {message_from_llm.function_call.name} for session {llm_session_id}.")
                fn_name = message_from_llm.function_call.name
                try:
                    fn_args = json.loads(message_from_llm.function_call.arguments or "{}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse arguments for function {fn_name}: {message_from_llm.function_call.arguments}")
                    raise ValueError(f"Invalid arguments for function {fn_name}")

                if fn_name not in COMMANDS:
                    logger.error(f"LLM requested unknown function: {fn_name}")
                    raise ValueError(f"Unknown function requested: {fn_name}")
                
                # Execute the command (wrapper from registry.py)
                try:
                    logger.info(f"Executing command: {fn_name} with args: {fn_args}")
                    loop = asyncio.get_running_loop()
                    fn_result = await loop.run_in_executor(None, lambda: COMMANDS[fn_name](**fn_args))
                    logger.info(f"Command {fn_name} executed. Result: {str(fn_result)[:100]}...")
                except Exception as exec_e:
                    logger.error(f"Error executing command {fn_name}: {exec_e}", exc_info=True)
                    # 오류 발생 시, 디버그 정보 업데이트 및 바로 반환
                    error_debug = (
                        f"```DEBUG\n"
                        f"ORCHESTRATOR set: {registry.ORCHESTRATOR is not None}\n"
                        f"Function Call Failed: {fn_name}({fn_args})\n"
                        f"Error: {type(exec_e).__name__}: {exec_e}\n"
                        f"```"
                    )
                    response_text = f"(내부 명령 '{fn_name}' 실행 중 오류 발생)"
                    # Return the error message and the specific error debug info
                    return response_text, None, error_debug # Return 3 values

                # -----------------------------------------
                #   2nd completion - Send result back to LLM
                # -----------------------------------------
                logger.info(f"Sending function result back to LLM for final response (session {llm_session_id}).")
                second_completion = await openai_client.chat.completions.create(
                    model=settings.LLM_MAIN_TIER_MODEL,
                    messages=messages + [
                        message_from_llm, # Include the function call request
                        {
                            "role": "function",
                            "name": fn_name,
                            "content": json.dumps(fn_result, ensure_ascii=False),
                        },
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                    # NOTE: Do not pass functions here, we want a direct answer now
                )
                response_text = second_completion.choices[0].message.content
                logger.info(f"Received final response from OpenAI after function call (session {llm_session_id}).")

            # --- Order Parsing (applies to final response_text) --- 
            if response_text:
                # Add final AI response to history
                message_history.append({"role": "assistant", "content": response_text})
                session_info['message_history'] = message_history
                
                # Parse suggested order (logic remains the same)
                try:
                    json_start = response_text.rfind('{\n  "suggested_order":')
                    if json_start != -1:
                        json_part = response_text[json_start:]
                        try:
                             order_data = json.loads(json_part)
                             if "suggested_order" in order_data:
                                 suggested_order = order_data["suggested_order"]
                                 if all(k in suggested_order for k in ['symbol', 'action', 'quantity', 'price']) \
                                    and isinstance(suggested_order['quantity'], int) \
                                    and isinstance(suggested_order['price'], int): 
                                     logger.info(f"Parsed suggested order: {suggested_order}")
                                 else:
                                     logger.warning(f"Parsed JSON for order is incomplete or invalid: {suggested_order}")
                                     suggested_order = None
                        except json.JSONDecodeError as json_e:
                            logger.warning(f"Failed to parse JSON from LLM response: {json_e}. Response part: {json_part[:100]}...")
                            suggested_order = None
                except Exception as parse_e:
                     logger.error(f"Error during suggested order parsing: {parse_e}", exc_info=True)
                     suggested_order = None
            else:
                 response_text = "(AI가 빈 응답을 반환했습니다.)"

        except RateLimitError as e:
             logger.error(f"OpenAI Rate Limit Exceeded: {e}")
             response_text = "(현재 요청량이 많아 잠시 후 다시 시도해주세요.)"
        except APIError as e:
             logger.error(f"OpenAI API Error: {e}")
             response_text = f"(API 오류 발생: {e})"
        except Exception as e:
             logger.error(f"Unexpected error interacting with OpenAI or executing function: {e}", exc_info=True)
             response_text = "(AI 응답 생성 또는 내부 명령 실행 중 오류가 발생했습니다.)"
             
        # Return final response, order, and debug info
        return response_text, suggested_order, debug_info # Return 3 values

    async def on_message(self, message: Message):
        # Ignore messages from the bot itself
        if message.author == self.user:
            return
        
        # Check if the message is in an active trade session thread
        if isinstance(message.channel, Thread) and message.channel.id in self.active_sessions:
            session_info = self.active_sessions[message.channel.id]
            user_id = session_info['user_id']
            
            # Ignore messages from users other than the one who started the session
            if message.author.id != user_id:
                 return

            logger.info(f"[Session:{message.channel.id}] Received message from user {message.author.id}: {message.content[:50]}...")
            session_info['last_interaction_time'] = asyncio.get_event_loop().time()

            # Log user message to DB
            await self.log_message_to_db(session_uuid=session_info['llm_session_id'], actor="user", message=message.content)

            # --- LLM Interaction --- 
            async with message.channel.typing():
                try:
                    # LLM 호출 및 디버그 정보까지 함께 돌려받도록 시그니처 변경
                    response_text, suggested_order, debug_info = await self.get_openai_response(session_info, message.content)
                    
                    # Log AI response to DB
                    await self.log_message_to_db(
                        session_uuid=session_info['llm_session_id'], 
                        actor="ai", 
                        message=response_text,
                        suggested_order_json=json.dumps(suggested_order) if suggested_order else None
                    )
                    
                    # Send response to Discord
                    view = None
                    if suggested_order:
                        view = OrderConfirmationView(bot=self, session_thread_id=message.channel.id, order_details=suggested_order, db_session_factory=self.db_session_factory)
                    
                    # 자연어 질문에는 debug_info 없이 순수 답변만 전송
                    await message.channel.send(response_text, view=view)
                    logger.info(f"[Session:{message.channel.id}] Sent response to user.")

                except Exception as e:
                    logger.error(f"[Session:{message.channel.id}] Error processing message: {e}", exc_info=True)
                    tb = traceback.format_exc() # Format the traceback
                    debug_msg = (
                        f"```DEBUG\n"
                        f"ORCHESTRATOR set: {registry.ORCHESTRATOR is not None}\n"
                        f"Error: {type(e).__name__}: {e}\n"
                        f"Traceback:\n{tb}"
                        f"```"
                    )
                    # Truncate if too long for Discord message limit (2000 chars)
                    if len(debug_msg) > 1900:
                        debug_msg = debug_msg[:1900] + "... (truncated)```"
                        
                    await message.channel.send(
                        "⚠️ 처리 중 오류가 발생했습니다. 관리자에게 문의하세요.\n" + debug_msg
                    )
        else:
            # Process other messages or commands if needed (currently only handles session threads)
            await super().on_message(message) # Process regular commands if any
            
    async def check_inactive_sessions(self):
        """Periodically check for inactive sessions and archive them."""
        await self.wait_until_ready()
        while not self.is_closed():
            await asyncio.sleep(60 * 5) # Check every 5 minutes
            now = asyncio.get_event_loop().time()
            inactive_threshold = 60 * 30 # 30 minutes
            sessions_to_archive = []

            for thread_id, session_info in list(self.active_sessions.items()): # Iterate over a copy
                last_interaction = session_info.get('last_interaction_time', 0)
                if now - last_interaction > inactive_threshold:
                    sessions_to_archive.append(thread_id)
            
            for thread_id in sessions_to_archive:
                try:
                    thread = self.get_channel(thread_id) or await self.fetch_channel(thread_id)
                    if isinstance(thread, Thread) and not thread.archived:
                        logger.info(f"Archiving inactive session thread: {thread_id}")
                        # --- Summarization (using MemoryRAG) --- 
                        try:
                             # Directly instantiate MemoryRAG for summary (pass factory)
                             memory_agent = MemoryRAG(db_session_factory=self.db_session_factory)
                             memory_agent.summarize_and_upsert(session_uuid=session_info['llm_session_id'])
                             # Fetch the summary from DB to display in thread (optional)
                             db = self.db_session_factory()
                             try:
                                 stmt = select(TradingSession.summary).where(TradingSession.session_uuid == session_info['llm_session_id'])
                                 summary = db.scalars(stmt).first() or "(요약 정보 없음)"
                             finally:
                                 db.close()
                             await thread.send(f"*세션이 30분 동안 비활성 상태여서 종료 및 요약합니다.*\n{summary}")
                             logger.info(f"Session {session_info['llm_session_id']} summary completed.")
                        except Exception as summary_e:
                             logger.error(f"Failed to summarize session {session_info['llm_session_id']}: {summary_e}")
                             await thread.send("*세션이 30분 동안 비활성 상태여서 종료합니다. (요약 중 오류 발생)*")
                             # Update DB to mark inactive even if summary fails?
                             db = self.db_session_factory()
                             try:
                                 stmt = select(TradingSession).where(TradingSession.session_uuid == session_info['llm_session_id'])
                                 session_obj = db.scalars(stmt).first()
                                 if session_obj and session_obj.is_active:
                                     session_obj.is_active = False
                                     session_obj.end_time = datetime.now(timezone.utc)
                                     db.commit()
                             except Exception as db_e:
                                 logger.error(f"Failed to mark session {session_info['llm_session_id']} inactive in DB after summary failure: {db_e}")
                                 db.rollback()
                             finally:
                                 db.close()
                                 
                        await thread.edit(archived=True, locked=True)
                        logger.info(f"Thread {thread_id} archived and locked.")
                    # Remove from active sessions
                    del self.active_sessions[thread_id]
                except discord.NotFound:
                    logger.warning(f"Thread {thread_id} not found for archiving. Removing from active list.")
                    if thread_id in self.active_sessions:
                        del self.active_sessions[thread_id]
                except discord.Forbidden:
                     logger.error(f"Missing permissions to archive thread {thread_id}.")
                     # Consider removing from active sessions anyway to prevent loop
                     if thread_id in self.active_sessions:
                         del self.active_sessions[thread_id]
                except Exception as e:
                    logger.error(f"Error archiving thread {thread_id}: {e}", exc_info=True)
                    # Potentially remove from active list to avoid repeated errors
                    if thread_id in self.active_sessions:
                        del self.active_sessions[thread_id]

# --- Views (Buttons) --- 
class OrderConfirmationView(ui.View):
    def __init__(self, bot: TradingBot, session_thread_id: int, order_details: dict, db_session_factory):
        super().__init__(timeout=60 * 10) # View timeout after 10 minutes
        self.bot = bot
        self.session_thread_id = session_thread_id
        self.order_details = order_details
        self.db_session_factory = db_session_factory # Store factory
        self.confirmed = False

    async def log_confirmation_to_db(self, confirmed: bool):
        session_info = self.bot.active_sessions.get(self.session_thread_id)
        if session_info:
             await self.bot.log_message_to_db(
                 session_uuid=session_info['llm_session_id'],
                 actor="system", # Or user?
                 message=f"Order {'confirmed' if confirmed else 'cancelled'} by user.",
                 suggested_order_json=json.dumps(self.order_details), # Log the order context
                 order_confirmed=confirmed
             )
             
    @ui.button(label="✅ 주문 실행", style=ButtonStyle.green, custom_id="confirm_order")
    async def confirm_button(self, interaction: Interaction, button: ui.Button):
        # Check if the interaction user is the one who started the session
        session_info = self.bot.active_sessions.get(self.session_thread_id)
        if not session_info or interaction.user.id != session_info['user_id']:
            await interaction.response.send_message("세션을 시작한 사용자만 주문을 실행할 수 있습니다.", ephemeral=True)
            return
        
        if self.confirmed:
             await interaction.response.send_message("이미 처리된 주문입니다.", ephemeral=True)
             return

        self.confirmed = True
        logger.info(f"[Session:{session_info['llm_session_id']}|Th:{self.session_thread_id}] User confirmed order: {self.order_details}")
        await interaction.response.defer() # Acknowledge interaction, will edit later

        # Log confirmation to DB
        await self.log_confirmation_to_db(confirmed=True)

        # Disable buttons after confirmation
        for item in self.children:
            item.disabled = True
        await interaction.edit_original_response(view=self)
        
        # --- Execute Order via Backend --- 
        try:
            await interaction.followup.send(f"{interaction.user.mention} 주문을 실행합니다... (실제 실행 로직 연결 필요) {self.order_details}")
            # TODO: Send order details to backend/orchestrator for execution
            # execution_result = self.bot.backend_client.execute_order(session_info['llm_session_id'], self.order_details)
            await asyncio.sleep(2) # Simulate execution
            execution_result = {'success': True, 'message': '[Mock] 주문이 성공적으로 접수되었습니다.'}
            
            result_message = f"✅ **주문 실행 결과:** {execution_result['message']}" if execution_result['success'] else f"❌ **주문 실행 실패:** {execution_result['message']}"
            await interaction.followup.send(result_message)
            logger.info(f"[Session:{session_info['llm_session_id']}|Th:{self.session_thread_id}] Mock order execution result sent.")
            
        except Exception as e:
            logger.error(f"[Session:{session_info['llm_session_id']}|Th:{self.session_thread_id}] Error executing order: {e}", exc_info=True)
            await interaction.followup.send("주문 실행 중 오류가 발생했습니다.")
        
        # Update last interaction time
        if self.session_thread_id in self.bot.active_sessions:
             self.bot.active_sessions[self.session_thread_id]['last_interaction_time'] = asyncio.get_event_loop().time()
        self.stop() # Stop the view

    @ui.button(label="❌ 취소", style=ButtonStyle.red, custom_id="cancel_order")
    async def cancel_button(self, interaction: Interaction, button: ui.Button):
        session_info = self.bot.active_sessions.get(self.session_thread_id)
        if not session_info or interaction.user.id != session_info['user_id']:
            await interaction.response.send_message("세션을 시작한 사용자만 취소할 수 있습니다.", ephemeral=True)
            return
            
        logger.info(f"[Session:{session_info['llm_session_id']}|Th:{self.session_thread_id}] User cancelled order confirmation.")
        await interaction.response.defer() # Acknowledge interaction

        # Log cancellation to DB
        await self.log_confirmation_to_db(confirmed=False)

        # Disable buttons
        for item in self.children:
            item.disabled = True
        await interaction.edit_original_response(content="주문 제안이 취소되었습니다.", view=self)
        
        # Update last interaction time
        if self.session_thread_id in self.bot.active_sessions:
            self.bot.active_sessions[self.session_thread_id]['last_interaction_time'] = asyncio.get_event_loop().time()
        self.stop()

    async def on_timeout(self):
        logger.info(f"[Session:{self.session_thread_id}] Order confirmation view timed out.")
        # Disable buttons on timeout
        for item in self.children:
            item.disabled = True
        # Attempt to edit the original message if possible (might fail if message deleted)
        try:
            message = await self.message # Assumes view has reference to the message it was sent with
            if message:
                await message.edit(content="*주문 확인 시간이 초과되었습니다.*", view=self)
        except Exception as e:
             logger.warning(f"[Session:{self.session_thread_id}] Could not edit message on view timeout: {e}")

# --- Slash Commands --- 
bot = TradingBot()

@bot.tree.command(name="trade", description="AI 트레이딩 세션을 시작합니다.")
async def trade_command(interaction: Interaction):
    """Handles the /trade command to start a new trading session thread."""
    user = interaction.user
    channel = interaction.channel
    logger.info(f"Received /trade command from {user.id} in channel {channel.id}")

    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message("텍스트 채널에서만 /trade 명령을 사용할 수 있습니다.", ephemeral=True)
        return

    try:
        # Create a new thread for the session
        thread_name = f"🤖 Trade Session - {user.display_name} ({datetime.now().strftime('%H:%M')})"
        # Ensure the bot has permission to create public threads
        thread = await channel.create_thread(name=thread_name, auto_archive_duration=1440) # 24 hours archive
        logger.info(f"Created thread {thread.id} for user {user.id}")
        
        # Send initial message in the thread
        initial_message = (f"{user.mention}, 안녕하세요! 👋\n"
                           f"이 스레드에서 ETF 트레이딩 관련 질문을 하거나 시장 정보를 요청할 수 있습니다.\n"
                           f"(예: \"오늘 시장 요약해줘\", \"KODEX 200 전망 알려줘\")\n\n"
                           f"*참고: 저는 실제 주문을 제안할 수 있으며, ✅ 버튼을 눌러 실행을 요청할 수 있습니다.*\n"
                           f"*30분 동안 활동이 없으면 스레드는 자동으로 종료됩니다.*")
        await thread.send(initial_message)
        
        # Register the new session
        session_uuid = str(uuid.uuid4()) # Unique ID for backend/LLM interaction
        db = bot.db_session_factory()
        try:
             new_session_db = TradingSession(
                 session_uuid=session_uuid,
                 discord_thread_id=str(thread.id), # Store as string
                 discord_user_id=str(user.id) # Store as string
                 # account_id= # TODO: Link to user's account if needed/possible
             )
             db.add(new_session_db)
             db.commit()
             logger.info(f"Created TradingSession entry in DB for UUID {session_uuid}")
        except Exception as e:
             logger.error(f"Failed to create TradingSession in DB for thread {thread.id}: {e}", exc_info=True)
             db.rollback()
             # Fail the command if DB entry fails?
             await thread.delete() # Clean up thread
             await interaction.response.send_message("세션 생성 중 DB 오류가 발생했습니다.", ephemeral=True)
             return # Stop further processing
        finally:
             db.close()
        # --- End DB Create --- 
        
        bot.active_sessions[thread.id] = {
            'user_id': user.id,
            'channel_id': channel.id,
            'start_time': asyncio.get_event_loop().time(),
            'last_interaction_time': asyncio.get_event_loop().time(),
            'llm_session_id': session_uuid,
            'message_history': [] # Initialize message history
        }
        logger.info(f"Started active session tracking for UUID {session_uuid} (Thread: {thread.id})")
        
        # Respond to the original interaction (ephemeral)
        await interaction.response.send_message(f"새로운 트레이딩 세션 스레드를 시작했습니다: {thread.mention}", ephemeral=True)

    except discord.Forbidden:
        logger.error(f"Bot lacks permission to create threads in channel {channel.id}.")
        await interaction.response.send_message("이 채널에 스레드를 생성할 권한이 없습니다. 서버 관리자에게 문의하세요.", ephemeral=True)
    except Exception as e:
        logger.error(f"Error handling /trade command: {e}", exc_info=True)
        # Attempt to respond ephemerally if interaction not already responded to
        if not interaction.response.is_done():
             await interaction.response.send_message("세션 시작 중 오류가 발생했습니다.", ephemeral=True)
        else: # If already responded (e.g., due to earlier error), try followup
             try:
                 await interaction.followup.send("세션 시작 중 오류가 발생했습니다.", ephemeral=True)
             except Exception as followup_e:
                  logger.error(f"Failed to send followup error message: {followup_e}")

@bot.tree.command(name="run_cycle", description="수동으로 일일 자동매매 사이클을 시작합니다.")
async def run_cycle_command(interaction: Interaction):
    """Handles the /run_cycle command to manually trigger the orchestrator."""
    user = interaction.user
    logger.info(f"Received /run_cycle command from {user.id} ({user.name})...")

    fastapi_trigger_endpoint = "http://localhost:8000/trigger_cycle" # TODO: Make configurable?

    await interaction.response.defer(ephemeral=True) # Acknowledge command, might take time

    try:
        logger.info(f"Attempting to trigger orchestrator cycle via endpoint: {fastapi_trigger_endpoint}")
        async with aiohttp.ClientSession() as session:
            async with session.post(fastapi_trigger_endpoint) as response:
                response_status = response.status
                response_json = await response.json()
                logger.info(f"FastAPI trigger response: Status={response_status}, Body={response_json}")

                if response.ok and response_json.get("status") == "triggered":
                    await interaction.followup.send(f"✅ Orchestrator 사이클 실행을 백그라운드에서 시작했습니다.", ephemeral=True)
                else:
                     await interaction.followup.send(f"❌ Orchestrator 사이클 트리거 실패 (Status: {response_status}): {response_json.get('detail', 'Unknown error')}", ephemeral=True)

    except aiohttp.ClientConnectorError as e:
        logger.error(f"Cannot connect to FastAPI backend at {fastapi_trigger_endpoint} to trigger cycle: {e}")
        await interaction.followup.send(f"❌ 백엔드({fastapi_trigger_endpoint}) 연결 실패. 사이클을 시작할 수 없습니다.", ephemeral=True)
    except Exception as e:
        logger.error(f"Error triggering orchestrator cycle: {e}", exc_info=True)
        await interaction.followup.send(f"❌ 사이클 트리거 중 예외 발생: {e}", ephemeral=True)

# --- NEW DEBUG COMMAND --- 
@bot.tree.command(name="debug_balance", description="(디버그) 직접 계좌 잔고 JSON 반환")
async def debug_balance(interaction: Interaction):
    """Directly calls the registered get_balance command for debugging."""
    logger.info(f"Received /debug_balance command from {interaction.user.id}")
    # 직접 registry에 등록된 get_balance 호출
    from src.utils.registry import COMMANDS, ORCHESTRATOR # Import ORCHESTRATOR as well
    
    # Check if orchestrator is ready
    if ORCHESTRATOR is None or not hasattr(ORCHESTRATOR, 'broker'):
        await interaction.response.send_message(
            "오류: Orchestrator 또는 Broker가 아직 준비되지 않았습니다.",
            ephemeral=True
        )
        return
        
    try:
        # Execute the command wrapper directly
        result = COMMANDS['get_balance']()
        await interaction.response.send_message(
            f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```",
            ephemeral=True
        )
    except KeyError:
        logger.error("Command 'get_balance' not found in registry.")
        await interaction.response.send_message(
            "오류: 'get_balance' 명령어를 레지스트리에서 찾을 수 없습니다.",
            ephemeral=True
        )
    except Exception as e:
        logger.error(f"Error executing debug_balance command: {e}", exc_info=True)
        await interaction.response.send_message(
            f"명령어 실행 중 오류 발생: ```{type(e).__name__}: {e}```",
            ephemeral=True
        )

# --- NEW DEBUG COMMAND --- 
@bot.tree.command(name="debug_market_summary", description="(디버그) 시장 동향 요약 및 디버그 정보 반환")
@app_commands.describe(query="요약할 시장 동향의 쿼리 (예: '한국 ETF 시장 트렌드')")
async def debug_market_summary(interaction: Interaction, query: str):
    """사용자 쿼리를 기반으로 Finnhub+LLM 요약을 호출하고 결과를 JSON&debug로 반환합니다."""
    logger.info(f"Received /debug_market_summary command from {interaction.user.id} with query: '{query}'")
    from src.utils.registry import COMMANDS, ORCHESTRATOR
    
    if ORCHESTRATOR is None or not hasattr(ORCHESTRATOR, 'info_crawler'):
        await interaction.response.send_message(
            "오류: Orchestrator 또는 InfoCrawler가 아직 준비되지 않았습니다.", 
            ephemeral=True
        )
        return
        
    await interaction.response.defer(ephemeral=True) # Acknowledge interaction, might take time
    try:
        # 1) 시장 요약 호출 (Using the command wrapper which calls orchestrator.info_crawler.get_market_summary)
        # Make sure to run synchronous code in an executor if it blocks
        loop = asyncio.get_running_loop()
        raw_result = await loop.run_in_executor(None, lambda: COMMANDS['get_market_summary'](query))
        
        # 2) 디버그 정보 생성
        debug_info = (
            f"Orchestrator set: {ORCHESTRATOR is not None}\n"
            f"User query: {query}\n"
            f"Raw result length: {len(str(raw_result))}"
        )
        
        # Format result (assuming it's a string, try JSON otherwise)
        try:
            # If the result is JSON parseable string, format it nicely
            parsed_json = json.loads(raw_result)
            formatted_result = json.dumps(parsed_json, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, TypeError):
            # Otherwise, treat as plain text
            formatted_result = str(raw_result)
            
        # Construct the final message
        final_message = (
            f"```json\n{formatted_result}\n```\n" # Use json for structure, even if it's just a string inside
            f"```DEBUG\n{debug_info}\n```"
        )
        
        # Check length before sending
        if len(final_message) > 2000:
            truncated_result = formatted_result[:(1950 - len(debug_info))] + "... (truncated)"
            final_message = (
                 f"```json\n{truncated_result}\n```\n"
                 f"```DEBUG\n{debug_info}\n```"
             )
             
        await interaction.followup.send(final_message, ephemeral=True)
        
    except KeyError:
        logger.error("Command 'get_market_summary' not found in registry.")
        await interaction.followup.send(
            "오류: 'get_market_summary' 명령어를 레지스트리에서 찾을 수 없습니다.", 
            ephemeral=True
        )
    except Exception as e:
        logger.error(f"Error executing debug_market_summary command: {e}", exc_info=True)
        tb = traceback.format_exc()
        error_message = (
             f"명령 실행 중 오류 발생: ```{type(e).__name__}: {e}```\n"
             f"```Traceback:\n{tb[:1500]}...```"
        )
        if len(error_message) > 2000:
            error_message = error_message[:1997] + "..."
        await interaction.followup.send(error_message, ephemeral=True)
# -------------------------

# --- Orchestrator Communication --- 
# This function is intended to be called by the Orchestrator.
# In a real-world scenario, this might be part of a class or use a 
# more robust communication mechanism (like API endpoint, message queue).

async def send_discord_request(request_type: DiscordRequestType, data: dict) -> bool:
    """Receives requests from Orchestrator and sends messages to Discord.
    
    Args:
        request_type: The type of request (e.g., ORDER_CONFIRMATION).
        data: The data payload associated with the request.
        
    Returns:
        True if the message was sent successfully (or placeholder success), False otherwise.
    """
    logger.info(f"Received request from Orchestrator: Type={request_type}, Data Keys={list(data.keys())}")
    
    if request_type == DiscordRequestType.ORDER_CONFIRMATION:
        request_id = data.get("request_id")
        orders = data.get("orders")
        if not request_id or not orders:
            logger.error("Missing request_id or orders in ORDER_CONFIRMATION data.")
            return False
            
        if not ORDER_CONFIRMATION_CHANNEL_ID:
             logger.error("Order Confirmation Channel ID is not configured. Cannot send confirmation message.")
             return False
                 
        try:
            channel = bot.get_channel(ORDER_CONFIRMATION_CHANNEL_ID)
            if not channel:
                logger.error(f"Cannot find confirmation channel with ID: {ORDER_CONFIRMATION_CHANNEL_ID}")
                return False

            # Format the message
            embed = discord.Embed(
                title="🚨 Order Confirmation Required 🚨", 
                description=f"LLM has proposed the following trades. Please review and approve/reject/hold.\nRequest ID: `{request_id}`",
                color=discord.Color.orange()
            )
            
            order_details = ""
            total_estimated_value = 0
            for i, order in enumerate(orders):
                action = order.get('action', 'N/A').upper()
                symbol = order.get('symbol', 'N/A')
                quantity = order.get('quantity', 'N/A')
                reason = order.get('reason', 'N/A')
                # TODO: Estimate value based on current price for better user info
                order_details += f"**{i+1}. {action} {symbol} ({quantity} shares)**\n   Reason: _{reason}_\n"
                
            embed.add_field(name="Proposed Orders", value=order_details, inline=False)
            # embed.add_field(name="Estimated Total Value", value=f"~{total_estimated_value:,.0f} KRW", inline=False)
            embed.set_footer(text="Please respond within 1 hour.")

            # Create the view with buttons
            view = ConfirmationView(request_id, orders)
            
            # Send the message to the channel
            await channel.send(embed=embed, view=view)
            logger.info(f"Sent order confirmation message {request_id} to channel {channel.id}")
            return True # Indicate message sent

        except discord.errors.Forbidden:
             logger.error(f"Bot lacks permissions to send messages/embeds/views in channel {ORDER_CONFIRMATION_CHANNEL_ID}.")
             return False
        except Exception as e:
            logger.error(f"Failed to send order confirmation message {request_id}: {e}", exc_info=True)
            return False
            
    elif request_type == DiscordRequestType.GENERAL_NOTIFICATION:
        message = data.get("message", "No message content.")
        # TODO: Implement sending general notifications to a specific channel or user
        logger.info(f"Received general notification to send: {message[:100]}...")
        # channel = bot.get_channel(NOTIFICATION_CHANNEL_ID) 
        # if channel: await channel.send(message)
        pass
        return True # Placeholder success
        
    elif request_type == DiscordRequestType.ALERT:
        message = data.get("message", "No alert content.")
        # TODO: Implement sending alerts (maybe tagging specific roles/users)
        logger.warning(f"Received ALERT to send: {message[:100]}...")
        # channel = bot.get_channel(ALERT_CHANNEL_ID)
        # if channel: await channel.send(f"@here **ALERT:** {message}")
        pass
        return True # Placeholder success
        
    else:
        logger.warning(f"Unknown request type received: {request_type}")
        return False

# --- Main Execution --- 
def run_discord_bot():
    if not settings.DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN not found in settings. Cannot start bot.")
        return
        
    try:
        bot.run(settings.DISCORD_TOKEN)
    except discord.errors.LoginFailure:
         logger.error("Failed to log in to Discord. Check your DISCORD_TOKEN.")
    except Exception as e:
         logger.error(f"An error occurred while running the Discord bot: {e}", exc_info=True)

if __name__ == "__main__":
    # Configure logging for the bot
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Optional: Lower discord lib logging level if too verbose
    # logging.getLogger('discord').setLevel(logging.WARNING)
    
    print("Attempting to run the Discord bot...")
    run_discord_bot() 