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

from src.config import settings
# Placeholder for backend client - might replace with direct agent calls or specific LLM client
# from some_backend_client import BackendClient 
from src.db.models import SessionLocal, TradingSession, SessionLog # DB Models
from sqlalchemy import select

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Configuration --- 
DISCORD_TOKEN = settings.DISCORD_TOKEN
GUILD_ID = None # Optional: Specify guild ID for faster command registration
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

# --- Bot Class --- 
class TradingBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=INTENTS) # Prefix not really used for slash commands
        self.active_sessions = {} # thread_id: {user_id, last_interaction_time, llm_session_id, etc.}
        self.db_session_factory = SessionLocal # Store factory
        # self.backend_client = BackendClient() # Placeholder for backend communication

    async def setup_hook(self):
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
              
    async def get_openai_response(self, session_info: dict, user_message: str) -> tuple[str, dict | None]:
        """Get response from OpenAI main tier model, managing conversation history."""
        if not openai_client:
            return "(죄송합니다, OpenAI API가 설정되지 않아 응답할 수 없습니다.)", None

        llm_session_id = session_info['llm_session_id']
        message_history = session_info.get('message_history', [])

        # Add user message to history
        message_history.append({"role": "user", "content": user_message})
        
        # System prompt defining the bot's role and desired output format
        system_prompt = (
            "당신은 한국 ETF 시장에 대한 금융 도우미 AI입니다. 사용자의 질문에 답변하고, 필요시 KIS API 형식에 맞는 주문 제안을 할 수 있습니다." 
            "시장 분석, 종목 정보 제공, 간단한 계산 등을 수행합니다. "
            "투자 관련 조언은 제공하지만, 최종 결정은 사용자의 책임임을 명시해야 합니다."
            "만약 매수 또는 매도 주문을 제안해야 한다면, 반드시 다음 JSON 형식으로 제안 내용을 응답 끝에 포함시키세요: "
            '\n{\n  "suggested_order": {\n    "symbol": "종목코드 (예: 069500)",\n    "action": "buy 또는 sell",\n    "quantity": 주문수량 (정수),\n    "price": 주문가격 (지정가=실제가격, 시장가=0)\n  }\n}'
            "주문 제안이 없다면, JSON 부분을 포함하지 마세요."
            "모든 답변은 한국어로 제공하세요."
        )
        
        messages = [
            {"role": "system", "content": system_prompt}
        ] + message_history[-10:] # Keep last 10 messages for context (adjust as needed)
        
        response_text = "(오류 발생)"
        suggested_order = None
        
        try:
            logger.info(f"Sending request to OpenAI model: {settings.LLM_MAIN_TIER_MODEL} for session {llm_session_id}")
            completion = await openai_client.chat.completions.create(
                model=settings.LLM_MAIN_TIER_MODEL,
                messages=messages,
                temperature=0.7, # Adjust creativity/factuality
                max_tokens=1000
            )
            response_text = completion.choices[0].message.content
            logger.info(f"Received response from OpenAI for session {llm_session_id}")

            # Add AI response to history
            message_history.append({"role": "assistant", "content": response_text})
            session_info['message_history'] = message_history # Update session state

            # --- 주문 제안 JSON 파싱 시도 --- 
            try:
                # 응답 텍스트에서 JSON 부분만 추출 시도
                json_start = response_text.rfind('{\n  "suggested_order":')
                if json_start != -1:
                    json_part = response_text[json_start:]
                    # JSON 앞뒤의 불필요한 텍스트 제거 (모델이 정확히 형식 따르지 않을 경우 대비)
                    # json_part = json_part.strip().replace('\'', '"') # 작은따옴표 처리 등
                    try:
                         order_data = json.loads(json_part)
                         if "suggested_order" in order_data:
                             suggested_order = order_data["suggested_order"]
                             # 간단한 유효성 검사
                             if all(k in suggested_order for k in ['symbol', 'action', 'quantity', 'price']) \
                                and isinstance(suggested_order['quantity'], int) \
                                and isinstance(suggested_order['price'], int): 
                                 logger.info(f"Parsed suggested order: {suggested_order}")
                                 # 응답 텍스트에서 JSON 부분 제거 (선택 사항)
                                 # response_text = response_text[:json_start].strip()
                             else:
                                 logger.warning(f"Parsed JSON for order is incomplete or invalid: {suggested_order}")
                                 suggested_order = None
                    except json.JSONDecodeError as json_e:
                        logger.warning(f"Failed to parse JSON from LLM response: {json_e}. Response part: {json_part[:100]}...")
                        suggested_order = None
            except Exception as parse_e:
                 logger.error(f"Error during suggested order parsing: {parse_e}", exc_info=True)
                 suggested_order = None
                 
        except RateLimitError as e:
             logger.error(f"OpenAI Rate Limit Exceeded: {e}")
             response_text = "(현재 요청량이 많아 잠시 후 다시 시도해주세요.)"
        except APIError as e:
             logger.error(f"OpenAI API Error: {e}")
             response_text = f"(API 오류 발생: {e})"
        except Exception as e:
             logger.error(f"Unexpected error interacting with OpenAI: {e}", exc_info=True)
             response_text = "(AI 응답 생성 중 오류가 발생했습니다.)"
             
        return response_text, suggested_order

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
                    # Get response from OpenAI
                    response_text, suggested_order = await self.get_openai_response(session_info, message.content)
                    
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
                    
                    await message.channel.send(response_text, view=view)
                    logger.info(f"[Session:{message.channel.id}] Sent response to user.")

                except Exception as e:
                    logger.error(f"[Session:{message.channel.id}] Error processing message: {e}", exc_info=True)
                    await message.channel.send("죄송합니다, 메시지를 처리하는 중 오류가 발생했습니다.")
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

# --- Main Execution --- 
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in environment settings.")
    elif not openai_client and not genai:
        print("Warning: Neither OpenAI nor Google API keys are set. LLM features will be limited.")
        # Decide if the bot should run without LLM features or exit
        # exit()
        bot.run(DISCORD_TOKEN) # Run anyway, but LLM calls will fail gracefully
    else:
        bot.run(DISCORD_TOKEN) 