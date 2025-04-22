import logging
import uuid
from datetime import datetime, timezone
import pytz
import discord  # import discord module itself for Intents and submodules
from discord import ButtonStyle, ui  # ButtonStyle and ui module
from discord.ui import View, Button  # View and Button for UI components
from discord.ext import commands
from discord import Interaction, Embed
from discord import app_commands
from src.agents.finnhub_client import FinnhubClient
from src.agents.memory_rag import MemoryRAG
from qdrant_client import QdrantClient
from src.config import settings
from src.agents.orchestrator import Orchestrator
from src.brokers.kis import KisBroker
from src.db.models import TradingSession, SessionLocal
from src.utils.registry import set_orchestrator
from src.utils.discord_utils import DiscordRequestType
from src.discord.utils import send_discord_request
import asyncio
from src.utils.azure_openai import azure_chat_completion  # REST-based AI chat support

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

GUILD_ID = 1363088557517967582

logger = logging.getLogger(__name__)

class TradeCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="balance", description="현재 계좌의 잔고(예수금, 총자산 등)를 조회합니다.")
    async def balance(self, interaction: Interaction):
        orchestrator = self.bot.get_orchestrator()
        if not orchestrator:
            await interaction.response.send_message("Orchestrator가 준비되지 않았습니다.")
            return
        try:
            balance = orchestrator.broker.get_balance()
            # 예시: 주요 정보만 Embed로 표시
            embed = Embed(
                title="💰 계좌 잔고",
                color=0x2ecc71,
                timestamp=datetime.now(timezone.utc)
            )
            available_cash = balance.get('available_cash', 'N/A')
            asset_value    = balance.get('total_asset_value', 'N/A')
            total_pnl      = balance.get('total_pnl', 'N/A')
            pnl_percent    = balance.get('total_pnl_percent', 'N/A')
            embed.add_field(name="예수금", value=(f"{available_cash:,}원" if isinstance(available_cash, (int, float)) else f"{available_cash}원"), inline=False)
            embed.add_field(name="총자산", value=(f"{asset_value:,}원" if isinstance(asset_value, (int, float)) else f"{asset_value}원"), inline=False)
            embed.add_field(name="총손익", value=(f"{total_pnl:,}원" if isinstance(total_pnl, (int, float)) else f"{total_pnl}원"), inline=False)
            embed.add_field(name="총손익률", value=(f"{pnl_percent}%" if isinstance(pnl_percent, (int, float)) else f"{pnl_percent}%"), inline=False)
            await interaction.response.send_message(embed=embed)
        except Exception as e:
            logger.error(f"/balance command error: {e}", exc_info=True)
            await interaction.response.send_message(f"잔고 조회 중 오류가 발생했습니다: {e}")

    @app_commands.command(name="trade", description="새로운 트레이딩 세션을 시작합니다.")
    async def trade(self, interaction: Interaction):
        user = interaction.user
        logger.info(f"Received /trade command from {user.id}")

        # 새 트레이딩 세션 생성
        thread_name = f"Trade Session - {user.display_name} ({datetime.now().strftime('%H:%M')})"
        thread = await interaction.channel.create_thread(name=thread_name, auto_archive_duration=1440)
        logger.info(f"Created thread {thread.id} for user {user.id}")

        # DB에 트레이딩 세션 정보 저장
        session_uuid = str(uuid.uuid4())
        db = self.bot.db_session_factory()
        try:
            new_session = TradingSession(
                session_uuid=session_uuid,
                discord_thread_id=str(thread.id),
                discord_user_id=str(user.id)
            )
            db.add(new_session)
            db.commit()
            logger.info(f"Created TradingSession entry in DB for UUID {session_uuid}")
        except Exception as e:
            logger.error(f"Failed to create TradingSession in DB: {e}", exc_info=True)
            db.rollback()
            await thread.delete()
            await interaction.response.send_message("세션 시작 중 오류가 발생했습니다.", ephemeral=True)
            return
        finally:
            db.close()

        self.bot.active_sessions[thread.id] = {
            'user_id': user.id,
            'start_time': datetime.now(),
            'last_interaction_time': datetime.now(),
            'llm_session_id': session_uuid
        }

        # 안내 임베드 메시지 생성 (들여쓰기 수정)
        embed = Embed(
            title="📢 트레이딩 세션 시작 안내",
            description=(
                f"{user.mention}님, 새로운 트레이딩 세션이 시작되었습니다!\n\n"
                "아래 명령어와 버튼을 통해 주문을 생성, 확인, 실행할 수 있습니다.\n"
                "- `/confirm_order` : 주문 확인 및 실행\n"
                "- `/balance` : 계좌 잔고 조회\n\n"
                "주문 실행/취소 시 반드시 안내 메시지와 버튼을 확인해 주세요."
            ),
            color=0x5865f2,
            timestamp=datetime.now(timezone.utc)
        )
        embed.set_footer(
            text="KIS ETF Autotrade • Powered by AI",
            icon_url="https://cdn-icons-png.flaticon.com/512/4712/4712032.png"
        )
        await thread.send(embed=embed)
        await interaction.response.send_message(
            f"새로운 트레이딩 세션 스레드를 시작했습니다: {thread.mention}"
        )

    @app_commands.command(name="market_summary", description="시장 동향을 요약하여 보여줍니다.")
    async def market_summary(self, interaction: Interaction, query: str):
        orchestrator = self.bot.get_orchestrator()
        if not orchestrator:
            await interaction.response.send_message("Orchestrator가 준비되지 않았습니다.")
            return

        # get_market_summary() 는 동기 함수이므로 블로킹 방지
        import asyncio
        loop = asyncio.get_running_loop()
        market_summary = await loop.run_in_executor(
            None,
            orchestrator.info_crawler.get_market_summary,
            query
        )
        embed = Embed(
            title="📊 시장 동향",
            description=market_summary,
            color=0x3498db,
            timestamp=datetime.now(timezone.utc)
        )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="confirm_order", description="주문을 확인하고 실행합니다.")
    async def confirm_order(self, interaction: Interaction, order_details: str):
        view = OrderConfirmationView(bot=self.bot, session_thread_id=interaction.channel.id, order_details=order_details)
        await interaction.response.send_message("주문을 확인하고 실행하려면 버튼을 눌러주세요.", view=view)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        logger.debug(f"[on_message] Received message in channel {message.channel.id} from {message.author.id}: {message.content!r}")
        # AI chat in active session threads
        if message.author.bot:
            return
        channel_id = message.channel.id
        if channel_id not in self.bot.active_sessions:
            return

        session = self.bot.active_sessions[channel_id]
        # Initialize or retrieve conversation history with system prompt for function usage
        history = session.get("history", [{"role": "system", "content":
            "당신은 주식 시장을 위한 전문 금융 뉴스 및 트레이딩 어시스턴트입니다.\n"
            "- get_balance(): 내 계좌의 잔고와 총 자산을 조회합니다.\n"
            "- get_positions(): 현재 보유 중인 종목 목록을 조회합니다.\n"
            "- get_market_summary(query: str): 입력한 영어 (query)에 맞는 시장 요약 정보를 제공합니다.\n"
            "- search_symbols(query: str): 종목명 또는 심볼로 주식/ETF를 검색합니다. 심볼(티커, 종목코드 등)이 확실하지 않을 때 이 함수를 먼저 사용하세요.\n"
            "- get_quote(symbol: str): 특정 주식/ETF의 현재 시세를 조회합니다. 정확한 심볼(티커, 종목코드 등)을 알고 있다면 이 함수로 바로 시세를 조회하세요.\n"
            "- get_historical_data(symbol: str, timeframe: str, start_date: str, end_date: str, period: str): 과거 가격 데이터를 조회합니다.\n"
            "- order_cash(symbol: str, quantity: str, price: str, order_type: str, buy_sell_code: str): 현금 주문을 실행합니다.\n"
            "- get_overseas_trading_status(): 해외 주식 거래 가능 여부를 확인합니다.\n"
            "모든 시장 요약(get_market_summary)은 영어 쿼리로만 동작합니다다.\n"
            "시장 조사를를 요청받으면, 최신 기사와 신뢰성 있는 정보를 바탕으로 중복 없이 핵심만 요약하고, 서로 다른 의견이 있으면 명확히 언급하며, 날짜/시간이 있다면 최신 정보에 더 가중치를 두고, 명확한 시장 방향성이 보이면 결론도 포함하며, 기사 출처와 URL을 명시하여 반드시 한국어로 설명명하세요.\n"
            "함수 호출이 필요할 경우 반드시 다음과 같은 JSON 형식으로 답변하세요:\n"
            '{"function": "<함수명>", "arguments": {...}}\n'
            "주식/ETF의 시세를 조회할 때, 입력이 정확한 심볼(티커, 종목코드 등)인지 먼저 확인하세요. 심볼이 확실하면 get_quote를 바로 호출하고, 그렇지 않으면 search_symbols로 심볼을 찾은 후 get_quote를 호출하세요.\n"
            "그 외에는 자연스럽게 한국어로 답변하세요."
        }])
        history.append({"role": "user", "content": message.content})

        logger.debug(f"[on_message] 1st call to azure_chat_completion (detect function_call)")
        loop = asyncio.get_running_loop()
        from src.utils import registry
        # 모든 등록 함수의 function spec 추출
        functions = [fn._oas for fn in registry.COMMANDS.values() if hasattr(fn, '_oas')]
        # 1st call: detect function_call via named args using functools.partial (workaround run_in_executor)
        import functools
        fn_call = functools.partial(
            azure_chat_completion,
            deployment=settings.AZURE_OPENAI_DEPLOYMENT_GPT4,
            messages=history,
            max_tokens=1000,
            temperature=0.7,
            functions=functions,
            function_call="auto"
        )
        logger.debug(f"[on_message] 1st call azure_chat_completion kwargs: {fn_call.keywords}")
        try:
            resp = await loop.run_in_executor(None, fn_call)
        except Exception as e:
            logger.error(f"[on_message] azure_chat_completion error: {e}", exc_info=True)
            raise
        assistant_msg = resp["choices"][0]["message"]

        # ---------- 함수 호출인지 확인 ----------
        import json as _json
        tool_calls = assistant_msg.get("tool_calls")
        function_call = assistant_msg.get("function_call")
        if tool_calls or function_call:
            if tool_calls:
                # 최신 스펙 (role: tool/tool_call_id)
                tool_call = tool_calls[0]
                func_name = tool_call["function"].get("name") or tool_call.get("name")
                args = tool_call.get("arguments")
                tool_call_id = tool_call.get("id")
                args_dict = _json.loads(args) if isinstance(args, str) else args
            else:
                # 구버전 (role: function)
                func_name = function_call["name"]
                args = function_call["arguments"]
                tool_call_id = assistant_msg.get("id")
                args_dict = _json.loads(args) if isinstance(args, str) else args
            logger.info(f"[on_message] Detected function/tool_call: {func_name} {args_dict}")

            from src.utils import registry
            func = registry.COMMANDS.get(func_name)
            if not func:
                await message.channel.send(f"알 수 없는 함수 호출: {func_name}")
                return

            # 동기 함수 실행은 executor 로
            # 키워드 인자를 functools.partial을 사용하여 함수에 바인딩
            import functools
            bound_func = functools.partial(func, **args_dict)
            result = await loop.run_in_executor(None, bound_func)
            # content는 반드시 문자열이어야 함
            result_str = result if isinstance(result, str) else _json.dumps(result, ensure_ascii=False, default=str)

            # assistant 호출 메시지 + tool/function 결과 메시지를 history에 추가
            if tool_calls:
                # 최신 스펙
                history.extend([
                    assistant_msg,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": func_name,
                        "content": result_str
                    }
                ])
            else:
                # 구버전
                history.extend([
                    assistant_msg,
                    {
                        "role": "function",
                        "name": func_name,
                        "content": result_str
                    }
                ])

            logger.debug(f"[on_message] 2nd call to azure_chat_completion (final answer)")
            # 두 번째 호출에는 tools/tool_choice/functions/function_call 인자 절대 전달하지 않음
            resp2 = await loop.run_in_executor(
                None,
                azure_chat_completion,
                settings.AZURE_OPENAI_DEPLOYMENT_GPT4,
                history,
                1000,
                0.5
            )
            final_answer = resp2["choices"][0]["message"]["content"]
            await message.channel.send(final_answer)

            # 대화 이력 저장
            history.append({"role": "assistant", "content": final_answer})
        else:
            # 함수 호출이 아니면 바로 내용 출력
            content = assistant_msg.get("content", "")
            history.append(assistant_msg)
            await message.channel.send(content)

        # Update session history & last touch
        session["history"] = history
        session["last_interaction_time"] = datetime.now()
        self.bot.active_sessions[channel_id] = session
        logger.debug(f"[on_message] handler finished.")

# 디스코드 봇 클래스 정의
class TradingBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=self._get_intents())
        self.db_session_factory = SessionLocal
        self.active_sessions = {}  # 세션 추적을 위한 저장소

    def _get_intents(self):
        intents = discord.Intents.default()
        intents.message_content = True  # 메시지 내용 읽기
        intents.members = True  # 서버 멤버 관련 기능
        return intents

    async def setup_hook(self):
        # Initialize Orchestrator here and assign to config
        logger.info("Initializing Orchestrator...")
        try:
            # Initialize Finnhub Client
            finnhub_client = FinnhubClient(token=settings.FINNHUB_API_KEY)
            logger.info("Finnhub client initialized successfully.")

            # Initialize MemoryRAG
            memory_rag = MemoryRAG(
                db_session_factory=self.db_session_factory,
                qdrant_client=QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY # API 키가 설정되어 있다면 전달
                ),
                llm_model=settings.AZURE_OPENAI_DEPLOYMENT_GPT4 # Use summary model for RAG
            )
            logger.info(f"MemoryRAG initialized with DB factory, Qdrant, and LLM model: {settings.AZURE_OPENAI_DEPLOYMENT_GPT4}")

            # Initialize KisBroker
            # Determine base_url based on virtual account setting
            base_url = settings.KIS_VIRTUAL_URL if settings.KIS_VIRTUAL_ACCOUNT else settings.BASE_URL
            broker = KisBroker(
                app_key=settings.APP_KEY, 
                app_secret=settings.APP_SECRET, 
                base_url=base_url, # 추가: base_url 전달
                virtual_account=settings.KIS_VIRTUAL_ACCOUNT,
                cano=settings.CANO, # 수정: account_no -> cano
                acnt_prdt_cd=settings.ACNT_PRDT # 수정: account_prod_code -> acnt_prdt_cd
            )
            # Select the correct URL for logging based on the setting
            active_kis_url = base_url # Use the determined base_url for logging
            logger.info(f"KisBroker initialized for {'Virtual' if settings.KIS_VIRTUAL_ACCOUNT else 'Real'} Trading (URL: {active_kis_url}).")

            # Initialize Orchestrator with all components
            logger.info("Initializing Orchestrator...")
            orchestrator_instance = Orchestrator(
                broker=broker,
                db_session_factory=self.db_session_factory,
                qdrant_client=QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY # API 키가 설정되어 있다면 전달
                ),
                finnhub_client=finnhub_client, # Pass finnhub client
                memory_rag=memory_rag         # Pass memory rag
            )
            logger.info("Orchestrator initialized successfully.")

            # Assign the initialized orchestrator to the config module's variable
            set_orchestrator(orchestrator_instance)
            logger.info("Orchestrator instance assigned to config.ORCHESTRATOR")

        except Exception as e:
            logger.critical(f"Failed to initialize Orchestrator: {e}", exc_info=True)
            # Consider exiting if orchestrator fails to initialize
            # await self.close() # Or raise the exception to stop the bot

        # Cog 등록 (slash commands)
        await self.add_cog(TradeCog(self))
        guild = discord.Object(id=GUILD_ID)
        # 길드 전용으로만 커맨드 등록 (글로벌 sync 호출 X)
        await self.tree.sync(guild=guild)
        logger.info(f"[GUILD {GUILD_ID}] Slash commands registered (guild only). Logged in as {self.user} (ID: {self.user.id})")
        logger.info("Bot is ready.")

    async def on_ready(self):
        logger.info(f"{self.user} has connected to Discord!")

    def get_orchestrator(self):
        """Orchestrator 인스턴스를 반환"""
        from src.utils.registry import ORCHESTRATOR
        return ORCHESTRATOR

# 주문 확인 버튼을 위한 View 클래스
class OrderConfirmationView(View):
    def __init__(self, bot: TradingBot, session_thread_id: int, order_details: str):
        super().__init__(timeout=60 * 10)  # 10분 동안 유효
        self.bot = bot
        self.session_thread_id = session_thread_id
        self.order_details = order_details
        self.confirmed = False

    @ui.button(label="✅ 주문 실행", style=ButtonStyle.green)
    async def confirm_button(self, interaction: Interaction, button: Button):
        session_info = self.bot.active_sessions.get(self.session_thread_id)
        if not session_info or interaction.user.id != session_info['user_id']:
            await interaction.response.send_message("세션을 시작한 사용자만 주문을 실행할 수 있습니다.", ephemeral=True)
            return

        if self.confirmed:
            embed = Embed(
                title="오류",
                description="이미 처리된 주문입니다.",
                color=0xe74c3c,
                timestamp=datetime.now(timezone.utc)
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        self.confirmed = True
        await interaction.response.defer()  # 응답 지연
        # 임베드로 주문 실행 안내
        embed = Embed(
            title="✅ 주문 실행",
            description=f"{interaction.user.mention}님의 주문이 실행되었습니다!",
            color=0x2ecc71,
            timestamp=datetime.now(timezone.utc)
        )
        embed.add_field(name="주문 상세", value=self.order_details, inline=False)
        embed.set_footer(text="KIS ETF Autotrade", icon_url="https://cdn-icons-png.flaticon.com/512/4712/4712032.png")
        await interaction.followup.send(embed=embed)
        logger.info(f"User confirmed order: {self.order_details}")

        # 여기에 주문 실행 로직을 추가합니다. (예: KIS API 호출)

    @ui.button(label="❌ 취소", style=ButtonStyle.red)
    async def cancel_button(self, interaction: Interaction, button: Button):
        session_info = self.bot.active_sessions.get(self.session_thread_id)
        if not session_info or interaction.user.id != session_info['user_id']:
            embed = Embed(
                title="오류",
                description="세션을 시작한 사용자만 취소할 수 있습니다.",
                color=0xe74c3c,
                timestamp=datetime.now(timezone.utc)
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return

        await interaction.response.defer()
        # 임베드로 주문 취소 안내
        embed = Embed(
            title="❌ 주문 취소",
            description=f"{interaction.user.mention}님의 주문 제안이 취소되었습니다.",
            color=0xe74c3c,
            timestamp=datetime.now(timezone.utc)
        )
        embed.add_field(name="주문 상세", value=self.order_details, inline=False)
        embed.set_footer(text="KIS ETF Autotrade", icon_url="https://cdn-icons-png.flaticon.com/512/4712/4712032.png")
        await interaction.followup.send(embed=embed)

        # 취소 처리 (DB 기록, 로그 등)
        logger.info(f"User canceled order confirmation: {self.order_details}")

# 봇 실행
bot = TradingBot()

# 봇 실행
bot.run(settings.DISCORD_TOKEN)
