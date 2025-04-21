import logging
import uuid
from datetime import datetime, timezone
import discord  # import discord module itself for Intents and submodules
from discord import ButtonStyle, ui  # ButtonStyle and ui module
from discord.ui import View, Button  # View and Button for UI components
from discord.ext import commands
from discord import Interaction, Embed
from discord import app_commands

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

GUILD_ID = 1363088557517967582

from src.config import settings
from src.agents.orchestrator import Orchestrator
from src.brokers.kis import KisBroker
from src.db.models import TradingSession, SessionLocal
from src.utils.registry import set_orchestrator
from src.utils.discord_utils import DiscordRequestType
from src.discord.utils import send_discord_request
import asyncio
from qdrant_client import QdrantClient
from src.utils.azure_openai import azure_chat_completion  # REST-based AI chat support

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
            embed.add_field(name="예수금", value=f"{balance.get('available_cash', 'N/A'):,}원", inline=False)
            embed.add_field(name="총자산", value=f"{balance.get('total_asset_value', 'N/A'):,}원", inline=False)
            embed.add_field(name="총손익", value=f"{balance.get('total_pnl', 'N/A'):,}원", inline=False)
            embed.add_field(name="총손익률", value=f"{balance.get('total_pnl_percent', 'N/A')}%", inline=False)
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

        await interaction.response.send_message(f"새로운 트레이딩 세션 스레드를 시작했습니다: {thread.mention}")

    @app_commands.command(name="market_summary", description="시장 동향을 요약하여 보여줍니다.")
    async def market_summary(self, interaction: Interaction, query: str):
        orchestrator = self.bot.get_orchestrator()
        if not orchestrator:
            await interaction.response.send_message("Orchestrator가 준비되지 않았습니다.")
            return
        market_summary = await orchestrator.info_crawler.get_market_summary(query)
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
            "당신은 아래 도구들을 사용할 수 있는 AI 트레이딩 어시스턴트입니다.\n"
            "- get_balance(): 내 계좌의 잔고와 총 자산을 조회합니다.\n"
            "- get_positions(): 현재 보유 중인 종목 목록을 조회합니다.\n"
            "- get_market_summary(query: str): 입력한 질의(query)에 맞는 시장 요약 정보를 가져옵니다.\n"
            "- search_news(query: str): 최신 뉴스 기사를 검색합니다.\n"
            "- search_symbols(query: str): 종목명 또는 심볼로 주식/ETF를 검색합니다.\n"
            "- search_web(query: str): 일반 웹 검색을 수행합니다.\n"
            "- get_quote(symbol: str): 특정 주식/ETF의 현재 시세를 조회합니다.\n"
            "- get_historical_data(symbol: str, timeframe: str, start_date: str, end_date: str, period: str): 과거 가격 데이터를 조회합니다.\n"
            "- order_cash(symbol: str, quantity: str, price: str, order_type: str, buy_sell_code: str): 현금 주문을 실행합니다.\n"
            "- get_overseas_trading_status(): 해외 주식 거래 가능 여부를 확인합니다.\n"
            "위 도구를 사용해야 할 때는 반드시 다음과 같은 JSON 형식으로 답변하세요:\n"
            "{\"function\": \"<함수명>\", \"arguments\": {...}}\n"
            "그 외에는 자연스럽게 한국어로 답변하세요."
        }])
        history.append({"role": "user", "content": message.content})

        logger.debug(f"[on_message] Calling azure_chat_completion with deployment={settings.AZURE_OPENAI_DEPLOYMENT_GPT35!r} and history length={len(history)}")
        # Call REST-based AI agent asynchronously
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None,
            azure_chat_completion,
            settings.AZURE_OPENAI_DEPLOYMENT_GPT35,
            history,
            1000,
            0.7
        )
        logger.debug(f"[on_message] azure_chat_completion response: {resp}")

        reply = resp["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": reply})

        # Update session history
        session["history"] = history
        self.bot.active_sessions[channel_id] = session

        import json
        from src.utils import registry
        logger.debug(f"[on_message] Checking if reply is function call JSON: {reply}")
        try:
            parsed = json.loads(reply)
            if isinstance(parsed, dict) and "function" in parsed and "arguments" in parsed:
                func_name = parsed["function"]
                args = parsed["arguments"]
                logger.info(f"[on_message] Detected function call: {func_name} with args {args}")
                func = registry.COMMANDS.get(func_name)
                if func:
                    result = func(**args)
                    logger.info(f"[on_message] Function {func_name} executed, result: {result}")
                    # 함수 실행 결과를 assistant 메시지로 history에 추가
                    history.append({"role": "assistant", "content": f"[{func_name} 실행 결과]\n{result}"})
                    session["history"] = history
                    self.bot.active_sessions[channel_id] = session
                    logger.debug(f"[on_message] Calling LLM again to summarize function result.")
                    # 함수 실행 결과를 요약하도록 LLM 재호출
                    resp2 = await loop.run_in_executor(
                        None,
                        azure_chat_completion,
                        settings.AZURE_OPENAI_DEPLOYMENT_GPT35,
                        history,
                        1000,
                        0.7
                    )
                    summary = resp2["choices"][0]["message"]["content"]
                    logger.debug(f"[on_message] LLM summary: {summary}")
                    await message.channel.send(summary)
                else:
                    logger.warning(f"[on_message] Unknown function call: {func_name}")
                    await message.channel.send(f"알 수 없는 함수 호출: {func_name}")
            else:
                await message.channel.send(reply)
        except Exception as e:
            logger.warning(f"[on_message] Exception in function call parsing/execution: {e}")
            await message.channel.send(reply)
        logger.debug(f"[on_message] on_message handler finished.")

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
        # Orchestrator 초기화 및 등록
        await self._initialize_orchestrator()

        # Cog 등록 (slash commands)
        await self.add_cog(TradeCog(self))
        guild = discord.Object(id=GUILD_ID)
        # 길드 전용으로만 커맨드 등록 (글로벌 sync 호출 X)
        await self.tree.sync(guild=guild)
        logger.info(f"[GUILD {GUILD_ID}] Slash commands registered (guild only). Logged in as {self.user} (ID: {self.user.id})")
        logger.info("Bot is ready.")

    async def _initialize_orchestrator(self):
        """Orchestrator를 초기화하고 에이전트와 연결합니다."""
        try:
            # KIS Broker 인스턴스 생성
            broker = KisBroker(
                app_key=settings.APP_KEY,
                app_secret=settings.APP_SECRET,
                base_url=settings.BASE_URL,
                cano=settings.CANO,
                acnt_prdt_cd=settings.ACNT_PRDT
            )

            # Qdrant client 인스턴스 생성
            qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )

            # Orchestrator 초기화
            orchestrator = Orchestrator(
                broker=broker,
                db_session_factory=self.db_session_factory,
                qdrant_client=qdrant_client
            )
            set_orchestrator(orchestrator)
            logger.info("Orchestrator initialized and connected.")

        except Exception as e:
            logger.error("Error initializing orchestrator", exc_info=True)
            raise e

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
            await interaction.response.send_message("이미 처리된 주문입니다.", ephemeral=True)
            return

        self.confirmed = True
        await interaction.response.defer()  # 응답 지연
        await interaction.followup.send(f"{interaction.user.mention} 주문을 실행합니다... {self.order_details}")
        logger.info(f"User confirmed order: {self.order_details}")

        # 여기에 주문 실행 로직을 추가합니다. (예: KIS API 호출)

    @ui.button(label="❌ 취소", style=ButtonStyle.red)
    async def cancel_button(self, interaction: Interaction, button: Button):
        session_info = self.bot.active_sessions.get(self.session_thread_id)
        if not session_info or interaction.user.id != session_info['user_id']:
            await interaction.response.send_message("세션을 시작한 사용자만 취소할 수 있습니다.", ephemeral=True)
            return

        await interaction.response.defer()
        await interaction.followup.send("주문 제안이 취소되었습니다.")

        # 취소 처리 (DB 기록, 로그 등)
        logger.info(f"User canceled order confirmation: {self.order_details}")

# 봇 실행
bot = TradingBot()

# 봇 실행
bot.run(settings.DISCORD_TOKEN)
