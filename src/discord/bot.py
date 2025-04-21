import logging
import uuid
from datetime import datetime, timezone
import discord  # import discord module itself for Intents and submodules
from discord import ButtonStyle, ui  # ButtonStyle and ui module
from discord.ui import View, Button  # View and Button for UI components
from discord.ext import commands
from discord import Interaction, Embed
from src.config import settings
from src.agents.orchestrator import Orchestrator
from src.brokers.kis import KisBroker
from src.db.models import TradingSession, SessionLocal
from src.utils.registry import set_orchestrator
from src.utils.discord_utils import DiscordRequestType
from src.discord.utils import send_discord_request
import asyncio

logger = logging.getLogger(__name__)

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

        # 명령어 동기화
        await self.tree.sync()
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
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

            # Orchestrator 초기화
            orchestrator = Orchestrator(
                broker=broker,
                db_session_factory=self.db_session_factory
            )
            set_orchestrator(orchestrator)
            logger.info("Orchestrator initialized and connected.")

        except Exception as e:
            logger.error("Error initializing orchestrator", exc_info=True)
            raise e

    async def on_ready(self):
        logger.info(f"{self.user} has connected to Discord!")

    # 새로운 트레이딩 세션을 시작하는 명령어
    @commands.command(name="trade", help="새로운 트레이딩 세션을 시작합니다.")
    async def trade(self, ctx):
        user = ctx.author
        logger.info(f"Received /trade command from {user.id}")

        # 새 트레이딩 세션 생성
        thread_name = f"Trade Session - {user.display_name} ({datetime.now().strftime('%H:%M')})"
        thread = await ctx.channel.create_thread(name=thread_name, auto_archive_duration=1440)  # 24시간 후 자동 아카이브
        logger.info(f"Created thread {thread.id} for user {user.id}")

        # DB에 트레이딩 세션 정보 저장
        session_uuid = str(uuid.uuid4())  # UUID 생성
        db = self.db_session_factory()
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
            await ctx.send("세션 시작 중 오류가 발생했습니다.", ephemeral=True)
            return
        finally:
            db.close()

        self.active_sessions[thread.id] = {
            'user_id': user.id,
            'start_time': datetime.now(),
            'last_interaction_time': datetime.now(),
            'llm_session_id': session_uuid
        }

        await ctx.send(f"새로운 트레이딩 세션 스레드를 시작했습니다: {thread.mention}")

    # 시장 동향 요약 요청 명령어
    @commands.command(name="market_summary", help="시장 동향을 요약하여 보여줍니다.")
    async def market_summary(self, ctx, query: str):
        orchestrator = self.get_orchestrator()

        if not orchestrator:
            await ctx.send("Orchestrator가 준비되지 않았습니다.")
            return

        market_summary = await orchestrator.info_crawler.get_market_summary(query)
        embed = Embed(
            title="📊 시장 동향",
            description=market_summary,
            color=0x3498db,
            timestamp=datetime.now(timezone.utc)
        )

        await ctx.send(embed=embed)

    # 트레이딩 주문 확인
    @commands.command(name="confirm_order", help="주문을 확인하고 실행합니다.")
    async def confirm_order(self, ctx, order_details: str):
        # 사용자 확인을 위한 메시지 전송
        view = OrderConfirmationView(bot=self, session_thread_id=ctx.channel.id, order_details=order_details)
        await ctx.send("주문을 확인하고 실행하려면 버튼을 눌러주세요.", view=view)

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
