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

ICON = {
    'pending': '⚪️',
    'in_progress': '🟡',
    'completed': '✅',
    'error': '❌',
}

class TradeCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="market_summary", description="시장 동향을 요약하여 보여줍니다.")
    async def market_summary(self, interaction: Interaction, query: str):
        # Removed: use AI-triggered function call path in on_message instead
        pass

    @app_commands.command(name="confirm_order", description="주문을 확인하고 실행합니다.")
    async def confirm_order(self, interaction: Interaction, order_details: str):
        view = OrderConfirmationView(bot=self.bot, session_thread_id=interaction.channel.id, order_details=order_details)
        await interaction.response.send_message("주문을 확인하고 실행하려면 버튼을 눌러주세요.", view=view)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # 1) 기본 필터링
        if message.author.bot:
            return
        thread_id = message.channel.id
        if thread_id not in self.bot.active_sessions:
            return

        # 2) 대화 이력 구성 (생략)
        #    --- (기존 history 작성 코드) ---

        # 3) AI 함수 호출 감지
        from src.utils import registry
        import functools, json
        resp = await run_azure_completion_and_detect_function_call(history, registry.COMMANDS)
        func_call = resp["choices"][0]["message"].get("function_call")
        if func_call and func_call["name"] == "get_market_summary":
            args = json.loads(func_call["arguments"])

            # 4) 메인 이벤트 루프 캡처 및 상태 메시지 생성
            import asyncio
            loop = asyncio.get_running_loop()
            status_msg = await message.channel.send("⚪️ 시장 요약 시작 대기 중…")

            # 5) 단계별 노티파이어 정의
            def notifier(step: str):
                mapping = {
                    "기사 수집 중":   "🟡 기사 수집 중…",
                    "기사 수집 완료": "✅ 기사 수집 완료!",
                    "기사 크롤링 중": "🟡 기사 크롤링 중…",
                    "기사 크롤링 완료": "✅ 기사 크롤링 완료!",
                    "요약 중":       "🟡 요약 중…",
                    "요약 완료":     "✅ 요약 완료!"
                }
                content = mapping.get(step)
                if content:
                    asyncio.run_coroutine_threadsafe(
                        status_msg.edit(content=content),
                        loop
                    )

            # 6) InfoCrawler에 콜백 등록
            crawler = self.bot.get_orchestrator().info_crawler
            crawler.status_notifier = notifier

            # 7) 워커 스레드에서 실제 요약 실행
            summary = await loop.run_in_executor(
                None,
                functools.partial(crawler.get_market_summary, args["user_query"])
            )

            # 8) 콜백 해제 및 최종 결과 업데이트
            crawler.status_notifier = None
            await status_msg.edit(content=summary)
            return

        # 9) 그 외 메시지는 기존 로직대로 처리
        #    --- (기존 function_call / AI 응답 처리 코드) ---

        # 10) 세션 히스토리 갱신
        session = self.bot.active_sessions[thread_id]
        session["last_interaction_time"] = datetime.now()
        self.bot.active_sessions[thread_id] = session

class OrderConfirmationView(View):
    def __init__(self, bot, session_thread_id, order_details):
        super().__init__(timeout=60 * 10)
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
        await interaction.response.defer()
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
        embed = Embed(
            title="❌ 주문 취소",
            description=f"{interaction.user.mention}님의 주문 제안이 취소되었습니다.",
            color=0xe74c3c,
            timestamp=datetime.now(timezone.utc)
        )
        embed.add_field(name="주문 상세", value=self.order_details, inline=False)
        embed.set_footer(text="KIS ETF Autotrade", icon_url="https://cdn-icons-png.flaticon.com/512/4712/4712032.png")
        await interaction.followup.send(embed=embed)
        logger.info(f"User canceled order confirmation: {self.order_details}")

# 봇 실행
# bot = TradingBot()
# bot.run(settings.DISCORD_TOKEN)
