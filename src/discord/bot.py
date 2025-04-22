import logging
import uuid
from datetime import datetime, timezone
import pytz
import discord  # 하위 모듈 포함
from discord import ButtonStyle, ui, Interaction, Embed, app_commands
from discord.ui import View, Button
from discord.ext import commands
from src.agents.finnhub_client import FinnhubClient
from src.agents.memory_rag import MemoryRAG
from qdrant_client import QdrantClient
from src.config import settings
from src.agents.orchestrator import Orchestrator
from src.brokers.kis import KisBroker
from src.db.models import TradingSession, SessionLocal
from src.utils.registry import set_orchestrator, COMMANDS, ORCHESTRATOR
from src.utils.discord_utils import DiscordRequestType
from src.discord.utils import send_discord_request
import asyncio
import json
from src.utils.azure_openai import azure_chat_completion  # REST-based AI 채팅 지원

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# 상태 아이콘 맵
ICON = {
    'pending': '⚪️',
    'in_progress': '🟡',
    'completed': '✅',
    'error': '❌',
}

GUILD_ID = 1363088557517967582

class TradeCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    async def _update_tool_status(self, thread_id: int, tool_name: str, status: str, description: str = None):
        sess = self.bot.active_sessions.get(thread_id)
        if not sess:
            return
        channel = self.bot.get_channel(thread_id)
        msg = await channel.fetch_message(sess['status_msg_id'])
        embed = msg.embeds[0]
        for i, field in enumerate(embed.fields):
            if field.name == tool_name:
                embed.set_field_at(i, name=tool_name, value=ICON[status], inline=False)
                break
        if description:
            embed.description = description
        await msg.edit(embed=embed)

    @app_commands.command(name="balance", description="현재 계좌의 잔고(예수금, 총자산 등)를 조회합니다.")
    async def balance(self, interaction: Interaction):
        thread_id = interaction.channel.id
        await self._update_tool_status(thread_id, "잔고조회", "in_progress")
        orchestrator = self.bot.get_orchestrator()
        if not orchestrator:
            await interaction.response.send_message("Orchestrator가 준비되지 않았습니다.")
            await self._update_tool_status(thread_id, "잔고조회", "error")
            return
        try:
            balance = orchestrator.broker.get_balance()
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
            await self._update_tool_status(thread_id, "잔고조회", "completed")
        except Exception as e:
            logger.error(f"/balance command error: {e}", exc_info=True)
            await self._update_tool_status(thread_id, "잔고조회", "error")
            await interaction.response.send_message(f"잔고 조회 중 오류가 발생했습니다: {e}")

    @app_commands.command(name="trade", description="새로운 트레이딩 세션을 시작합니다.")
    async def trade(self, interaction: Interaction):
        user = interaction.user
        logger.info(f"Received /trade command from {user.id}")

        thread_name = f"Trade Session - {user.display_name} ({datetime.now().strftime('%H:%M')})"
        thread = await interaction.channel.create_thread(name=thread_name, auto_archive_duration=1440)
        logger.info(f"Created thread {thread.id} for user {user.id}")

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
        await interaction.response.send_message(f"새로운 트레이딩 세션 스레드를 시작했습니다: {thread.mention}")

        status_embed = Embed(title="🔄 작업 상태", color=0x5865f2)
        for stage in ("잔고조회", "주문검증", "주문실행"):
            status_embed.add_field(name=stage, value=ICON['pending'], inline=False)
        status_msg = await thread.send(embed=status_embed)
        self.bot.active_sessions[thread.id]['status_msg_id'] = status_msg.id

    @app_commands.command(name="market_summary", description="시장 동향을 요약하여 보여줍니다.")
    async def market_summary(self, interaction: Interaction, query: str):
        await interaction.response.defer()
        sent_msg = await interaction.followup.send("🟡 기사 수집 중...")
        orchestrator = self.bot.get_orchestrator()
        if not orchestrator:
            return await interaction.followup.send("Orchestrator가 준비되지 않았습니다.")

        # 상태 알림 콜백 설정
        loop = asyncio.get_running_loop()
        def status_notifier(key: str):
            mapping = {
                "기사 수집 중":      "🟡 기사 수집 중...",
                "기사 수집 완료":    "✅ 기사 수집 완료!",
                "기사 크롤링 중":    "🟡 기사 크롤링 중...",
                "기사 크롤링 완료":  "✅ 기사 크롤링 완료!",
                "요약 중":          "🟡 요약 중...",
                "요약 완료":        "✅ 요약 완료!"
            }
            if content := mapping.get(key):
                asyncio.run_coroutine_threadsafe(sent_msg.edit(content=content), loop)
        orchestrator.info_crawler.status_notifier = status_notifier

        summary = await loop.run_in_executor(None, orchestrator.info_crawler.get_market_summary, query)
        await sent_msg.edit(content=summary)
        orchestrator.info_crawler.status_notifier = None

    @app_commands.command(name="confirm_order", description="주문을 확인하고 실행합니다.")
    async def confirm_order(self, interaction: Interaction, order_details: str):
        view = OrderConfirmationView(bot=self.bot, session_thread_id=interaction.channel.id, order_details=order_details)
        await interaction.response.send_message("주문을 확인하고 실행하려면 버튼을 눌러주세요.", view=view)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        channel_id = message.channel.id
        if channel_id not in self.bot.active_sessions:
            return

        session = self.bot.active_sessions[channel_id]
        history = session.get("history") or []

        # system 프롬프트 및 시각 지정
        now_kst = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")
        base_system = ("당신은 주식 시장을 위한 전문 금융 뉴스 및 트레이딩 어시스턴트입니다.\n"
                       "함수 호출은 지정된 JSON 형식으로 응답하세요.")
        history = [{"role": "system", "content": f"현재 시각은 {now_kst} (KST)입니다."},
                   {"role": "system", "content": base_system}] + history + [{"role": "user", "content": message.content}]

        loop = asyncio.get_running_loop()
        # 함수 스펙 목록
        from src.utils.registry import COMMANDS
        functions = [fn._oas for fn in COMMANDS.values() if hasattr(fn, '_oas')]

        # 1차 호출: function_call 감지
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: azure_chat_completion(
                    deployment=settings.AZURE_OPENAI_DEPLOYMENT_GPT4,
                    messages=history,
                    max_tokens=1000,
                    temperature=0.7,
                    functions=functions,
                    function_call="auto"
                )
            )
        except Exception as e:
            logger.error(f"azure_chat_completion error: {e}", exc_info=True)
            return
        assistant_msg = resp["choices"][0]["message"]
        function_call = assistant_msg.get("function_call")

        if function_call:
            # 함수 호출 처리
            func_name = function_call["name"]
            args_dict = json.loads(function_call.get("arguments", "{}"))
            from src.utils.registry import COMMANDS as REG
            if func_name not in REG:
                await message.channel.send(f"알 수 없는 함수 호출: {func_name}")
                return
            func = REG[func_name]
            result = await loop.run_in_executor(None, lambda: func(**args_dict))
            result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

            # history 업데이트
            history.extend([
                assistant_msg,
                {"role": "function", "name": func_name, "content": result_str}
            ])
            # 2차 호출: 최종 답변 생성
            resp2 = await loop.run_in_executor(
                None,
                lambda: azure_chat_completion(
                    deployment=settings.AZURE_OPENAI_DEPLOYMENT_GPT4,
                    messages=history,
                    max_tokens=3000,
                    temperature=0.5
                )
            )
            final_content = resp2["choices"][0]["message"]["content"]
            await message.channel.send(final_content)
            history.append({"role": "assistant", "content": final_content})
        else:
            # 함수 호출 없이 바로 응답
            content = assistant_msg.get("content", "")
            history.append(assistant_msg)
            await message.channel.send(content)

        # 세션 업데이트
        session["history"] = history
        session["last_interaction_time"] = datetime.now()
        self.bot.active_sessions[channel_id] = session


class TradingBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=self._get_intents())
        self.db_session_factory = SessionLocal
        self.active_sessions = {}

    def _get_intents(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        return intents

    async def setup_hook(self):
        logger.info("Initializing Orchestrator...")
        try:
            finnhub_client = FinnhubClient(token=settings.FINNHUB_API_KEY)
            memory_rag = MemoryRAG(
                db_session_factory=self.db_session_factory,
                qdrant_client=QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY),
                llm_model=settings.AZURE_OPENAI_DEPLOYMENT_GPT4
            )
            base_url = settings.KIS_VIRTUAL_URL if settings.KIS_VIRTUAL_ACCOUNT else settings.BASE_URL
            broker = KisBroker(
                app_key=settings.APP_KEY,
                app_secret=settings.APP_SECRET,
                base_url=base_url,
                virtual_account=settings.KIS_VIRTUAL_ACCOUNT,
                cano=settings.CANO,
                acnt_prdt_cd=settings.ACNT_PRDT
            )
            orchestrator_instance = Orchestrator(
                broker=broker,
                db_session_factory=self.db_session_factory,
                qdrant_client=QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY),
                finnhub_client=finnhub_client,
                memory_rag=memory_rag
            )
            set_orchestrator(orchestrator_instance)
            logger.info("Orchestrator initialized and set.")
        except Exception as e:
            logger.critical(f"Failed to initialize Orchestrator: {e}", exc_info=True)

        await self.add_cog(TradeCog(self))
        guild = discord.Object(id=GUILD_ID)
        await self.tree.sync(guild=guild)
        logger.info("Slash commands registered.")

    async def on_ready(self):
        logger.info(f"{self.user} connected to Discord!")

    def get_orchestrator(self):
        return ORCHESTRATOR


class OrderConfirmationView(View):
    def __init__(self, bot: TradingBot, session_thread_id: int, order_details: str):
        super().__init__(timeout=60 * 10)
        self.bot = bot
        self.session_thread_id = session_thread_id
        self.order_details = order_details
        self.confirmed = False

    @ui.button(label="✅ 주문 실행", style=ButtonStyle.green)
    async def confirm_button(self, interaction: Interaction, button: Button):
        session_info = self.bot.active_sessions.get(self.session_thread_id)
        if not session_info or interaction.user.id != session_info['user_id']:
            return await interaction.response.send_message("세션을 시작한 사용자만 주문을 실행할 수 있습니다.", ephemeral=True)
        if self.confirmed:
            return await interaction.response.send_message(embed=Embed(
                title="오류", description="이미 처리된 주문입니다.", color=0xe74c3c, timestamp=datetime.now(timezone.utc)
            ), ephemeral=True)
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
        logger.info(f"Order executed: {self.order_details}")

    @ui.button(label="❌ 취소", style=ButtonStyle.red)
    async def cancel_button(self, interaction: Interaction, button: Button):
        session_info = self.bot.active_sessions.get(self.session_thread_id)
        if not session_info or interaction.user.id != session_info['user_id']:
            return await interaction.response.send_message(embed=Embed(
                title="오류", description="세션을 시작한 사용자만 취소할 수 있습니다.", color=0xe74c3c, timestamp=datetime.now(timezone.utc)
            ), ephemeral=True)
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
        logger.info(f"Order canceled: {self.order_details}")


bot = TradingBot()
bot.run(settings.DISCORD_TOKEN)
