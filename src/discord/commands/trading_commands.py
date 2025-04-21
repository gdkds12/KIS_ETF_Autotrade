from discord import app_commands, Interaction
import asyncio, json, aiohttp, uuid, logging
from datetime import datetime
from src.discord.trading_bot import bot
from src.db.models import TradingSession

logger = logging.getLogger(__name__)

@bot.tree.command(name="trade", description="AI 트레이딩 세션을 시작합니다.")
async def trade_command(interaction: Interaction):
    user = interaction.user
    channel = interaction.channel
    logger.info(f"Received /trade command from {user.id} in channel {channel.id}")
    if not hasattr(channel, "create_thread"):
        await interaction.response.send_message("텍스트 채널에서만 /trade 명령을 사용할 수 있습니다.", ephemeral=True)
        return
    try:
        thread_name = f"🤖 Trade Session - {user.display_name} ({datetime.now().strftime('%H:%M')})"
        thread = await channel.create_thread(name=thread_name, auto_archive_duration=1440)
        logger.info(f"Created thread {thread.id} for user {user.id}")
        initial_message = (f"{user.mention}, 안녕하세요! 👋\n"
                           f"이 스레드에서 ETF 트레이딩 관련 질문을 하거나 시장 정보를 요청할 수 있습니다.\n"
                           f"(예: \"오늘 시장 요약해줘\", \"KODEX 200 전망 알려줘\")\n\n"
                           f"*참고: 저는 실제 주문을 제안할 수 있으며, ✅ 버튼을 눌러 실행을 요청할 수 있습니다.*\n"
                           f"*30분 동안 활동이 없으면 스레드는 자동으로 종료됩니다.*")
        await thread.send(initial_message)
        session_uuid = str(uuid.uuid4())
        db = bot.db_session_factory()
        try:
            new_session_db = TradingSession(
                session_uuid=session_uuid,
                discord_thread_id=str(thread.id),
                discord_user_id=str(user.id)
            )
            db.add(new_session_db)
            db.commit()
            logger.info(f"Created TradingSession entry in DB for UUID {session_uuid}")
        except Exception as e:
            logger.error(f"Failed to create TradingSession in DB for thread {thread.id}: {e}", exc_info=True)
            db.rollback()
            await thread.delete()
            await interaction.response.send_message("세션 생성 중 DB 오류가 발생했습니다.", ephemeral=True)
            return
        finally:
            db.close()
        bot.active_sessions[thread.id] = {
            'user_id': user.id,
            'channel_id': channel.id,
            'start_time': asyncio.get_event_loop().time(),
            'last_interaction_time': asyncio.get_event_loop().time(),
            'llm_session_id': session_uuid,
            'message_history': []
        }
        logger.info(f"Started active session tracking for UUID {session_uuid} (Thread: {thread.id})")
        await interaction.response.send_message(f"새로운 트레이딩 세션 스레드를 시작했습니다: {thread.mention}", ephemeral=True)
    except Exception as e:
        logger.error(f"Error handling /trade command: {e}", exc_info=True)
        if not interaction.response.is_done():
            await interaction.response.send_message("세션 시작 중 오류가 발생했습니다.", ephemeral=True)
        else:
            try:
                await interaction.followup.send("세션 시작 중 오류가 발생했습니다.", ephemeral=True)
            except Exception as followup_e:
                logger.error(f"Failed to send followup error message: {followup_e}")

@bot.tree.command(name="run_cycle", description="수동으로 일일 자동매매 사이클을 시작합니다.")
async def run_cycle_command(interaction: Interaction):
    user = interaction.user
    logger.info(f"Received /run_cycle command from {user.id} ({user.name})...")
    fastapi_trigger_endpoint = "http://localhost:8000/trigger_cycle"
    await interaction.response.defer(ephemeral=True)
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
