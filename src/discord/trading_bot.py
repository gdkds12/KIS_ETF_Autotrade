import discord
from discord.ext import commands
import logging
import asyncio
from datetime import datetime, timezone
import uuid
from src.config import settings
from src.db.models import SessionLocal, TradingSession

logger = logging.getLogger(__name__)

class TradingBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=discord.Intents.all())
        self.active_sessions = {}
        self.db_session_factory = SessionLocal

    # ... (이전 TradingBot 클래스의 모든 메서드와 유틸 함수 이곳에 복사)

# --- Bot Instance ---
bot = TradingBot()

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
