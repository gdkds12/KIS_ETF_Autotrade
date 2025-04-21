from discord import app_commands, Interaction
import asyncio, json
from src.utils.registry import COMMANDS
from src.utils import registry
from src.discord.bot import bot

@bot.tree.command(name="get_balance", description="현재 계좌 예수금·총자산 조회")
async def get_balance_command(interaction: Interaction):
    if registry.ORCHESTRATOR is None:
        await interaction.response.send_message("오류: Orchestrator가 준비되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: COMMANDS['get_balance']())
    await interaction.followup.send(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```", ephemeral=True)

@bot.tree.command(name="get_positions", description="보유 포지션 조회")
async def get_positions_command(interaction: Interaction):
    if registry.ORCHESTRATOR is None:
        await interaction.response.send_message("오류: Orchestrator가 준비되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: COMMANDS['get_positions']())
    await interaction.followup.send(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```", ephemeral=True)

@bot.tree.command(name="market_summary", description="시장 동향 요약")
@app_commands.describe(query="요약할 시장 동향 쿼리")
async def market_summary_command(interaction: Interaction, query: str):
    if registry.ORCHESTRATOR is None:
        await interaction.response.send_message("오류: Orchestrator가 준비되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: COMMANDS['get_market_summary'](query=query))
    await interaction.followup.send(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```", ephemeral=True)

@bot.tree.command(name="search_news", description="최신 뉴스 검색")
@app_commands.describe(query="검색할 뉴스 키워드")
async def search_news_command(interaction: Interaction, query: str):
    if registry.ORCHESTRATOR is None:
        await interaction.response.send_message("오류: Orchestrator가 준비되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: COMMANDS['search_news'](query=query))
    await interaction.followup.send(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```", ephemeral=True)

@bot.tree.command(name="search_symbols", description="종목/회사 검색")
@app_commands.describe(query="검색할 종목/회사 이름 또는 코드")
async def search_symbols_command(interaction: Interaction, query: str):
    if registry.ORCHESTRATOR is None:
        await interaction.response.send_message("오류: Orchestrator가 준비되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: COMMANDS['search_symbols'](query=query))
    await interaction.followup.send(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```", ephemeral=True)

@bot.tree.command(name="search_web", description="웹 검색")
@app_commands.describe(query="검색할 웹 키워드")
async def search_web_command(interaction: Interaction, query: str):
    if registry.ORCHESTRATOR is None:
        await interaction.response.send_message("오류: Orchestrator가 준비되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: COMMANDS['search_web'](query=query))
    await interaction.followup.send(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```", ephemeral=True)

@bot.tree.command(name="multi_search", description="멀티 검색 및 요약")
@app_commands.describe(query="검색할 키워드", attempts="시도 횟수(1~10)")
async def multi_search_command(interaction: Interaction, query: str, attempts: str = "3"): 
    if registry.ORCHESTRATOR is None:
        await interaction.response.send_message("오류: Orchestrator가 준비되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: COMMANDS['multi_search'](query=query, attempts=attempts))
    await interaction.followup.send(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```", ephemeral=True)

@bot.tree.command(name="get_quote", description="현재 시세 조회")
@app_commands.describe(symbol="조회할 종목 심볼")
async def get_quote_command(interaction: Interaction, symbol: str):
    if registry.ORCHESTRATOR is None:
        await interaction.response.send_message("오류: Orchestrator가 준비되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: COMMANDS['get_quote'](symbol=symbol))
    await interaction.followup.send(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```", ephemeral=True)

@bot.tree.command(name="get_historical_data", description="과거 시세 데이터 조회")
@app_commands.describe(symbol="종목 코드", timeframe="기간 구분(D/W/M)", start_date="시작일(YYYYMMDD)", end_date="종료일(YYYYMMDD)", period="데이터 포인트 수")
async def get_historical_data_command(interaction: Interaction, symbol: str, timeframe: str, start_date: str, end_date: str, period: str):
    if registry.ORCHESTRATOR is None:
        await interaction.response.send_message("오류: Orchestrator가 준비되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: COMMANDS['get_historical_data'](symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date, period=period))
    await interaction.followup.send(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```", ephemeral=True)

@bot.tree.command(name="order_cash", description="현금 주문 실행")
@app_commands.describe(symbol="종목 코드", quantity="수량", price="가격", order_type="주문 유형(00:지정가, 01:시장가)", buy_sell_code="매수(02) 또는 매도(01) 코드")
async def order_cash_command(interaction: Interaction, symbol: str, quantity: str, price: str, order_type: str, buy_sell_code: str):
    if registry.ORCHESTRATOR is None:
        await interaction.response.send_message("오류: Orchestrator가 준비되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: COMMANDS['order_cash'](symbol=symbol, quantity=quantity, price=price, order_type=order_type, buy_sell_code=buy_sell_code))
    await interaction.followup.send(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```", ephemeral=True)

@bot.tree.command(name="get_overseas_trading_status", description="해외 거래 가능 여부 조회")
async def get_overseas_trading_status_command(interaction: Interaction):
    if registry.ORCHESTRATOR is None:
        await interaction.response.send_message("오류: Orchestrator가 준비되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: COMMANDS['get_overseas_trading_status']())
    await interaction.followup.send(f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```", ephemeral=True)
