"""Very small registry for LLM‑callable functions.

Usage:

    from src.utils.registry import command

    @command
    def get_balance():
        ...
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict, Any
import pandas as pd
import logging
from src.config import ORCHESTRATOR

COMMANDS: Dict[str, Callable[..., Any]] = {}

# Utility so main.py can tell us about the live orchestrator
def set_orchestrator(obj: Any) -> None:
    """Store a reference to the (running) orchestrator so wrappers can reach it."""
    global ORCHESTRATOR
    ORCHESTRATOR = obj

def command(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator – registers *fn* so the LLM can discover & invoke it.

    It also attaches a minimal OpenAI function‑spec (``._oas``) which we later pass
    to :pyfunc:`openai.chat.completions.create`.
    """

    name = fn.__name__
    sig = inspect.signature(fn)

    # Very naïve JSON schema – all params are plain strings.
    parameters_schema = {
        "type": "object",
        "properties": {
            k: {"type": "string"} for k in sig.parameters.keys()
        },
        "required": list(sig.parameters.keys()),
    }

    COMMANDS[name] = fn
    fn._oas = {
        "name": name,
        "description": fn.__doc__ or "",
        "parameters": parameters_schema,
    }

    return fn

# --- NEW WRAPPERS --- 
@command
def get_balance() -> dict:
    """현재 계좌 예수금·총자산."""
    if ORCHESTRATOR is None:
        return {"error": "orchestrator not ready"}
    return ORCHESTRATOR.broker.get_balance()


@command
def get_positions() -> list:
    """보유 포지션."""
    if ORCHESTRATOR is None:
        return []
    return ORCHESTRATOR.broker.get_positions()


@command
def get_market_summary(query: str) -> str:
    """사용자 질의(query)에 맞춘 시장 동향 요약."""
    if ORCHESTRATOR is None:
        return "(orchestrator not ready)"
    # Pass query to info_crawler's method
    return ORCHESTRATOR.info_crawler.get_market_summary(user_query=query) # Pass query here

@command
def search_symbols(query: str) -> str:
    """종목명 또는 심볼로 주식/ETF를 검색합니다. Searches for stocks/ETFs by name or symbol."""
    logger = logging.getLogger(__name__)
    logger.info(f"[search_symbols] Searching for symbol/company: {query}")
    try:
        # Use Finnhub for symbol lookup as it handles names better
        search_result = ORCHESTRATOR.finnhub_client.symbol_lookup(query)
        if search_result and search_result.get('result'):
            # Return the first result found
            first_result = search_result['result'][0]
            symbol = first_result.get('symbol')
            description = first_result.get('description')
            logger.info(f"[search_symbols] Found symbol: {symbol} ({description}) for query: {query}")
            # Format the output string
            return f"검색 결과: 심볼 '{symbol}' ({description})을(를) 찾았습니다. 이 심볼로 시세를 조회할 수 있습니다."
        else:
            logger.warning(f"[search_symbols] No symbol found for query: {query} via Finnhub")
            # Optionally, could try KIS search here as a fallback if needed
            # kis_result = ORCHESTRATOR.kis_broker.search_symbol(query)
            # if kis_result: ...
            return f"'{query}'에 대한 종목 정보를 찾을 수 없습니다."

    except Exception as e:
        logger.error(f"[search_symbols] Error searching for symbol '{query}': {e}", exc_info=True)
        return f"'{query}' 종목 검색 중 오류가 발생했습니다: {e}"

@command
def get_quote(symbol: str) -> str:
    """Gets the current quote for the given stock symbol.

    Args:
        symbol: The stock symbol.

    Returns:
        The current quote information.
    """
    if ORCHESTRATOR is None or not hasattr(ORCHESTRATOR, 'kis'):
        return "(Orchestrator or KIS interface not ready)"
    is_foreign = ORCHESTRATOR.kis.is_overseas_symbol(symbol)
    if is_foreign:
        # 해외 주식: Finnhub만 사용
        return ORCHESTRATOR.info_crawler.finnhub.get_quote(symbol)
    else:
        # 국내 주식: KIS만 사용
        return ORCHESTRATOR.kis.get_quote(symbol=symbol, is_foreign=False)


@command
def get_historical_data(symbol: str, timeframe: str, start_date: str, end_date: str, period: str) -> list:
    """과거 시세 데이터 조회 (일·주·월봉 등)."""
    if ORCHESTRATOR is None:
        return []
    # period는 문자열로 들어오므로 int 변환
    # KIS get_historical_data now returns list of dicts directly
    return ORCHESTRATOR.broker.get_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date or None,
        end_date=end_date or None,
        period=int(period) if period else None # Keep int conversion with check
    )

@command
def order_cash(symbol: str, quantity: str, price: str, order_type: str, buy_sell_code: str) -> dict:
    """현금 주문 실행 (시장가·지정가)."""
    if ORCHESTRATOR is None:
        return {"error": "orchestrator not ready"}
    return ORCHESTRATOR.broker.order_cash(
        symbol=symbol,
        quantity=int(quantity),
        price=int(price),
        order_type=order_type,
        buy_sell_code=buy_sell_code
    )

@command
def get_overseas_trading_status() -> dict:
    """해외 주식(ETF 포함) 거래 가능 여부 조회."""
    if ORCHESTRATOR is None:
        return {"error": "orchestrator not ready"}
    return ORCHESTRATOR.broker.get_overseas_status()
