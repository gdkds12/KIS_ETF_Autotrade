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

COMMANDS: Dict[str, Callable[..., Any]] = {}

# Global runtime object (lazy‑set later)
ORCHESTRATOR = None  # type: Any

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
def search_news(query: str) -> list:
    """Finnhub로 관련 뉴스 검색 (query는 심볼 또는 키워드)."""
    if ORCHESTRATOR is None or not hasattr(ORCHESTRATOR, 'info_crawler'):
        return []
    # Decide whether to use query as symbol or rely on general category search in the method
    # Passing query directly; the method can decide how to use it.
    return ORCHESTRATOR.info_crawler.search_news(query=query, category='general') # Example: use query for symbol if needed, else general

@command
def search_symbols(query: str) -> list:
    """Finnhub로 종목/회사 검색 (query)."""
    if ORCHESTRATOR is None or not hasattr(ORCHESTRATOR, 'info_crawler'):
        return []
    return ORCHESTRATOR.info_crawler.search_symbols(query)

@command
def search_web(query: str) -> str:
    """Searches Finnhub for the given query.

    Args:
        query: The search query.

    Returns:
        Search results from Finnhub.
    """
    # Ensure Orchestrator and finnhub client are available
    if ORCHESTRATOR is None or not hasattr(ORCHESTRATOR, 'finnhub'):
        return "(Orchestrator or Finnhub client not ready)"
    # Execute Finnhub search via Orchestrator
    return ORCHESTRATOR.finnhub.search(query=query)

@command
def get_quote(symbol: str) -> str:
    """Gets the current quote for the given stock symbol.

    Args:
        symbol: The stock symbol.

    Returns:
        The current quote information.
    """
    # Ensure Orchestrator and KIS interface are available
    if ORCHESTRATOR is None or not hasattr(ORCHESTRATOR, 'kis'):
        return "(Orchestrator or KIS interface not ready)"
    # Execute get_quote via Orchestrator's KIS interface
    return ORCHESTRATOR.kis.get_quote(symbol=symbol)

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

# @command(
#     name="search_news",
#     description="Search for news articles related to a specific topic or company.",
#     parameters=[
#         {
#             "name": "query",
#             "description": "The topic or company name to search news for.",
#             "type": "string",
#             "required": True,
#         },
#         {
#             "name": "days_back",
#             "description": "How many days back to search for news. Defaults to 7.",
#             "type": "integer",
#             "required": False,
#         }
#     ],
#     enabled=True,
# )
# def search_news(query: str, days_back: int = 7) -> str:
#     """Searches for news articles.
#
#     Args:
#         query: The search query.
#         days_back: How many days back to search.
#
#     Returns:
#         A summary of news articles.
#     """
#     return ORCHESTRATOR.info_crawler.search_news(query=query, days_back=days_back) 