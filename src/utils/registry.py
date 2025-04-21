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
    """Finnhub 기반 최신 뉴스 검색"""
    if ORCHESTRATOR is None:
        return []
    # Decide whether to use query as symbol or rely on general category search in the method
    # Passing query directly; the method can decide how to use it.
    return ORCHESTRATOR.info_crawler.search_news(query=query)

@command
def search_symbols(query: str) -> list:
    """KIS API로 종목/회사 검색 (query)."""
    if ORCHESTRATOR is None:
        return []
    # KIS API를 활용한 종목 검색 우선시
    broker = ORCHESTRATOR.broker
    # 해외 여부 자동 감지
    is_foreign = broker.is_overseas_symbol(query)
    # Try broker symbol search, fallback on missing method or error
    try:
        results = broker.search_symbol(query=query, is_foreign=is_foreign)
    except Exception:
        results = []
    if results:
        return results
    # fallback: 기존 InfoCrawler 검색
    if hasattr(ORCHESTRATOR, 'info_crawler'):
        return ORCHESTRATOR.info_crawler.search_web(query=query)
    return []

@command
def search_web(query: str) -> list[dict]:
    """Google Custom Search API 기반 일반 웹 검색"""
    if ORCHESTRATOR is None:
        return []
    # Google Custom Search API 기반 검색으로 대체
    try:
        return ORCHESTRATOR.info_crawler.search_web(query=query)
    except Exception as e:
        logging.error(f"Google Custom Search API search failed: {e}")
        return []

@command
def multi_search(query: str, attempts: str = "3") -> dict:
    """여러 번의 news/web 검색을 병렬 수행해 종합 요약합니다."""
    if ORCHESTRATOR is None or not hasattr(ORCHESTRATOR, 'info_crawler'):
        return {"error": "Orchestrator 또는 InfoCrawler 미준비"}
    # attempts는 문자열로 들어오므로 int 변환
    try:
        n = int(attempts)
    except ValueError:
        logging.warning(f"Invalid attempts value '{attempts}', defaulting to 3.")
        logger.warning(f"Invalid attempts value '{attempts}', defaulting to 3.")
        n = 3
    except Exception as e: # Catch other potential errors during conversion
        logger.error(f"Error converting attempts '{attempts}' to int: {e}, defaulting to 3.")
        n = 3
        
    # Ensure n is within reasonable bounds if needed (e.g., 1 to 10)
    n = max(1, min(n, 10)) # Example bounds
    
    # The called function now returns a dict, so we return it directly
    return ORCHESTRATOR.info_crawler.multi_search(query=query, attempts=n)

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
    # Auto-detect foreign stock symbols
    is_foreign = ORCHESTRATOR.kis.is_overseas_symbol(symbol)
    # Execute get_quote with appropriate flag
    return ORCHESTRATOR.kis.get_quote(symbol=symbol, is_foreign=is_foreign)

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