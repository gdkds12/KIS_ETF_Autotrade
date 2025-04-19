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

@command(
    name="search_web",
    description="Search the web for financial information, news, or stock symbols using Finnhub. Use this for specific queries about companies, stocks, or market news.",
    parameters=[
        {
            "name": "query",
            "description": "The search query string.",
            "type": "string",
            "required": True,
        }
    ],
    enabled=True,
)
def search_web(query: str) -> str:
    """Searches Finnhub for the given query.

    Args:
        query: The search query.

    Returns:
        Search results from Finnhub.
    """
    return ORCHESTRATOR.finnhub.search(query=query)

@command(
    name="get_quote",
    description="Get the current quote for a stock symbol.",
    parameters=[
        {
            "name": "symbol",
            "description": "The stock symbol to get the quote for.",
            "type": "string",
            "required": True,
        }
    ],
    enabled=True,
)
def get_quote(symbol: str) -> str:
    """Gets the current quote for the given stock symbol.

    Args:
        symbol: The stock symbol.

    Returns:
        The current quote information.
    """
    return ORCHESTRATOR.kis.get_quote(symbol=symbol)

@command
def get_historical_data(symbol: str, timeframe: str, start_date: str, end_date: str, period: str) -> list:
    """
    과거 시세(일/주/월봉)를 조회합니다.
    - symbol: 종목 코드 (e.g. "069500")
    - timeframe: 'D', 'W' 또는 'M'
    - start_date, end_date: YYYYMMDD 형식 (필요 없으면 빈 문자열)
    - period: (일봉 조회 시) 기간(정수)
    """
    if ORCHESTRATOR is None:
        return []
    # 문자열 파싱
    # KIS 브로커의 get_historical_data는 pandas DataFrame을 반환하므로 변환 필요
    pd_data = ORCHESTRATOR.broker.get_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date or None,
        end_date=end_date or None,
        period=int(period) if period else None
    )
    # DataFrame → JSON serializable list of dict
    records = []
    # Pandas DataFrame의 인덱스와 행을 순회
    # reset_index()를 사용하여 인덱스를 일반 컬럼으로 변환 ('date' 등)
    for idx, row in pd_data.reset_index().iterrows():
        # 날짜 형식 변환 (Pandas Timestamp -> YYYY-MM-DD string)
        rec = {"date": row['index'].strftime("%Y-%m-%d")} # 날짜 컬럼명 확인 필요 (index or date?)
        # 나머지 컬럼 순회
        for col in pd_data.columns:
            # numpy 타입도 정수/실수로 변환
            val = row[col]
            if hasattr(val, "item"): # numpy scaler type 체크
                val = val.item()
            # NaN 값은 None으로 변환 (JSON 호환)
            if pd.isna(val):
                val = None
            rec[col] = val
        records.append(rec)
    return records

@command
def order_cash(symbol: str, quantity: str, price: str, order_type: str, buy_sell_code: str) -> dict:
    """
    현금 주문을 실행합니다.
    - symbol: 종목 코드
    - quantity: 수량 (정수 문자열)
    - price: 주문 가격 (정수 문자열, 시장가=0)
    - order_type: "00" (지정가) 또는 "01" (시장가)
    - buy_sell_code: "02" 매수, "01" 매도
    """
    if ORCHESTRATOR is None:
        return {}
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