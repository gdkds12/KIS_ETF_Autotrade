# -*- coding: utf-8 -*-
"""
Finnhub API 클라이언트 모듈

이 모듈은 Finnhub API와의 상호작용을 위한 클라이언트 클래스를 제공합니다.
"""
import finnhub
import os
import logging
from typing import List, Dict, Any
# from src.config import settings # Settings might not be needed directly anymore

logger = logging.getLogger(__name__)

class FinnhubClientError(Exception):
    """Custom exception for FinnhubClient errors."""
    pass

class FinnhubClient:
    def __init__(self, token: str):
        """Initializes the Finnhub client using the provided API token."""
        if not token:
            logger.error("Finnhub API token is missing.")
            raise ValueError("Finnhub API token is required.")
            
        try:
            # Finnhub 파이썬 SDK: Client(api_key=...) 사용
            self.client = finnhub.Client(api_key=token)
            logger.info("Finnhub client initialized successfully.")
        except Exception as e:
             logger.error(f"Failed to initialize Finnhub client: {e}", exc_info=True)
             raise FinnhubClientError(f"Finnhub client initialization failed: {e}")

    def get_quote(self, symbol: str):
        """Fetches the real-time quote for a given stock symbol."""
        logger.debug(f"Fetching quote for symbol: {symbol}")
        try:
            # Use the client directly
            quote_data = self.client.quote(symbol)
            logger.debug(f"Received quote for {symbol}: {quote_data}")
            return quote_data
        except finnhub.FinnhubAPIException as e: # Use FinnhubAPIException if provided by client
            logger.error(f"Finnhub API error fetching quote for {symbol}: {e}", exc_info=True)
            raise FinnhubClientError(f"Finnhub API error fetching quote: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching quote for {symbol}: {e}", exc_info=True)
            raise FinnhubClientError(f"Unexpected error fetching quote: {e}")

    def get_candles(self, symbol: str, resolution: str, _from: int, to: int):
        """Fetches candle (chart) data for a given stock symbol."""
        logger.debug(f"Fetching candles for {symbol} (Resolution: {resolution}, From: {_from}, To: {to})")
        try:
             # Use the client directly
            candle_data = self.client.stock_candles(symbol, resolution, _from, to)
            logger.debug(f"Received candle data for {symbol}")
            return candle_data
        except finnhub.FinnhubAPIException as e:
            logger.error(f"Finnhub API error fetching candles for {symbol}: {e}", exc_info=True)
            raise FinnhubClientError(f"Finnhub API error fetching candles: {e}")
        except Exception as e:
             logger.error(f"Unexpected error fetching candles for {symbol}: {e}", exc_info=True)
             raise FinnhubClientError(f"Unexpected error fetching candles: {e}")

    def search(self, query: str) -> dict:
        """Performs a symbol lookup using the query."""
        logger.debug(f"Searching symbols with query: '{query}'")
        try:
            search_result = self.client.symbol_lookup(query)
            logger.debug(f"Symbol search returned results for '{query}'")
            # The client might return a dict directly
            return search_result
        except finnhub.FinnhubAPIException as e:
            logger.error(f"Finnhub API error searching symbols for '{query}': {e}", exc_info=True)
            raise FinnhubClientError(f"Finnhub API error searching symbols: {e}")
        except Exception as e:
            logger.error(f"Unexpected error searching symbols for '{query}': {e}", exc_info=True)
            raise FinnhubClientError(f"Unexpected error searching symbols: {e}")

    def get_company_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """특정 회사의 뉴스를 가져옵니다.

        Args:
            symbol (str): 뉴스 기사를 가져올 회사 심볼 (예: 'AAPL')
            start_date (str): 뉴스 시작 날짜 (YYYY-MM-DD 형식)
            end_date (str): 뉴스 종료 날짜 (YYYY-MM-DD 형식)

        Returns:
            List[Dict[str, Any]]: 뉴스 기사 목록. 각 기사는 딕셔너리 형태.
                                 오류 발생 시 빈 리스트 반환.

        Raises:
            FinnhubClientError: API 호출 중 오류 발생 시
        """
        logger.debug(f"Fetching company news for {symbol} ({start_date} to {end_date})")
        try:
            news = self.client.company_news(symbol, _from=start_date, to=end_date)
            return news
        except finnhub.FinnhubAPIException as e:
            logger.error(f"Finnhub API error fetching company news for {symbol}: {e}", exc_info=True)
            raise FinnhubClientError(f"Finnhub API error fetching company news: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching company news for {symbol}: {e}", exc_info=True)
            raise FinnhubClientError(f"Unexpected error fetching company news: {e}")

    def get_general_news(self, category: str = 'general', min_id: int = 0) -> List[Dict[str, Any]]:
        """일반 시장 뉴스를 가져옵니다.

        Args:
            category (str, optional): 뉴스 카테고리 (예: 'general', 'forex', 'crypto', 'merger'). 기본값 'general'.
            min_id (int, optional): 페이징을 위한 최소 뉴스 ID. 기본값 0.

        Returns:
            List[Dict[str, Any]]: 뉴스 기사 목록. 각 기사는 딕셔너리 형태.
                                 오류 발생 시 빈 리스트 반환.

        Raises:
            FinnhubClientError: API 호출 중 오류 발생 시
        """
        logger.debug(f"Fetching general news for category '{category}'")
        try:
            news = self.client.general_news(category, min_id=min_id)
            return news
        except finnhub.FinnhubAPIException as e:
            logger.error(f"Finnhub API error fetching general news for '{category}': {e}", exc_info=True)
            raise FinnhubClientError(f"Finnhub API error fetching general news: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching general news for '{category}': {e}", exc_info=True)
            raise FinnhubClientError(f"Unexpected error fetching general news: {e}")

    def close(self):
        """Closes the API client connection and releases resources."""
        # The finnhub.Client might manage its session implicitly
        # or use requests underneath. Explicit closing might not be needed
        # unless documented or causing issues.
        # Check if session attribute exists and try closing if it does.
        if hasattr(self.client, 'session') and self.client.session:
            try:
                self.client.session.close()
                logger.info("Closed Finnhub client session.")
            except Exception as e:
                # Log error but don't prevent program flow if closing fails
                logger.warning(f"Could not close Finnhub client session: {e}", exc_info=True)
                pass
        else:
            logger.debug("No explicit session found on Finnhub client to close.")

# Example usage (optional, for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load environment variables (requires python-dotenv)
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("FINNHUB_API_KEY")

    if not api_key:
        print("Error: FINNHUB_API_KEY environment variable not set.")
    else:
        client = None
        try:
            client = FinnhubClient(token=api_key)
            
            print("\n--- Testing Get Quote ---")
            quote = client.get_quote("AAPL")
            print(f"AAPL Quote: {quote}")

            print("\n--- Testing Search ---")
            search_res = client.search("Microsoft")
            print(f"Search results for 'Microsoft': {search_res.get('result', [])[:3]}") # Show top 3

            # Add candle test if needed
            # print("\n--- Testing Get Candles ---")
            # from datetime import datetime, timedelta
            # to_ts = int(datetime.now().timestamp())
            # from_ts = int((datetime.now() - timedelta(days=30)).timestamp())
            # candles = client.get_candles("AAPL", "D", from_ts, to_ts)
            # print(f"AAPL Daily Candles (last 30 days): Found {len(candles.get('c', []))} entries.")

        except FinnhubClientError as e:
            print(f"Finnhub Client Error: {e}")
        except ValueError as e:
            print(f"Value Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            # Ensure client is closed if initialized
            if client:
                client.close() 