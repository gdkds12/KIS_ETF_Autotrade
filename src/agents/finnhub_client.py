# -*- coding: utf-8 -*-
"""
Finnhub API 클라이언트 모듈

이 모듈은 Finnhub API와의 상호작용을 위한 클라이언트 클래스를 제공합니다.
"""
import finnhub
from typing import List, Dict, Any
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FinnhubClientError(Exception):
    """Finnhub API 관련 오류를 위한 사용자 정의 예외 클래스"""
    pass

class FinnhubClient:
    """Finnhub API와 상호작용하기 위한 클라이언트 클래스"""
    def __init__(self, api_key: str):
        """Finnhub 클라이언트를 초기화합니다.

        Args:
            api_key (str): Finnhub API 키
        """
        if not api_key:
            logger.error("Finnhub API 키가 설정되지 않았습니다.")
            raise ValueError("Finnhub API 키가 필요합니다.")
        try:
            self.client = finnhub.Client(api_key=api_key)
            logger.info("Finnhub 클라이언트가 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"Finnhub 클라이언트 초기화 중 오류 발생: {e}", exc_info=True)
            raise FinnhubClientError(f"Finnhub 클라이언트 초기화 실패: {e}")

    def search(self, query: str) -> Dict[str, Any]:
        """주어진 쿼리에 대해 Finnhub 심볼 검색을 수행합니다.

        Args:
            query (str): 검색할 쿼리 (예: 'Apple')

        Returns:
            Dict[str, Any]: 검색 결과. 'result' 키 아래에 일치하는 심볼 목록 포함.
                            오류 발생 시 빈 딕셔너리 반환.

        Raises:
            FinnhubClientError: API 호출 중 오류 발생 시
        """
        try:
            # logger.debug(f"Finnhub에서 '{query}' 검색 중...")
            res = self.client.symbol_lookup(query)
            # logger.debug(f"Finnhub 검색 결과 수신: {len(res.get('result', []))}개 항목")
            # 결과 필터링 또는 추가 처리 가능 (예: 미국 시장만 필터링)
            return res
        except finnhub.FinnhubAPIException as e:
            logger.error(f"Finnhub API 오류 발생 (검색: '{query}'): {e}")
            raise FinnhubClientError(f"Finnhub API 오류: {e}")
        except Exception as e:
            logger.error(f"Finnhub 검색 중 예상치 못한 오류 발생 ('{query}'): {e}", exc_info=True)
            raise FinnhubClientError(f"Finnhub 검색 중 예상치 못한 오류: {e}")

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
        try:
            # logger.debug(f"{symbol}에 대한 회사 뉴스 검색 ({start_date} ~ {end_date})...")
            news = self.client.company_news(symbol, _from=start_date, to=end_date)
            # logger.debug(f"{symbol} 뉴스 결과 수신: {len(news)}개 항목")
            return news
        except finnhub.FinnhubAPIException as e:
            logger.error(f"Finnhub API 오류 발생 (회사 뉴스: {symbol}): {e}")
            raise FinnhubClientError(f"Finnhub API 오류: {e}")
        except Exception as e:
            logger.error(f"{symbol} 회사 뉴스 검색 중 예상치 못한 오류 발생: {e}", exc_info=True)
            raise FinnhubClientError(f"{symbol} 회사 뉴스 검색 중 예상치 못한 오류: {e}")

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
        try:
            # logger.debug(f"'{category}' 카테고리의 일반 뉴스 검색 중 (min_id: {min_id})...")
            news = self.client.general_news(category, min_id=min_id)
            # logger.debug(f"일반 뉴스 결과 수신: {len(news)}개 항목")
            return news
        except finnhub.FinnhubAPIException as e:
            logger.error(f"Finnhub API 오류 발생 (일반 뉴스: {category}): {e}")
            raise FinnhubClientError(f"Finnhub API 오류: {e}")
        except Exception as e:
            logger.error(f"'{category}' 카테고리 일반 뉴스 검색 중 예상치 못한 오류 발생: {e}", exc_info=True)
            raise FinnhubClientError(f"'{category}' 일반 뉴스 검색 중 예상치 못한 오류: {e}")

# 사용 예시 (테스트 목적)
if __name__ == '__main__':
    try:
        # 설정에서 API 키 로드
        client = FinnhubClient(api_key=settings.FINNHUB_API_KEY)

        # 심볼 검색 테스트
        search_results = client.search('Apple')
        print("--- 심볼 검색 결과 ('Apple') ---")
        if search_results and 'result' in search_results:
            print(f"총 {search_results.get('count', 0)}개 결과 중 일부:")
            for item in search_results['result'][:5]: # 처음 5개 결과만 출력
                print(f"  - {item['symbol']}: {item['description']}")
        else:
            print("검색 결과 없음 또는 오류")
        print("\n")

        # 회사 뉴스 테스트 (예: AAPL)
        from datetime import date, timedelta
        today = date.today()
        start_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        company_news = client.get_company_news('AAPL', start_date, end_date)
        print(f"--- 회사 뉴스 결과 ('AAPL', {start_date} ~ {end_date}) ---")
        if company_news:
            print(f"총 {len(company_news)}개 뉴스 중 최근 3개:")
            for news_item in company_news[:3]: # 최근 3개 뉴스만 출력
                print(f"  - [{news_item['datetime']}] {news_item['headline']}")
                print(f"    URL: {news_item['url']}")
        else:
            print("회사 뉴스 없음 또는 오류")
        print("\n")

        # 일반 뉴스 테스트
        general_news = client.get_general_news()
        print("--- 일반 뉴스 결과 ('general') ---")
        if general_news:
            print(f"총 {len(general_news)}개 뉴스 중 최근 3개:")
            for news_item in general_news[:3]: # 최근 3개 뉴스만 출력
                print(f"  - [{news_item['datetime']}] {news_item['headline']}")
                print(f"    URL: {news_item['url']}")
        else:
            print("일반 뉴스 없음 또는 오류")

    except FinnhubClientError as e:
        print(f"Finnhub 클라이언트 오류: {e}")
    except ValueError as e:
        print(f"설정 오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}") 