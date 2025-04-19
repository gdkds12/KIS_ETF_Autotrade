# 시장 이슈 수집·요약 
import logging
import feedparser
import requests
from requests.exceptions import RequestException
import openai # Import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

# --- OpenAI 모델 초기화 (InfoCrawler 용) --- 
if settings.OPENAI_API_KEY:
    # Check if openai.api_key is already set or needs setting
    # Note: Setting it globally might have side effects if other parts use different keys.
    # Consider passing the client or key if needed.
    if not getattr(openai, 'api_key', None):
        openai.api_key = settings.OPENAI_API_KEY
    logger.info(f"InfoCrawler will use OpenAI model: {settings.LLM_LIGHTWEIGHT_TIER_MODEL}")
else:
    logger.warning("OPENAI_API_KEY not set. InfoCrawler LLM summary will be basic.")

class InfoCrawler:
    def __init__(self):
        """InfoCrawler 초기화"""
        self.fnguide_rss_url = "https://www.example-fnguide-rss.com/" # Placeholder
        self.krx_rss_url = "https://www.example-krx-rss.com/" # Placeholder
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AutotradeETFB<x_bin_568>Bot/1.0'})
        
        # Finnhub API key setup
        if not settings.FINNHUB_API_KEY:
            logger.warning("FINNHUB_API_KEY not set. Market news fetching will fail.")
            self.finnhub_key = None
        else:
            self.finnhub_key = settings.FINNHUB_API_KEY
        logger.info("InfoCrawler initialized.")

    def _fetch_raw_data(self) -> list[str]:
        """실제 웹 크롤링 또는 RSS 피드 읽기 로직 (Placeholder)
        
        Returns:
            시장 관련 텍스트 스니펫 리스트
        """
        logger.info("Fetching raw market data (using placeholder)...")
        # --- Placeholder --- 
        # In a real implementation, use libraries like requests and BeautifulSoup
        # to fetch and parse data from financial news sites, KRX, Fnguide RSS, etc.
        raw_data = [
            "코스피 지수가 외국인과 기관의 동반 매수세에 힘입어 3거래일 만에 반등하며 2750선을 회복했습니다.",
            "미국 연준의 금리 인상 속도 조절 기대감이 투자 심리를 개선시킨 것으로 풀이됩니다.",
            "반도체 관련주가 강세를 보였으며, 특히 삼성전자와 SK하이닉스의 주가 상승폭이 컸습니다.",
            "국제 유가는 지정학적 리스크 완화 소식에 소폭 하락했습니다.",
            "오늘 밤 발표될 미국 소비자물가지수(CPI) 결과에 시장의 관심이 집중되고 있습니다."
        ]
        # --- End Placeholder ---
        logger.info(f"Fetched {len(raw_data)} raw data snippets.")
        return raw_data

    def _summarize_with_llm(self, prompt: str) -> str:
        """OpenAI LLM을 사용하여 주어진 프롬프트에 대한 응답(요약)을 생성합니다."""
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not set. Cannot summarize.")
            return "(LLM 요약 불가: API 키 없음)"
        if not prompt:
            logger.warning("Empty prompt provided to LLM.")
            return "(LLM 요약 불가: 빈 프롬프트)"

        try:
            logger.info(f"Requesting OpenAI completion using {settings.LLM_LIGHTWEIGHT_TIER_MODEL}...")
            messages = [
                {"role": "system", "content": "You are an expert in summarizing financial news headlines in Korean based on user queries."}, # System prompt
                {"role": "user", "content": prompt}
            ]
            # Use the synchronous client for simplicity within this potentially sync function
            # If InfoCrawler methods become async, use AsyncOpenAI client
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY) # Create a client instance
            resp = client.chat.completions.create(
                model=settings.LLM_LIGHTWEIGHT_TIER_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500 # Adjust token limit as needed
            )
            summary = resp.choices[0].message.content.strip()
            logger.info("Received summary from OpenAI.")
            return summary
        except openai.APIError as e:
             logger.error(f"OpenAI API Error during summarization: {e}", exc_info=True)
             return f"(OpenAI API 오류: {e})"
        except Exception as e:
            logger.error(f"OpenAI summarization failed: {e}", exc_info=True)
            return f"(요약 불가: {e})"

    def get_market_summary(self, user_query: str) -> str:
        """
        사용자 요청(user_query)에 맞춰 Finnhub 뉴스 헤드라인을 수집하고,
        LLM으로 요약해 돌려줍니다.
        """
        logger.info(f"Getting market summary for query: {user_query!r}")
        
        if not self.finnhub_key:
            return "(오류: Finnhub API 키가 설정되지 않았습니다.)"
            
        # 1) Finnhub에서 뉴스 헤드라인 가져오기
        params = {"category": "general", "token": self.finnhub_key}
        try:
            logger.info("Fetching news headlines from Finnhub...")
            resp = self.session.get("https://finnhub.io/api/v1/news", params=params, timeout=5)
            resp.raise_for_status()
            articles = resp.json()[:5]
            logger.info(f"Fetched {len(articles)} headlines from Finnhub.")
            if not articles:
                 return "(Finnhub에서 관련 뉴스를 찾을 수 없습니다.)"
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub API request failed: {e}", exc_info=True)
            return "(시장 뉴스 조회 중 API 요청 오류가 발생했습니다.)"
        except Exception as e:
            logger.error(f"Failed to process Finnhub response: {e}", exc_info=True)
            return "(시장 뉴스 조회 중 오류가 발생했습니다.)"

        # 2) 사용자 질문과 헤드라인 조합하여 프롬프트 생성
        snippets = [f"- {a.get('headline', a.get('summary', '')).strip()}" for a in articles if a.get('headline') or a.get('summary')]
        if not snippets:
             return "(뉴스 헤드라인 정보를 처리할 수 없습니다.)"
             
        prompt = (
            f"사용자 질문: {user_query}\n\n"
            f"최근 주요 뉴스 헤드라인:\n"
            + "\n".join(snippets) + "\n\n"
            f"위 뉴스를 바탕으로 사용자 질문에 대해 한국어로 간결하게 답변해주세요."
        )
        logger.debug(f"Generated prompt for LLM summarization:\n{prompt}")

        # 3) LLM 요약 호출 (Now uses OpenAI via _summarize_with_llm)
        return self._summarize_with_llm(prompt)

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not settings.FINNHUB_API_KEY or not settings.OPENAI_API_KEY:
        print("\nWarning: FINNHUB_API_KEY or OPENAI_API_KEY not set. Summarization might fail.")
    
    crawler = InfoCrawler()
    test_query = "최근 시장 동향은 어떤가요?"
    summary = crawler.get_market_summary(test_query)
    print(f"\n--- Market Summary for query: '{test_query}' ---")
    print(summary) 