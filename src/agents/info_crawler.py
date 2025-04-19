# 시장 이슈 수집·요약 
import logging
# Remove unused imports if feedparser/requests are no longer needed
# import feedparser 
# import requests 
# from requests.exceptions import RequestException
import openai # Keep OpenAI for summarization
import finnhub # Import Finnhub client

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
        """InfoCrawler 초기화 (Finnhub 연동)"""
        # Finnhub 클라이언트 초기화
        if not settings.FINNHUB_API_KEY:
            logger.error("FINNHUB_API_KEY is not set. Finnhub features will be disabled.")
            self.fh_client = None
        else:
            try:
                self.fh_client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)
                # Test connection (optional, e.g., fetch profile for a known symbol)
                # self.fh_client.company_profile2(symbol='AAPL') 
                logger.info("Finnhub client initialized successfully.")
            except Exception as e:
                 logger.error(f"Failed to initialize Finnhub client: {e}", exc_info=True)
                 self.fh_client = None
                 
        # Remove requests.Session if no longer used
        # self.session = requests.Session()
        # self.session.headers.update({'User-Agent': 'AutotradeETFB<x_bin_568>Bot/1.0'})
        
        logger.info("InfoCrawler initialized.") # General init message

    def _fetch_raw_data(self) -> list[str]:
        """Finnhub에서 최신 뉴스 헤드라인 목록을 조회합니다."""
        if not self.fh_client:
            logger.warning("Finnhub client not initialized. Cannot fetch news.")
            return []
            
        logger.info(f"Fetching general news headlines from Finnhub...")
        try:
            # 일반 뉴스 카테고리 조회
            # Note: Finnhub returns dicts, not just strings
            articles = self.fh_client.general_news(category='general', min_id=0)
            # Extract headlines, handling potential missing keys
            headlines = [str(a.get('headline', '')).strip() for a in articles if a.get('headline')]
            logger.info(f"Fetched {len(headlines)} headlines from Finnhub")
            return headlines[:10] # Limit to 10 for prompt length
        except finnhub.FinnhubAPIException as e:
            logger.error(f"Finnhub API error fetching news: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching or processing Finnhub news: {e}", exc_info=True)
            return []

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
        """사용자 요청(user_query)에 맞춰 Finnhub 뉴스 헤드라인을 수집하고, LLM으로 요약해 돌려줍니다."""
        logger.info(f"Getting market summary for query: {user_query!r}")
        
        headlines = self._fetch_raw_data() # Fetch headlines using finnhub client
        
        if not headlines:
            logger.warning("No headlines fetched from Finnhub.")
            # Return a message indicating no news or error during fetch
            return "(최신 시장 뉴스를 가져올 수 없습니다.)"

        # Build the prompt using the fetched headlines and user query
        snippets = [f"- {h}" for h in headlines]
        prompt = (
            f"사용자 질문: {user_query}\n\n"
            f"최근 주요 뉴스 헤드라인:\n"
            + "\n".join(snippets) + "\n\n"
            f"위 뉴스를 바탕으로 사용자 질문 '{user_query}'에 대해 한국어로 간결하게 답변해주세요."
        )
        logger.debug(f"Generated prompt for LLM summarization:\n{prompt}")

        # Call the LLM summarization function with the generated prompt
        summary = self._summarize_with_llm(prompt)
        logger.info(f"Generated market summary (length: {len(summary)}).")
        return summary

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