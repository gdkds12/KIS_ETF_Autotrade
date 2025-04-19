# 시장 이슈 수집·요약 
import logging
import feedparser
import requests
from requests.exceptions import RequestException
import google.generativeai as genai

from src.config import settings

logger = logging.getLogger(__name__)

# --- Gemini 모델 초기화 (InfoCrawler 용) --- 
info_llm_model = None
if settings.GOOGLE_API_KEY and settings.LLM_LIGHTWEIGHT_TIER_MODEL:
    try:
        # is_configured() 체크 대신 무조건 설정
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        info_llm_model = genai.GenerativeModel(settings.LLM_LIGHTWEIGHT_TIER_MODEL)
        logger.info(f"InfoCrawler initialized with optional LLM: {settings.LLM_LIGHTWEIGHT_TIER_MODEL}")
    except Exception as e:
        logger.warning(f"Failed to initialize optional LLM for InfoCrawler: {e}. Summarization might be basic.")
else:
    logger.warning("Google API Key or Lightweight LLM model not set. InfoCrawler LLM summary will be basic.")

class InfoCrawler:
    def __init__(self):
        """InfoCrawler 초기화"""
        self.fnguide_rss_url = "https://www.example-fnguide-rss.com/" # Placeholder
        self.krx_rss_url = "https://www.example-krx-rss.com/" # Placeholder
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AutotradeETFB<x_bin_568>Bot/1.0'})
        
        self.llm_model = info_llm_model
        # Add Finnhub API key check
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
        """LLM을 사용하여 주어진 프롬프트에 대한 응답(요약)을 생성합니다."""
        if not self.llm_model:
            logger.warning("LLM not available or no prompt provided for summary.")
            return "(LLM 요약 불가: 모델 없음)"
        if not prompt:
            logger.warning("Empty prompt provided to LLM.")
            return "(LLM 요약 불가: 빈 프롬프트)"

        try:
            logger.info(f"Requesting LLM completion for the provided prompt...")
            # Assuming generate_content takes the prompt directly
            response = self.llm_model.generate_content(prompt)
            summary = response.text.strip()
            logger.info("Received LLM completion.")
            return summary
        except Exception as e:
            logger.error(f"LLM completion generation failed: {e}", exc_info=True)
            return f"(LLM 요약 생성 중 오류 발생: {e})"

    def get_market_summary(self, user_query: str) -> str:
        """
        사용자 요청(user_query)에 맞춰 Finnhub 뉴스 헤드라인을 수집하고,
        LLM으로 요약해 돌려줍니다.
        """
        logger.info(f"Getting market summary for query: {user_query!r}")
        
        if not self.finnhub_key:
            return "(오류: Finnhub API 키가 설정되지 않았습니다.)"
            
        # 1) Finnhub에서 뉴스 헤드라인 가져오기
        # Use general news category for broader context
        params = {"category": "general", "token": self.finnhub_key}
        try:
            logger.info("Fetching news headlines from Finnhub...")
            resp = self.session.get("https://finnhub.io/api/v1/news", params=params, timeout=5)
            resp.raise_for_status()
            # Limit the number of articles to avoid overly long prompts
            articles = resp.json()[:5] # Get latest 5 general news articles
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

        # 3) LLM 요약 호출
        return self._summarize_with_llm(prompt)

    def _summarize_with_gemini(self, text: str) -> str:
        """Gemini 모델을 사용하여 텍스트 요약"""
        if not self.llm_model:
            logger.warning("Gemini model not available for summarization.")
            return "(요약 불가: Gemini 모델 없음)"
        
        prompt = f"다음 뉴스 기사들을 한국어로 간결하게 요약해줘. 핵심 내용만 포함하고, 시장에 미칠 영향 중심으로 작성해줘.\n\n{text}"
        
        try:
            logger.info(f"Sending text (approx {len(text)} chars) to Gemini for summarization...")
            # Safety settings 설정 (필요에 따라 조정)
            safety_settings = { 
                # genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                # genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                # genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                # genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
            response = self.llm_model.generate_content(prompt, safety_settings=safety_settings)
            summary = response.text
            logger.info("Successfully received summary from Gemini.")
            return summary
        except Exception as e:
            logger.error(f"Gemini summarization failed: {e}", exc_info=True)
            # 오류 응답 분석 (선택 사항)
            # if hasattr(e, 'response') and e.response.prompt_feedback:
            #     logger.error(f"Gemini prompt feedback: {e.response.prompt_feedback}")
            return f"(시장 정보 요약 중 오류 발생: {e})"

    def _fetch_rss(self, url: str):
        """지정된 URL에서 RSS 피드를 가져옵니다."""
        try:
            logger.info(f"Fetching RSS from: {url}")
            feed = feedparser.parse(url)
            if feed.bozo:
                logger.warning(f"Potential issue parsing RSS feed from {url}. Bozo bit set: {feed.bozo_exception}")
            if not feed.entries:
                 logger.warning(f"No entries found in RSS feed from {url}")
                 return None
            logger.info(f"Successfully fetched {len(feed.entries)} entries from {url}")
            return feed
        except Exception as e:
            logger.error(f"Error fetching or parsing RSS from {url}: {e}", exc_info=True)
            return None

    def _parse_feed(self, feed, source_name: str) -> list:
        """파싱된 RSS 피드에서 기사 정보를 추출합니다."""
        articles = []
        if not feed or not feed.entries:
            return articles
        for entry in feed.entries:
            summary = entry.get("summary", entry.get("description", ""))
            articles.append({
                "source": source_name,
                "title": entry.get("title", "제목 없음"),
                "link": entry.get("link", "링크 없음"),
                "published": entry.get("published", None),
                "published_parsed": entry.get("published_parsed", None),
                "summary": summary.strip()
            })
        return articles

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not settings.FINNHUB_API_KEY or not settings.GOOGLE_API_KEY:
        print("\nWarning: FINNHUB_API_KEY or GOOGLE_API_KEY not set. Summarization might fail.")
    
    crawler = InfoCrawler()
    test_query = "최근 시장 동향은 어떤가요?"
    summary = crawler.get_market_summary(test_query)
    print(f"\n--- Market Summary for query: '{test_query}' ---")
    print(summary) 