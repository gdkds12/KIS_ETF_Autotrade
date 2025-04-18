# 시장 이슈 수집·요약 
import logging
import feedparser
import requests
from requests.exceptions import RequestException
import google.generativeai as genai

from src.config import settings

logger = logging.getLogger(__name__)

class InfoCrawler:
    def __init__(self):
        """InfoCrawler 초기화"""
        self.fnguide_rss_url = "https://www.example-fnguide-rss.com/" # Placeholder
        self.krx_rss_url = "https://www.example-krx-rss.com/" # Placeholder
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AutotradeETFB<x_bin_568>Bot/1.0'})
        
        # Gemini 모델 설정
        self.gemini_model = None
        if settings.GOOGLE_API_KEY:
            try:
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                self.gemini_model = genai.GenerativeModel(settings.LLM_LIGHTWEIGHT_TIER_MODEL)
                logger.info(f"Gemini model '{settings.LLM_LIGHTWEIGHT_TIER_MODEL}' initialized for summarization.")
            except Exception as e:
                 logger.error(f"Failed to initialize Gemini model: {e}")
        else:
             logger.warning("GOOGLE_API_KEY not set. Summarization feature will be limited.")
             
        logger.info("InfoCrawler initialized.")

    def _summarize_with_gemini(self, text: str) -> str:
        """Gemini 모델을 사용하여 텍스트 요약"""
        if not self.gemini_model:
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
            response = self.gemini_model.generate_content(prompt, safety_settings=safety_settings)
            summary = response.text
            logger.info("Successfully received summary from Gemini.")
            return summary
        except Exception as e:
            logger.error(f"Gemini summarization failed: {e}", exc_info=True)
            # 오류 응답 분석 (선택 사항)
            # if hasattr(e, 'response') and e.response.prompt_feedback:
            #     logger.error(f"Gemini prompt feedback: {e.response.prompt_feedback}")
            return f"(시장 정보 요약 중 오류 발생: {e})"

    def get_market_summary(self, max_articles: int = 10) -> str:
        """Fnguide, KRX 등에서 시장 정보를 크롤링하고 Gemini로 요약합니다."""
        logger.info("Starting market information crawling...")
        articles = []
        try:
            # 1. Fetch Fnguide RSS
            fnguide_feed = self._fetch_rss(self.fnguide_rss_url)
            if fnguide_feed:
                articles.extend(self._parse_feed(fnguide_feed, "Fnguide"))

            # 2. Fetch KRX RSS
            krx_feed = self._fetch_rss(self.krx_rss_url)
            if krx_feed:
                articles.extend(self._parse_feed(krx_feed, "KRX"))

            if not articles:
                logger.warning("No articles found from any source.")
                return "수집된 시장 정보가 없습니다."

            # 4. 기사 정렬 및 선택
            articles.sort(key=lambda x: x.get('published_parsed'), reverse=True)
            selected_articles = articles[:max_articles]
            combined_text = "\n\n".join([f"[{a['source']}] {a['title']}\n{a['summary']}" for a in selected_articles])
            logger.info(f"Combined text from {len(selected_articles)} articles for summarization.")

            # 5. Gemini로 요약
            if self.gemini_model:
                summary = self._summarize_with_gemini(combined_text)
                return summary
            else:
                # Gemini 모델 없으면 원문 일부 반환
                logger.warning("Returning partial raw text as Gemini model is unavailable.")
                return f"최신 시장 뉴스 {len(selected_articles)}건 (요약 불가):\n{combined_text[:1000]}..."

        except Exception as e:
            logger.error(f"Failed to crawl or summarize market info: {e}", exc_info=True)
            return f"시장 정보 수집/요약 중 오류 발생: {e}"

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
    
    if not settings.GOOGLE_API_KEY:
        print("\nWarning: GOOGLE_API_KEY not set. Summarization will be skipped.")
    
    crawler = InfoCrawler()
    # Mock RSS feed for testing
    crawler.fnguide_rss_url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=earnings&apikey=demo" # Example RSS-like JSON
    crawler.krx_rss_url = "" # Disable KRX for this test
    
    # Modify parsing for AlphaVantage JSON feed (if using the example URL)
    def parse_alphavantage(feed_json, source_name):
        articles = []
        if not feed_json or 'feed' not in feed_json:
             return articles
        for entry in feed_json['feed'][:5]: # Take first 5
             articles.append({
                 'source': source_name,
                 'title': entry.get('title', 'N/A'),
                 'link': entry.get('url', 'N/A'),
                 'published': entry.get('time_published', None),
                 'published_parsed': None, # TODO: Parse time_published if needed for sorting
                 'summary': entry.get('summary', 'N/A')
             })
        return articles
        
    def fetch_alphavantage(url):
         try:
             response = crawler.session.get(url, timeout=10)
             response.raise_for_status()
             logger.info(f"Fetched data from {url}")
             return response.json()
         except RequestException as e:
             logger.error(f"Error fetching {url}: {e}")
             return None

    # Override fetch/parse for the example
    crawler._fetch_rss = fetch_alphavantage
    crawler._parse_feed = parse_alphavantage

    summary = crawler.get_market_summary()
    print("\n--- Market Summary (using Gemini if available) ---")
    print(summary) 