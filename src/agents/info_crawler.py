# 시장 이슈 수집·요약 
import logging
import requests # Use requests library directly
from requests.exceptions import RequestException
import openai # Keep OpenAI for summarization
import finnhub # Remove finnhub client import
import time
from concurrent.futures import ThreadPoolExecutor, as_completed # Import concurrent features

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
    """
    웹사이트 크롤링, 뉴스 검색 등을 수행하는 클래스입니다.
    """

    def __init__(self):
        """InfoCrawler 초기화 (Finnhub + Web Search)"""
        # Finnhub 설정
        self.api_key = settings.FINNHUB_API_KEY # Keep using this name for finnhub
        self.base_url = "https://finnhub.io/api/v1" # Finnhub API base URL
        
        # SerpAPI 설정 (일반 웹 검색용)
        if not settings.SERPAPI_API_KEY:
            logger.warning("SERPAPI_API_KEY not set. Web search functionality will be disabled.")
            self.serpapi_key = None
        else:
            self.serpapi_key = settings.SERPAPI_API_KEY
        self.serpapi_url = "https://serpapi.com/search"
        
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AutotradeETFB/1.0'})
        
        # Remove fh_client attribute
        # self.fh_client = finnhub.Client(api_key=settings.FINNHUB_API_KEY)
        
        # Keep LLM model reference if needed by _summarize_with_llm implementation
        # (Current _summarize_with_llm uses global openai key, so no instance needed here)
        # self.llm_model = info_llm_model 
        
        logger.info("InfoCrawler initialized (Finnhub + Web Search).")

    # Remove old _fetch_raw_data
    # def _fetch_raw_data(...)

    # New method to search news using requests
    def search_news(self, query: str = None, category: str = 'general') -> list[dict]:
        """Finnhub API 로 최신 뉴스 검색 (requests 사용)"""
        if not self.api_key:
            logger.warning("Finnhub API key not set. Cannot search news.")
            return []
            
        url = f"{self.base_url}/news"
        params = {"token": self.api_key, "category": category}
        # Finnhub /news endpoint doesn't directly support query text search,
        # but we can fetch general news or potentially company news if query is a symbol.
        # For simplicity, we'll stick to category for now.
        # If query IS a symbol, Finnhub has /company-news
        # if query and category is None: # Example: If query is likely a symbol
        #    url = f"{self.base_url}/company-news"
        #    params = {"symbol": query.upper(), "token": self.api_key, "from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}
        
        logger.info(f"Searching Finnhub news (category: {category})...")
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            news_list = resp.json()
            # Ensure it's a list before slicing
            if isinstance(news_list, list):
                 logger.info(f"Fetched {len(news_list)} news articles from Finnhub.")
                 return news_list
            else:
                 logger.error(f"Unexpected response format from Finnhub news API: {news_list}")
                 return []
        except RequestException as e:
             logger.error(f"Finnhub news request failed: {e}", exc_info=True)
             return []
        except Exception as e:
             logger.error(f"Error processing Finnhub news response: {e}", exc_info=True)
             return []

    # New method to search symbols using requests
    def search_symbols(self, query: str) -> list[dict]:
        """Finnhub API 로 종목/회사 검색 (requests 사용)"""
        if not self.api_key:
            logger.warning("Finnhub API key not set. Cannot search symbols.")
            return []
        if not query:
            logger.warning("Empty query for symbol search.")
            return []
            
        url = f"{self.base_url}/search"
        params = {"q": query, "token": self.api_key}
        logger.info(f"Searching Finnhub symbols for query: '{query}'...")
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("result", [])
            logger.info(f"Found {len(results)} symbol matches for query '{query}'.")
            return results
        except RequestException as e:
             logger.error(f"Finnhub symbol search request failed: {e}", exc_info=True)
             return []
        except Exception as e:
             logger.error(f"Error processing Finnhub symbol search response: {e}", exc_info=True)
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

    def get_market_summary(self, user_query: str, max_articles: int = 5) -> str:
        """사용자 질의(user_query)에 대한 시장 동향을 Finnhub 뉴스 기반으로 요약해서 반환"""
        logger.info(f"Getting market summary for query: {user_query!r}")
        
        # Fetch news tailored to the user query if provided
        if user_query:
            logger.info(f"Searching news relevant to query: {user_query}")
            # Assuming search_news can handle query text (otherwise adapt)
            news_list = self.search_news(query=user_query, category='general')
        else:
            logger.info("Searching general news as no specific query provided.")
            news_list = self.search_news(category='general')
        
        if not news_list:
            logger.warning("No news fetched from Finnhub for market summary.")
            return "(최신 시장 뉴스를 가져올 수 없습니다.)"

        # Extract headlines/summaries for the prompt
        snippets = []
        for item in news_list[:max_articles]: # Limit articles used in prompt
            headline = item.get("headline", "")
            summary = item.get("summary", "") or item.get("source", "") # Use summary or source as fallback
            if headline or summary:
                 snippets.append(f"- {headline.strip()} ({summary.strip()})" if headline and summary else f"- {headline.strip() or summary.strip()}")

        if not snippets:
            logger.warning("Could not extract usable snippets from fetched news.")
            return "(뉴스 내용을 처리할 수 없습니다.)"
            
        # Build the prompt using the fetched snippets and user query
        combined_news = "\n".join(snippets)
        prompt = (
            f"사용자 질문: {user_query}\n\n"
            f"최근 주요 뉴스 요약:\n"
            f"{combined_news}\n\n"
            f"위 뉴스를 바탕으로 사용자 질문 '{user_query}'에 대해 한국어로 간결하게 답변해주세요."
        )
        logger.debug(f"Generated prompt for LLM summarization:\n{prompt}")

        # Call the LLM summarization function with the generated prompt
        llm_summary = self._summarize_with_llm(prompt)
        logger.info(f"Generated market summary (length: {len(llm_summary)}).")
        return llm_summary

    def multi_search(self, query: str, attempts: int = 3, max_attempts: int = 10) -> dict:
        """범용 검색: query 기반으로 최소 3번, 최대 10번의 news/web 검색을 병렬 수행해 LLM으로 요약."""
        logger.info(f"Performing multi-search for query: '{query}' with {attempts} attempts (max {max_attempts})")
        # 1) 시도 횟수 보정
        tries = max(3, min(int(attempts), max_attempts)) # Ensure attempts is int

        # 2) 기본 키워드 확장 리스트 (동적 변형)
        suffixes = ["최신 뉴스", "시사 동향", "시장 분석", "지표", "최근 변화", "전망", "영향", "관련주"] # Expanded suffixes
        subqueries_set = {query}
        for s in suffixes:
            subqueries_set.add(f"{query} {s}")
            if len(subqueries_set) >= tries:
                break
        subqueries = list(subqueries_set)
        logger.debug(f"Generated {len(subqueries)} subqueries: {subqueries}")

        # 3) ThreadPoolExecutor로 병렬 검색
        def fetch(q):
            """Helper function to fetch news and web results for a subquery."""
            news_results = []
            web_results = []
            try:
                # Note: Limiting results inside fetch, [:1]
                news_results = self.search_news(query=q)[:1]
                # No need for sleep here if using ThreadPool
            except Exception as e:
                logger.error(f"Error fetching news for subquery '{q}': {e}", exc_info=True)
            try:
                web_results = self.search_web(query=q)[:1]
            except Exception as e:
                logger.error(f"Error fetching web results for subquery '{q}': {e}", exc_info=True)
            return (q, news_results, web_results)

        snippets = []
        # Use max_workers based on tries, but consider limiting it globally (e.g., max 10-15)
        max_workers = min(tries * 2, 10) 
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Map queries to future objects
            future_to_query = {pool.submit(fetch, q): q for q in subqueries}
            for future in as_completed(future_to_query):
                q = future_to_query[future]
                try:
                    _q, news, web = future.result() # q should match _q
                    q_snippets = []
                    if news:
                        item = news[0]
                        title = item.get("headline") or item.get("title") or ""
                        desc = item.get("summary") or item.get("snippet") or ""
                        q_snippets.append(f"- [뉴스] {q}: {title} ({desc})")
                    if web:
                        item = web[0]
                        title = item.get('title') or ""
                        desc = item.get('snippet') or ""
                        q_snippets.append(f"- [웹] {q}: {title} ({desc})")
                    if q_snippets:
                        snippets.extend(q_snippets)
                        
                except Exception as exc:
                    logger.error(f"Subquery '{q}' generated an exception: {exc}", exc_info=True)
                    snippets.append(f"- [오류] '{q}' 처리 중 오류 발생: {exc}")
        
        # Ensure consistent sorting if needed, though as_completed yields in completion order
        # snippets.sort() # Example if sorting is desired

        if not snippets:
            logger.warning(f"Multi-search for '{query}' yielded no results after parallel fetch.")
            return {
                "summary": "(관련 정보를 찾을 수 없습니다.)",
                "subqueries_count": len(subqueries),
                "snippets_count": 0
            }

        # 4) LLM 요약 프롬프트 구성
        combined = "\n".join(snippets)
        prompt = (
            f"사용자가 요청한 주제: {query}\n\n"
            f"아래는 {len(subqueries)}개의 연관 검색어(최대 {tries}개 시도)에 대한 뉴스 및 웹 검색 결과 요약입니다:\n\n{combined}\n\n"
            "위 내용을 바탕으로 사용자 요청에 대해 한국어로 간결하게 종합 분석 및 요약해 주세요."
        )
        logger.debug(f"Generated prompt for multi_search summary:\n{prompt[:500]}...")
        summary = self._summarize_with_llm(prompt)
        
        # Return a dictionary with summary and metadata
        return {
            "summary": summary,
            "subqueries_count": len(subqueries),
            "snippets_count": len(snippets)
        }

    def search_web(self, query: str, num_results: int = 5) -> list[dict]:
        """
        SerpAPI를 이용해 일반 웹 검색을 수행합니다.
        - query: 검색어
        - num_results: 최대 결과 개수
        반환 형식: [{"title":..., "link":..., "snippet":...}, ...]
        """
        if not self.serpapi_key:
            logger.error("SERPAPI_API_KEY 가 설정되지 않았습니다. 웹 검색을 수행할 수 없습니다.")
            return [] # Return empty list as error indicator
            
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "num": num_results,
            "engine": "google", # Specify search engine (e.g., google, naver)
            "gl": "kr", # Specify country (e.g., kr for Korea)
            "hl": "ko" # Specify language (e.g., ko for Korean)
        }
        logger.info(f"Performing web search for query: '{query}' using SerpAPI...")
        try:
            resp = self.session.get(self.serpapi_url, params=params, timeout=10)
            resp.raise_for_status() # Raise HTTPError for bad responses
            data = resp.json()
            results = []
            # Extract relevant fields from organic results
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet") or item.get("displayed_link") # Use displayed_link as fallback
                })
            logger.info(f"Web search completed. Found {len(results)} results for '{query}'.")
            return results
        except RequestException as e:
             logger.error(f"SerpAPI web search request failed for '{query}': {e}", exc_info=True)
             return []
        except Exception as e:
            logger.error(f"Error processing SerpAPI response for '{query}': {e}", exc_info=True)
            return []

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