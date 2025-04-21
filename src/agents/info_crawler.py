# 시장 이슈 수집·요약 
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from src.agents.finnhub_client import FinnhubClient
from src.config import settings
import datetime
import pytz

logger = logging.getLogger(__name__)


logger.info("InfoCrawler 초기화 완료: Google CSE + Finnhub 기반 정보 수집 및 Azure OpenAI 요약 사용")

class InfoCrawler:
    """
    웹사이트 크롤링, 뉴스 검색 등을 수행하는 클래스입니다.
    """

    def __init__(self):
        # Finnhub + Google Custom Search 기반으로 재구성
        self.finnhub = FinnhubClient(settings.FINNHUB_API_KEY)
        logger.info("InfoCrawler initialized (Google CSE + Finnhub).")

    def fetch_article_text(self, url: str) -> str:
        """
        주어진 URL에서 기사 본문을 추출합니다. (BeautifulSoup 기반)
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=7)
            if resp.status_code != 200:
                logger.warning(f"[fetch_article_text] HTTP {resp.status_code} for {url}")
                return ""
            soup = BeautifulSoup(resp.text, "html.parser")
            # 대표적인 기사 본문 추출 시도
            article = soup.find('article')
            if article and article.get_text(strip=True):
                return article.get_text(separator='\n', strip=True)
            main = soup.find('div', id='main')
            if main and main.get_text(strip=True):
                return main.get_text(separator='\n', strip=True)
            # 여러 <p> 태그를 합쳐서 본문 생성
            ps = soup.find_all('p')
            text = '\n'.join([p.get_text(strip=True) for p in ps if len(p.get_text(strip=True)) > 30])
            if len(text) > 100:
                return text
            # fallback: 전체 텍스트 중 길이가 긴 부분
            body = soup.find('body')
            if body:
                all_text = body.get_text(separator='\n', strip=True)
                if len(all_text) > 100:
                    return all_text
            return ""
        except Exception as e:
            logger.error(f"[fetch_article_text] Error fetching article from {url}: {e}", exc_info=True)
            return ""


    def get_market_summary(self, user_query: str, max_articles: int = 10) -> str:
        logger.info(f"[get_market_summary] called with user_query={user_query!r} max_articles={max_articles}")
        """사용자 질의(user_query)에 대한 시장 동향을 Finnhub 뉴스 기반으로 요약해서 반환"""
        logger.info(f"Getting market summary for query: {user_query!r}")
        
        # Fetch web results for the user query (web search only)
        if user_query:
            logger.info(f"Searching web results for query: {user_query!r}")
        else:
            logger.info("Searching general web results as no specific query provided.")
        # Google 검색 API 직접 호출로 대체
        logger.debug(f"[get_market_summary] Starting Google search for query: {user_query}")
        google_results = []
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": settings.GOOGLE_API_KEY,
                "cx": settings.GOOGLE_CX,
                "q": user_query,
                "num": 10
            }
            logger.debug(f"[get_market_summary] Google API request params: {params}")
            resp = requests.get(url, params=params, timeout=7)
            logger.debug(f"[get_market_summary] Google API response status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                google_results = data.get("items", [])
                logger.info(f"[get_market_summary] Google search returned {len(google_results)} items.")
                if google_results:
                    logger.debug(f"[get_market_summary] First Google result: {google_results[0]}")
            else:
                logger.warning(f"[get_market_summary] Google search error: HTTP {resp.status_code}, content: {resp.text}")
        except Exception as e:
            logger.error(f"[get_market_summary] Google search exception: {e}", exc_info=True)
        # Google 뉴스 결과: 항상 10개 가져오기
        google_news_list = google_results if isinstance(google_results, list) else []
        google_news_list = google_news_list[:10]
        logger.info(f"[get_market_summary] Collected {len(google_news_list)} Google web results.")

        # Finnhub 일반 뉴스 결과: 10개만 사용
        try:
            finnhub_news_list = self.finnhub.get_general_news(category='general')
            finnhub_news_list = finnhub_news_list[:10]
            logger.info(f"[get_market_summary] Collected {len(finnhub_news_list)} Finnhub news results.")
        except Exception as e:
            logger.error(f"[get_market_summary] Finnhub news fetch error: {e}", exc_info=True)
            finnhub_news_list = []

        # 기사 병합 (중복 URL 제거)
        url_set = set()
        merged_news = []
        # Google 뉴스: url, title/headline, snippet/summary
        for item in google_news_list:
            url = item.get('url') or item.get('link')
            if url and url not in url_set:
                url_set.add(url)
                merged_news.append({
                    'url': url,
                    'headline': item.get('headline', '') or item.get('title', ''),
                    'summary': item.get('summary', '') or item.get('snippet', '')
                })
        # Finnhub 뉴스: url, headline, summary (본문은 fetch_article_text로 크롤링)
        for item in finnhub_news_list:
            url = item.get('url', '')
            if url and url not in url_set:
                url_set.add(url)
                merged_news.append({
                    'url': url,
                    'headline': item.get('headline', ''),
                    'summary': item.get('summary', '')
                })

        logger.info(f"[get_market_summary] Total merged news count: {len(merged_news)}")
        if not merged_news:
            logger.warning("No news collected from Google or Finnhub.")
            return "(관련 웹 정보를 가져올 수 없습니다.)"

        # 기사 본문 크롤링 및 요약 준비
        articles_for_prompt = []
        urls = [item.get("url") for item in merged_news]
        with ThreadPoolExecutor(max_workers=10) as pool:
            future_to_url = {pool.submit(self.fetch_article_text, url): url for url in urls if url}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    article_text = future.result()
                    headline = next((item.get("headline", "") for item in merged_news if item.get("url") == url), "")
                    summary = next((item.get("summary", "") for item in merged_news if item.get("url") == url), "")
                    # 기사 본문이 충분히 길면 본문, 아니면 summary/headline 사용
                    if article_text and len(article_text) > 200:
                        content = article_text
                    elif summary:
                        content = summary
                    else:
                        content = headline
                    if headline or content:
                        articles_for_prompt.append(f"제목: {headline.strip()}\n내용: {content.strip()}\nURL: {url if url else ''}\n---")
                except Exception as exc:
                    logger.error(f"Subquery '{url}' generated an exception: {exc}", exc_info=True)
        logger.info(f"[get_market_summary] articles_for_prompt length: {len(articles_for_prompt)}; first item: {articles_for_prompt[0] if articles_for_prompt else None}")
        if not articles_for_prompt:
            logger.warning("Could not extract usable news article contents.")
            return "(뉴스 내용을 처리할 수 없습니다.)"

        # 1차 요약: 기사 전체를 한 번에 LLM에 보내 중복 없이 핵심만 요약
        from src.utils.azure_openai import azure_chat_completion
        
        # 실시간 KST 시간 가져오기
        now_kst = datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")
        
        system_prompt_1 = (
            f"You are a summarization expert. The current local time is {now_kst} (KST). "
            f"You will be given multiple news articles, each clearly delimited and labeled. "
            f"Summarize the following articles in Korean, removing redundancy and focusing on the core facts."
        )
        user_content_1 = '\n'.join(articles_for_prompt)
        messages_1 = [
            {"role": "system", "content": system_prompt_1},
            {"role": "user", "content": user_content_1}
        ]
        resp_1 = azure_chat_completion(settings.AZURE_OPENAI_DEPLOYMENT_GPT4_1_NANO, messages=messages_1, max_tokens=8000, temperature=0.3)
        first_summary = resp_1["choices"][0]["message"]["content"].strip()
        logger.info(f"[요약] 1차 요약 완료 (기사 {len(articles_for_prompt)}개, 요약 길이: {len(first_summary)})")
        return first_summary


 # Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not settings.FINNHUB_API_KEY or not settings.AZURE_OPENAI_API_KEY:
        print("\nWarning: FINNHUB_API_KEY or AZURE_OPENAI_API_KEY not set. Summarization might fail.")
    
    crawler = InfoCrawler()
    test_query = "최근 시장 동향은 어떤가요?"
    summary = crawler.get_market_summary(test_query)
    print(f"\n--- Market Summary for query: '{test_query}' ---")
    print(summary) 