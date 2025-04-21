# 시장 이슈 수집·요약 
import logging
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from bs4 import BeautifulSoup
from src.agents.finnhub_client import FinnhubClient
from src.config import settings

logger = logging.getLogger(__name__)

# ✅ GPT-4o 또는 o4-mini 계열은 max_completion_tokens, 그 외는 max_tokens 사용 (SDK 최신 버전 기준)
def get_token_param(model: str, limit: int) -> dict:
    if model.startswith("o4") or model.startswith("gpt-4o"):
        return {"max_completion_tokens": limit}
    else:
        return {"max_tokens": limit}

def get_temperature_param(model: str, temperature: float) -> dict:
    if model.startswith("o4") or model.startswith("gpt-4o"):
        return {}  # 기본값 1.0만 지원
    else:
        return {"temperature": temperature}

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


                messages=messages,
                **get_temperature_param(settings.LLM_LIGHTWEIGHT_TIER_MODEL, 0.0),
                **get_token_param(settings.LLM_LIGHTWEIGHT_TIER_MODEL, 100)
            )

    
            translation = resp.choices[0].message.content.strip()
            logger.info(f"Translated query to English: {translation}")
            return translation
        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            return text


    def _summarize_with_llm(self, snippets, query):
        """LLM을 사용하여 수집된 스니펫을 요약합니다."""
        if not snippets:
            return "(수집된 정보가 없습니다)"
            
        # 정보가 너무 많으면 LLM 토큰 한도를 초과할 수 있으므로 제한
        combined_text = "\n---\n".join(snippets[:20])  # 최대 20개 스니펫으로 제한
        
        try:
            # OpenAI에 직접 요약 요청
            from openai import OpenAI
            # Azure OpenAI 전역 설정
            openai.api_type = "azure"
            openai.api_base = settings.AZURE_OPENAI_ENDPOINT
            openai.api_version = settings.AZURE_OPENAI_API_VERSION
            openai.api_key = settings.AZURE_OPENAI_API_KEY
            # 클라이언트는 api_key만 전달
            client = OpenAI(api_key=settings.AZURE_OPENAI_API_KEY)

            
            system_prompt = """
당신은 수집된 정보를 명확하고 간결하게 요약하는 전문가입니다.
수집된 텍스트 조각들을 분석하여 일관된 요약을 생성하세요.
서로 모순되는 정보가 있으면 그 점을 명시하고,
날짜나 시간이 언급된 경우 가장 최신 정보에 더 가중치를 두세요.
가능한 한 객관적으로 정보를 요약하되, 명확한 추세가 보이면 결론도 포함하세요.
"""
            user_prompt = f"다음 정보를 바탕으로 '{query}'에 대해 요약해주세요:\n\n{combined_text}"
            
            # Use lightweight tier model for summarization
            model_name = settings.LLM_LIGHTWEIGHT_TIER_MODEL
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **get_temperature_param(model_name, 0.3),
                **get_token_param(model_name, 500),
            )  # ← 닫는 괄호 추가
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
        news_list = google_results
        logger.info(f"[get_market_summary] Collected {len(news_list)} web results.")
        
        # Normalize news_list to a list to avoid slicing on non-list types
        if not isinstance(news_list, list):
            news_list = []
        # 방어: news_list가 None이거나 리스트가 아니면 빈 리스트 처리
        if not news_list or not isinstance(news_list, list):
            logger.warning("No web results fetched for market summary.")
            return "(관련 웹 정보를 가져올 수 없습니다.)"

        # 1차 요약: 각 기사별 핵심 요약 수행 (경량 LLM)
        articles_for_prompt = []
        urls = [item.get("url") or item.get("link") for item in news_list[:max_articles]]
        with ThreadPoolExecutor(max_workers=10) as pool:
            future_to_url = {pool.submit(self.fetch_article_text, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    article_text = future.result()
                    headline = next((item.get("headline", "") or item.get("title", "") for item in news_list if (item.get("url") == url or item.get("link") == url)), "")
                    summary = next((item.get("summary", "") or item.get("snippet", "") for item in news_list if (item.get("url") == url or item.get("link") == url)), "")
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
        import datetime, pytz
        now_kst = "2025-04-22 03:41:08"  # 시스템에서 주어진 최신 시간 사용
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
        resp_1 = azure_chat_completion(settings.AZURE_OPENAI_DEPLOYMENT_GPT4, messages=messages_1, max_tokens=800, temperature=0.3)
        first_summary = resp_1["choices"][0]["message"]["content"].strip()
        logger.info(f"[요약] 1차 요약 완료 (기사 {len(articles_for_prompt)}개, 요약 길이: {len(first_summary)})")
        return first_summary


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