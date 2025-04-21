# 시장 이슈 수집·요약 
import logging, os, time, requests
from openai import OpenAI
from src.agents.finnhub_client import FinnhubClient
from src.config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed # Import concurrent features

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

    def _translate_to_en(self, text: str) -> str:
        """Translate Korean text to English using Azure OpenAI."""
        if not text:
            return text
        # If no Korean characters, return as-is
        import re
        if not re.search(r"[\uac00-\ud7a3]", text):
            return text
        try:
            client = OpenAI(api_key=settings.AZURE_OPENAI_API_KEY)
            messages = [
                {"role": "system", "content": "You are a translator that translates Korean to English."},
                {"role": "user", "content": f"Translate the following into English: {text}"}
            ]
            resp = client.chat.completions.create(
                model=settings.LLM_LIGHTWEIGHT_TIER_MODEL,
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

    # New method to search news using Bing Grounding
    def search_news(self, query: str = None, category: str = "general") -> list[dict]:
        """
        Finnhub API를 이용한 최신 뉴스 검색.
        - `query` 가 주어지면 결과를 제목/요약에 포함된 키워드 기준으로 필터링한다.
        """
        try:
            raw = self.finnhub.client.general_news(category=category)[:50]
            if query:
                q = query.lower()
                raw = [n for n in raw if q in n.get("headline", "").lower() or q in n.get("summary", "").lower()]
            return raw[:20]
        except Exception as e:
            logger.error(f"Finnhub news search error: {e}", exc_info=True)
            return []

    # New method to search symbols using requests
    def search_web(self, query: str, num_results: int = 5) -> list[dict]:
        """Google Custom Search JSON API를 이용한 일반 웹 검색"""
        if not query:
            logger.warning("Empty query for web search.")
            return []
        # Debug: verify API credentials
        if not settings.GOOGLE_API_KEY or not settings.GOOGLE_CX:
            logger.error("Google CSE credentials missing. Set GOOGLE_API_KEY and GOOGLE_CX in .env.")
            return []
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": settings.GOOGLE_API_KEY,
            "cx": settings.GOOGLE_CX,
            "q": query,
            "num": num_results
        }
        logger.debug(f"search_web: Sending request to {url} with params {params}")
        logger.debug(f"search_web: env GOOGLE_API_KEY prefix={settings.GOOGLE_API_KEY[:4]}..., GOOGLE_CX={settings.GOOGLE_CX}")
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            # Handle non-200 responses (e.g., 403 Forbidden) gracefully
            if resp.status_code != 200:
                logger.warning(f"Google Custom Search returned {resp.status_code} for query '{query}': {resp.text}")
                return []
            logger.debug(f"search_web: Received HTTP {resp.status_code}")
            data = resp.json()
            items = data.get("items", [])
            logger.debug(f"search_web: Retrieved {len(items)} items")
            results = []
            for item in items:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet")
                })
            return results
        except Exception as e:
            logger.error(f"Google Custom Search error for query '{query}': {e}", exc_info=True)
            return []
             
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

            
            system_prompt = (
                "당신은 수집된 정보를 명확하고 간결하게 요약하는 전문가입니다. "
                "수집된 텍스트 조각들을 분석하여 일관된 요약을 생성하세요. "
                "서로 모순되는 정보가 있으면 그 점을 명시하고, "
                "날짜나 시간이 언급된 경우 가장 최신 정보에 더 가중치를 두세요. "
                "가능한 한 객관적으로 정보를 요약하되, 명확한 추세가 보이면 결론도 포함하세요."
            )
            
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
            eng_query = self._translate_to_en(user_query)
            logger.info(f"Searching news relevant to query: {eng_query!r}")
            news_list = self.search_news(query=eng_query, category='general')
        else:
            logger.info("Searching general news as no specific query provided.")
            news_list = self.search_news(category='general')
        
        # Normalize news_list to a list to avoid slicing on non-list types
        if isinstance(news_list, dict):
            news_list = [news_list]
        elif not isinstance(news_list, list):
            news_list = []
        if not news_list:
            logger.warning("No news fetched from Finnhub for market summary.")
            return "(최신 시장 뉴스를 가져올 수 없습니다.)"

        # 1차 요약: 각 기사별 핵심 요약 수행 (경량 LLM)
        articles_for_prompt = []
        import requests
        from bs4 import BeautifulSoup
        def fetch_article_text(url):
            try:
                resp = requests.get(url, timeout=7)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Try to extract main content heuristically
                # 1. Look for <article>
                article_tag = soup.find('article')
                if article_tag:
                    return article_tag.get_text(separator=' ', strip=True)
                # 2. Fallback: largest <div> by text length
                divs = soup.find_all('div')
                if divs:
                    div = max(divs, key=lambda d: len(d.get_text()))
                    if len(div.get_text()) > 200:
                        return div.get_text(separator=' ', strip=True)
                # 3. Fallback: all <p> tags joined
                ps = soup.find_all('p')
                if ps:
                    text = ' '.join([p.get_text(separator=' ', strip=True') for p in ps])
                    if len(text) > 200:
                        return text
                return ""
            except Exception as e:
                logger.warning(f"Failed to fetch article text from {url}: {e}")
                return ""

        for idx, item in enumerate(news_list[:max_articles], 1):
            headline = item.get("headline", "")
            summary = item.get("summary", "") or item.get("source", "")
            url = item.get("url") or item.get("link")
            article_text = ""
            if url:
                article_text = fetch_article_text(url)
            if article_text and len(article_text) > 200:
                content = article_text
            elif summary:
                content = summary
            else:
                content = headline
            if headline or content:
                articles_for_prompt.append(f"[기사 {idx}]\n제목: {headline.strip()}\n내용: {content.strip()}\nURL: {url if url else ''}\n---")
        if not articles_for_prompt:
            logger.warning("Could not extract usable news article contents.")
            return "(뉴스 내용을 처리할 수 없습니다.)"

        # 1차 요약: 기사 전체를 한 번에 LLM에 보내 중복 없이 핵심만 요약
        from src.utils.azure_openai import azure_chat_completion
        import datetime, pytz
        now_kst = datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
        system_prompt_1 = (
            f"You are a summarization expert. The current local time is {now_kst} (KST). "
            f"You will be given multiple news articles, each clearly delimited and labeled. "
            f"Summarize the following articles in Korean, removing redundancy and focusing on the core facts. Do not answer the user query yet."
        )
        user_content_1 = '\n'.join(articles_for_prompt)
        messages_1 = [
            {"role": "system", "content": system_prompt_1},
            {"role": "user", "content": user_content_1}
        ]
        resp_1 = azure_chat_completion(settings.AZURE_OPENAI_DEPLOYMENT_GPT4, messages=messages_1, max_tokens=800, temperature=0.3)
        first_summary = resp_1["choices"][0]["message"]["content"].strip()
        logger.info(f"Generated first-phase summary from {len(articles_for_prompt)} articles (length: {len(first_summary)}).")

        # 2차 요약: 1차 요약 결과와 사용자 질문을 함께 LLM에 보내 최종 답변 생성
        system_prompt_2 = (
            f"You are a trading assistant. The current local time is {now_kst} (KST). "
            f"You will be given a summary of recent news articles. Answer the user's question below based on this summary, prioritizing recency and relevance. Answer in Korean."
        )
        user_content_2 = f"요약된 뉴스:
{first_summary}

사용자 질문: {user_query}"
        messages_2 = [
            {"role": "system", "content": system_prompt_2},
            {"role": "user", "content": user_content_2}
        ]
        resp_2 = azure_chat_completion(settings.AZURE_OPENAI_DEPLOYMENT_GPT4, messages=messages_2, max_tokens=800, temperature=0.3)
        final_answer = resp_2["choices"][0]["message"]["content"].strip()
        logger.info(f"Generated final answer from summarized news (length: {len(final_answer)}).")
        return final_answer


        # Second-phase: final summary using main LLM
        combined_first = "\n".join([f"- {ms}" for ms in item_summaries])
        from src.utils.azure_openai import azure_chat_completion
        import datetime, pytz
        now_kst = datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
        system_prompt2 = (
            f"You are a trading assistant summarizing market news. The current local time is {now_kst} (KST). "
            "Use the first-phase summaries below to generate a concise, relevant, and up-to-date final summary "
            f"for the user query '{user_query}'."
        )
        messages = [
            {"role": "system", "content": system_prompt2},
            {"role": "user", "content": combined_first}
        ]
        resp2 = azure_chat_completion(settings.AZURE_OPENAI_DEPLOYMENT_GPT4, messages=messages, max_tokens=500, temperature=0.3)
        final_summary = resp2["choices"][0]["message"]["content"].strip()
        logger.info(f"Generated final market summary (length: {len(final_summary)}).")
        return final_summary

    def multi_search(self, query: str, attempts: int = 3, max_attempts: int = 10) -> dict:
        """범용 검색: query 기반으로 최소 3번, 최대 10번의 news/web 검색을 병렬 수행해 LLM으로 요약."""
        logger.info(f"Performing multi-search for query: '{query}' with {attempts} attempts (max {max_attempts})")
        # 1) 시도 횟수 보정 (사용자 지정 attempts 반영, 1~max_attempts 범위)
        tries = 1  # 임시: 한번만 검색 수행

        # 2) 키워드 확장용 suffix 리스트 (attempts에 맞춰 다양한 suffix 사용)
        suffixes = [
            "최신 뉴스", "시장 동향", "분석", "전망", "이슈", "주요 토픽", "핫토픽"
        ]
        subqueries_set = {query}
        for s in suffixes:
            if len(subqueries_set) >= tries:
                break
            subqueries_set.add(f"{query} {s}")
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
        summary = self._summarize_with_llm(snippets, query)
        
        # Return a dictionary with summary and metadata
        return {
            "summary": summary,
            "subqueries_count": len(subqueries),
            "snippets_count": len(snippets)
        }

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