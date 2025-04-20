import requests
import logging

logger = logging.getLogger(__name__)

class BingSearchClient:
    """
    Bing Web Search v7 클라이언트.
    Azure Cognitive Services > Bing Search v7 리소스에서 발급받은
    API Key를 사용해 웹 검색 결과를 JSON으로 반환합니다.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"

    def search(self, query: str, count: int = 5) -> list[dict]:
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key
        }
        params = {
            "q": query,
            "count": count
        }
        try:
            resp = requests.get(self.endpoint, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("webPages", {}).get("value", [])
        except Exception as e:
            logger.error(f"Bing Search error for query '{query}': {e}", exc_info=True)
            return []}},{
