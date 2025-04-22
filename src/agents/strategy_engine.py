"""
StrategyEngine – 최고 성능 LLM에게 오늘 전략(JSON) 요청
"""
import json, logging
from datetime import datetime
from typing import List, Dict, Any
from src.utils.azure_openai import azure_chat_completion
from src.config import settings

logger = logging.getLogger(__name__)

class StrategyEngine:
    def __init__(self,
                 model_deployment: str = settings.AZURE_OPENAI_DEPLOYMENT_GPT4):
        self.model = model_deployment

    # ------------------------------------------------------------------
    def get_daily_plan(self,
                       news_summary: str,
                       recommendations: List[str],
                       portfolio_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """LLM 호출→ JSON action list 반환 (buy/sell/hold)"""
        prompt = f"""
### Context
Date (KST): {datetime.now().strftime('%Y‑%m‑%d')}
News Summary:
{news_summary}

Recommended Tickers: {recommendations}
Portfolio Snapshot:
{json.dumps(portfolio_snapshot, indent=2, ensure_ascii=False)}

### Task
Return ONLY a JSON array.  
Each item = {{
  "action_type": "buy"|"sell"|"hold",
  "symbol": "<ticker>",
  "quantity": <int>,          # positive integer
  "reason": "<short explanation>"
}}
- Use KR ETF codes for domestic, US tickers for overseas.
- Total BUY cost must not exceed available cash.
"""
        messages = [
            {"role": "system", "content": "You are an expert ETF auto‑trader assistant."},
            {"role": "user", "content": prompt}
        ]
        resp = azure_chat_completion(
            deployment=self.model,
            messages=messages,
            max_tokens=8000,
            temperature=0.3
        )
        raw = resp["choices"][0]["message"]["content"].strip()
        try:
            plan = json.loads(raw.split("```")[-1] if raw.startswith("```") else raw)
            logger.info(f"LLM plan parsed with {len(plan)} actions")
            return plan
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM plan; raw content kept for debug")
            raise
