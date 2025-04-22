"""
ReportGenerator – 기존 BriefingAgent 감싸고,
  뉴스·추천·전략·리스크 검증 결과를 종합 보고서로 만든다.
"""
import logging
from typing import List, Dict, Any
from src.agents.briefing import BriefingAgent

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        self.briefing = BriefingAgent()

    def generate(self,
                 news_summary: str,
                 recommendations: List[str],
                 strategy_actions: List[Dict[str, Any]],
                 risk_notes: List[str] | None = None) -> str:
        # 1) 기존 BriefingAgent로 실행 결과 요약
        exec_results = [{"action_type": a["action_type"],
                         "status": "pending",
                         "detail": a.get("reason", "")} for a in strategy_actions]
        base = self.briefing.create_report_from_actions(exec_results)

        # 2) 상단에 뉴스·추천 섹션 추가
        rec_list = "\n".join([f"- {s}" for s in recommendations]) or "N/A"
        risk_txt = "\n".join([f"- {n}" for n in (risk_notes or [])]) or "N/A"

        final_md = (
            f"### 🗞️ 오늘의 뉴스 요약\n\n{news_summary}\n\n"
            f"### 📌 추천 종목\n{rec_list}\n\n"
            f"{base}\n\n"
            f"### 🚦 RiskGuard 메모\n{risk_txt}"
        )
        return final_md
