# -*- coding:utf‑8 -*-
"""
Recommender – 사용자 프로필 + 시장/뉴스 데이터를 받아
오늘 매수 후보 종목(short‑list)을 산출
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class UserProfile:
    """간단한 사용자 성향 보관(예: DB나 설정에서 로드)"""
    def __init__(self, risk_level: str = "medium", sectors: List[str] | None = None):
        self.risk_level = risk_level                  # low/medium/high
        self.preferred_sectors = sectors or []        # ["IT", "반도체", ...]

class Recommender:
    def __init__(self, user_profile: UserProfile):
        self.profile = user_profile

    # ------------------------------------------------------------------
    def recommend(self,
                  news_summary: str,
                  market_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Args
        ----
        news_summary : InfoCrawler LLM 요약본문  
        market_data  : {symbol: {stck_prpr, prdy_ctrt, ...}, …}

        Returns
        -------
        symbols : 매수 후보 종목 코드 리스트 (최대 5~10개)
        """
        logger.info("Generating recommendations…")
        # TODO 1) rule + profile + 뉴스 키워드로 섹터 필터
        # TODO 2) price/모멘텀 조건(전일 대비, 거래대금 등)
        # 일단 데모: 변동률 +profile 섹터 미사용
        sorted_by_ctrt = sorted(
            [(s, v) for s, v in market_data.items() if v],
            key=lambda x: abs(float(x[1].get("prdy_ctrt", 0))), reverse=True
        )
        top_symbols = [s for s, _ in sorted_by_ctrt[:5]]
        logger.debug(f"Top symbols by pct‑change: {top_symbols}")
        return top_symbols
