# Discord 브리핑 생성 

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BriefingAgent:
    def __init__(self):
        """BriefingAgent 초기화"""
        logger.info("BriefingAgent initialized.")

    def create_markdown_report(self, order_results: list, portfolio_status: dict = None, market_summary: str = None) -> str:
        """거래 결과, 포트폴리오 상태, 시장 요약을 포함하는 Markdown 보고서 생성

        Args:
            order_results: Orchestrator가 반환한 주문 실행 결과 리스트
                           (KisBroker 응답 형식 및 원본 주문 정보 포함 가정)
            portfolio_status: 현재 포트폴리오 상태 (옵션, 예: 자산 가치, PnL)
            market_summary: InfoCrawler가 생성한 시장 요약 (옵션)

        Returns:
            Discord Embed 에 적합한 Markdown 형식 문자열
        """
        logger.info(f"Generating briefing report for {len(order_results)} order results.")
        report_parts = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- 보고서 제목 --- 
        report_parts.append(f"## 📊 일일 자동매매 결과 ({now_str})")
        report_parts.append("---")

        # --- 시장 요약 (제공된 경우) --- 
        if market_summary:
            report_parts.append("### 📰 시장 동향 요약")
            # Discord embed 필드 값 길이는 1024자 제한 유의
            summary_preview = market_summary.replace('>', '') # '>' 문자는 인용구로 해석될 수 있으므로 제거
            report_parts.append(f"> {summary_preview[:1000]}{'...' if len(summary_preview) > 1000 else ''}")
            report_parts.append("\n")

        # --- 주문 실행 결과 요약 --- 
        report_parts.append("### 🚀 주문 실행 결과")
        success_count = sum(1 for r in order_results if r.get("rt_cd") == "0")
        fail_count = len(order_results) - success_count
        
        if not order_results:
             report_parts.append("- 실행된 주문이 없습니다.")
        else:
            report_parts.append(f"- **총 주문 시도**: {len(order_results)} 건")
            report_parts.append(f"- **✅ 성공**: {success_count} 건")
            report_parts.append(f"- **❌ 실패**: {fail_count} 건")
            report_parts.append("\n**상세 내역:**")

            for i, result in enumerate(order_results):
                status_icon = "✅" if result.get("rt_cd") == "0" else "❌"
                msg = result.get("msg1", "(메시지 없음)")
                order_detail_str = ""
                
                # 원본 주문 정보 추출 (Orchestrator가 실패/성공 결과에 'order' 키로 넣어준다고 가정)
                order_info = result.get("order", {})
                symbol = order_info.get('symbol', '?')
                action = order_info.get('action', '?').upper()
                quantity = order_info.get('quantity', '?')
                price_type = "시장가" if order_info.get('price', -1) == 0 else f"{order_info.get('price', '?')}원"
                
                if symbol != '?':
                    order_detail_str = f" ({action} {symbol} {quantity}주 @ {price_type})"
                
                # 성공 시 추가 정보 (주문 번호 등)
                if result.get("rt_cd") == "0" and "output" in result:
                    order_no = result["output"].get("ODNO", "N/A")
                    order_detail_str += f" [주문번호:{order_no}]"
                
                report_parts.append(f"{i+1}. {status_icon} **{msg}**{order_detail_str}")
        
        report_parts.append("\n")

        # --- 포트폴리오 상태 (제공된 경우) --- 
        if portfolio_status:
            report_parts.append("### 📈 포트폴리오 현황")
            try:
                total_value = portfolio_status.get('total_asset_value', 0)
                cash = portfolio_status.get('available_cash', 0)
                pnl_total = portfolio_status.get('total_pnl', 0)
                pnl_percent = portfolio_status.get('total_pnl_percent', 0)
                positions = portfolio_status.get('positions', [])
                
                report_parts.append(f"- **총 평가 자산**: {total_value:,.0f} KRW")
                report_parts.append(f"- **총 평가 손익**: {pnl_total:,.0f} KRW ({pnl_percent:+.2f}%)")
            except Exception as e:
                report_parts.append(f"- **오류**: {e}")
        
        return "\n".join(report_parts) 