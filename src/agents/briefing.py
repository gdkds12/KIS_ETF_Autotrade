# Discord 브리핑 생성 

import logging
from datetime import datetime
from src.config import settings # Import settings for LLM config
import google.generativeai as genai # Import Gemini
import json # To potentially parse complex details if needed

logger = logging.getLogger(__name__)

# --- Gemini 모델 초기화 (BriefingAgent 용) --- 
briefing_llm_model = None
if settings.GOOGLE_API_KEY and settings.LLM_LIGHTWEIGHT_TIER_MODEL:
    try:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        briefing_llm_model = genai.GenerativeModel(settings.LLM_LIGHTWEIGHT_TIER_MODEL)
        logger.info(f"BriefingAgent initialized with LLM: {settings.LLM_LIGHTWEIGHT_TIER_MODEL}")
    except Exception as e:
         logger.error(f"Failed to initialize LLM for BriefingAgent: {e}", exc_info=True)
else:
    logger.warning("GOOGLE_API_KEY or LLM_LIGHTWEIGHT_TIER_MODEL not set. Briefing will be basic.")

class BriefingAgent:
    def __init__(self):
        # LLM client is initialized globally above
        self.llm_model = briefing_llm_model
        logger.info("BriefingAgent initialized.")

    def _generate_llm_summary(self, execution_results: list) -> str:
        """LLM을 사용하여 실행 결과에 대한 자연어 요약을 생성합니다."""
        if not self.llm_model:
            logger.warning("LLM not available for briefing summary.")
            return "(LLM 요약 생성 불가)"

        # Prepare context for LLM
        summary_context = "오늘 자동매매 사이클 실행 결과:\n"
        for result in execution_results:
            action_type = result.get('action_type', 'unknown')
            status = result.get('status', 'unknown')
            detail = result.get('detail', '')
            order_info = result.get('order', {})
            symbol = order_info.get('symbol')
            quantity = order_info.get('quantity')

            if action_type in ['buy', 'sell']:
                 summary_context += f"- {action_type.upper()} {symbol} {quantity}주 시도: {status.upper()}. 이유: {order_info.get('reason', 'N/A')}. 상세: {detail}\n"
            elif action_type == 'hold':
                 summary_context += f"- HOLD 결정. 이유: {detail}\n"
            elif action_type == 'briefing':
                 summary_context += f"- LLM 추가 노트: {detail}\n"

        prompt = f"""
다음은 자동매매 시스템의 일일 실행 결과입니다. 이 결과를 바탕으로 오늘 시장 상황과 실행된 주요 거래(성공/실패 포함), 그리고 주목할 만한 점을 포함하여 간결하고 이해하기 쉬운 한국어 요약 보고서를 작성해주세요.

{summary_context}

요약 보고서:
"""

        try:
            logger.info("Requesting LLM summary for briefing report...")
            response = self.llm_model.generate_content(prompt)
            llm_summary = response.text.strip()
            logger.info("Received LLM summary for briefing.")
            return llm_summary
        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}", exc_info=True)
            return f"(LLM 요약 생성 중 오류 발생: {e})"

    def create_report_from_actions(self, execution_results: list) -> str:
        """Orchestrator의 실행 결과 리스트를 받아 Markdown 보고서를 생성합니다.
           (LLM 요약 포함)
        Args:
            execution_results: Orchestrator._execute_action_plan의 결과 리스트
                Example: [
                    {'action_type': 'buy', 'status': 'success', 'detail': '...', 'kis_response': {...}, 'order': {...}},
                    {'action_type': 'sell', 'status': 'failed', 'detail': '...', 'kis_response': {...}, 'order': {...}},
                    {'action_type': 'hold', 'status': 'noted', 'detail': '...'},
                    {'action_type': 'briefing', 'status': 'noted', 'detail': '...'},
                    {'action_type': 'briefing_summary', 'notes': ['...']}
                ]

        Returns:
            생성된 Markdown 형식의 보고서 문자열
        """
        logger.info(f"Generating briefing report from {len(execution_results)} execution results...")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S KST")
        
        report_parts = []
        report_parts.append(f"## 📈 KIS ETF Autotrade Daily Report ({now_str}) 📊")
        report_parts.append("\n")

        # --- LLM 생성 요약 섹션 ---
        llm_summary = self._generate_llm_summary(execution_results)
        report_parts.append("**✨ AI 종합 브리핑 ✨**")
        report_parts.append(f"> {llm_summary.replace('\n', '\n> ')}") # Indent multiline summaries
        report_parts.append("\n")

        # --- 통계 요약 섹션 --- 
        summary = {"buy_success": 0, "sell_success": 0, "failed": 0, "hold": 0, "briefings": 0}
        llm_briefing_notes = []

        for result in execution_results:
            action_type = result.get('action_type')
            status = result.get('status')

            if action_type == 'buy' and status == 'success':
                summary["buy_success"] += 1
            elif action_type == 'sell' and status == 'success':
                summary["sell_success"] += 1
            elif action_type in ['buy', 'sell'] and status in ['failed', 'error']:
                summary["failed"] += 1
            elif action_type == 'hold':
                summary["hold"] += 1
            elif action_type == 'briefing':
                summary["briefings"] += 1
            elif action_type == 'briefing_summary':
                llm_briefing_notes.extend(result.get('notes', []))
        
        report_parts.append("**📊 Cycle Statistics:**")
        report_parts.append(f"- Successful Buys: {summary['buy_success']}")
        report_parts.append(f"- Successful Sells: {summary['sell_success']}")
        report_parts.append(f"- Failed/Errored Orders: {summary['failed']}")
        report_parts.append(f"- Hold Actions Noted: {summary['hold']}")
        report_parts.append(f"- LLM Briefing Notes Received: {summary['briefings'] + len(llm_briefing_notes)}")
        report_parts.append("\n")

        # --- LLM 브리핑 노트 섹션 (Orchestrator가 직접 전달한 노트) ---
        if llm_briefing_notes:
             report_parts.append("**📝 LLM Orchestrator Notes:**")
             for note in llm_briefing_notes:
                 report_parts.append(f"- {note}")
             report_parts.append("\n")

        # --- 상세 실행 결과 섹션 --- 
        report_parts.append("**⚙️ Execution Details:**")
        trade_actions_found = False
        for result in execution_results:
            action_type = result.get('action_type')
            status = result.get('status')
            detail = result.get('detail', 'N/A')
            order_info = result.get('order', {})
            symbol = order_info.get('symbol', 'N/A')
            quantity = order_info.get('quantity', 'N/A')
            
            if action_type in ['buy', 'sell']:
                trade_actions_found = True
                if status == 'success':
                    order_no = result.get('kis_response', {}).get('ODNO', 'N/A')
                    report_parts.append(f"- ✅ **{action_type.upper()} Success:** {symbol} {quantity} shares. Reason: {order_info.get('reason', 'N/A')} (Order#: {order_no})")
                elif status in ['failed', 'error']:
                    report_parts.append(f"- ❌ **{action_type.upper()} {status.upper()}:** {symbol} {quantity} shares. Reason: {order_info.get('reason', 'N/A')}. Detail: {detail}")
                else: # e.g., status == 'invalid' from RiskGuard
                     report_parts.append(f"- ⚠️ **{action_type.upper()} Skipped/Invalid:** {symbol} {quantity} shares. Reason: {order_info.get('reason', 'N/A')}. Detail: {detail}")

            # Optionally report hold/briefing actions here if needed
            # elif action_type == 'hold':
            #     report_parts.append(f"- ⏸️ HOLD: {detail}")
            # elif action_type == 'briefing' and status == 'noted':
            #      report_parts.append(f"- 📝 NOTE: {detail}")

        if not trade_actions_found:
            report_parts.append("- No trade orders were attempted or executed in this cycle.")
        
        report_parts.append("\n---\nEnd of Report")
        
        final_report = "\n".join(report_parts)
        logger.info(f"Generated briefing report (length: {len(final_report)}). Preview: {final_report[:200]}...")
        return final_report

# Example Usage (Conceptual)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("BriefingAgent class.")

    # Ensure API key is available for LLM summary test
    if not settings.GOOGLE_API_KEY:
         print("\nWARNING: GOOGLE_API_KEY not found in environment. LLM summary will be skipped.")

    agent = BriefingAgent()
    mock_results = [
        {'action_type': 'buy', 'status': 'success', 'detail': 'LLM directed buy submitted successfully.', 'kis_response': {'ODNO': '12345'}, 'order': {'symbol': '069500', 'action': 'buy', 'quantity': 10, 'price': 0, 'reason': 'Positive momentum'}},
        {'action_type': 'sell', 'status': 'failed', 'detail': 'sell order for 114800 failed: [APBK0013] Not enough balance/holding.', 'kis_response': {'rt_cd': '1', 'msg1': '잔고부족'}, 'order': {'symbol': '114800', 'action': 'sell', 'quantity': 5, 'price': 0, 'reason': 'Risk threshold breach'}},
        {'action_type': 'hold', 'status': 'noted', 'detail': 'Market uncertain.'},
        {'action_type': 'briefing', 'status': 'noted', 'detail': 'Keep an eye on oil prices.'},
        {'action_type': 'briefing_summary', 'notes': ['LLM decided to take partial profits on KODEX 200.', 'Monitoring KOSDAQ index closely.']}
    ]
    report = agent.create_report_from_actions(mock_results)
    print("\n--- Generated Report ---")
    print(report) 