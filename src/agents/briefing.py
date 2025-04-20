# Discord 브리핑 생성 

import logging
from datetime import datetime
from src.config import settings # Import settings for LLM config
from src.utils.azure_openai import azure_chat_completion
import json # To potentially parse complex details if needed

logger = logging.getLogger(__name__)

# GPT-4o 또는 o4-mini 계열은 max_completion_tokens, 그 외는 max_tokens 사용 (SDK 최신 버전 기준)
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

# --- OpenAI 모델 초기화 (BriefingAgent 용) ---
if not settings.OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. Briefing will be basic.")

class BriefingAgent:
    def __init__(self):
        # LLM client setup is handled globally
        logger.info("BriefingAgent initialized.")

    def _generate_llm_summary(self, execution_results: list) -> str:
        """OpenAI ChatCompletion을 사용해 실행 결과에 대한 요약을 생성합니다."""
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not set. Cannot generate summary.")
            return "(LLM 요약 생성 불가: API 키 미설정)"

        # Prepare context for LLM (remains the same)
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
            elif action_type == 'briefing_summary': # Handle notes from Orchestrator
                notes = result.get('notes', [])
                if notes:
                    summary_context += "- Orchestrator LLM 노트:\n"
                    for note in notes:
                         summary_context += f"  - {note}\n"
            # Add other action types if necessary
            
        prompt = f"""
다음은 자동매매 시스템의 일일 실행 결과입니다. 이 결과를 바탕으로 오늘 시장 상황과 실행된 주요 거래(성공/실패 포함), 그리고 주목할 만한 점을 포함하여 간결하고 이해하기 쉬운 한국어 요약 보고서를 작성해주세요.

{summary_context}

요약 보고서:
"""

        try:
            logger.info(f"Requesting OpenAI summary for briefing using {settings.LLM_LIGHTWEIGHT_TIER_MODEL}...")
            messages = [
                {"role": "system", "content": "You are an expert assistant that writes concise daily trading summary reports in Korean based on execution logs."}, # System prompt
                {"role": "user", "content": prompt}
            ]
            resp_json = azure_chat_completion(
                deployment=settings.LLM_LIGHTWEIGHT_TIER_MODEL,
                messages=messages,
                max_tokens=300,
                temperature=0.5
            )
            llm_summary = resp_json["choices"][0]["message"]["content"].strip()
            logger.info("Successfully received summary from OpenAI for briefing.")
            return llm_summary
        except Exception as e:
            logger.error(f"Azure OpenAI summarization failed for briefing: {e}", exc_info=True)
            return f"(LLM 요약 생성 중 오류 발생: {e})"

    def create_report_from_actions(self, execution_results: list) -> str:
        """Orchestrator의 실행 결과 리스트를 받아 Markdown 보고서를 생성합니다.
           (LLM 요약 포함)
        Args:
            execution_results: Orchestrator._execute_action_plan의 결과 리스트

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
        # 멀티라인 요약을 Markdown 인용블록 형태로 들여쓰기  
        indented = llm_summary.replace("\n", "\n> ")  
        report_parts.append(f"> {indented}")
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
    if not settings.OPENAI_API_KEY:
         print("\nWARNING: OPENAI_API_KEY not found in environment. LLM summary will be skipped.")

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