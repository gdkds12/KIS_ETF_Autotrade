# orchestrator.py
import logging
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import json # For parsing LLM response
from src.utils.azure_openai import azure_chat_completion
import uuid
from datetime import datetime
import asyncio # Add asyncio import

# TODO: Import other agents (InfoCrawler, MemoryRAG, Strategy, RiskGuard, Broker, Briefing)

# Import necessary components
from src.config import settings
from src.brokers.kis import KisBroker, KisBrokerError
from src.agents.info_crawler import InfoCrawler
from src.agents.memory_rag import MemoryRAG
from src.agents.portfolio_manager import PortfolioManager
# TradingStrategy 모듈 사용이 필요없다면 import 삭제
# from src.executors.trade_executor import TradeExecutorategy # <- 이 줄 삭제됨
from src.agents.risk_guard import RiskGuard
from src.agents.briefing import BriefingAgent
from src.agents.finnhub_client import FinnhubClient # <-- FinnhubClient 추가
# from src.db.models import SessionLocal # If direct DB session needed here
from qdrant_client import QdrantClient
# Update import path for DiscordRequestType
from src.utils.discord_utils import DiscordRequestType
# from src.agents.kis_developer import KisDeveloper # Removed
# from src.utils.logger import setup_logger # Remove custom logger import

# NEW: function registry
from src.utils.registry import command, COMMANDS

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

# --- Helper for Retrying Broker Operations ---
# Define which KIS errors might be worth retrying (e.g., temporary network issues, maybe rate limits)
def is_retryable_kis_error(exception):
    if isinstance(exception, KisBrokerError):
        # Example: Retry on specific HTTP errors or KIS error codes
        # if isinstance(exception.__cause__, requests.exceptions.Timeout):
        #     return True
        # if exception.response_data and exception.response_data.get('msg_cd') == 'APBK08040': # Rate limit
        #      logger.warning("KIS Rate limit detected, retrying...")
        #      return True
        # For now, let's retry most KisBrokerErrors except auth failures maybe
        return True # Be cautious with retries
    return False # Don't retry other exception types by default

kis_retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(KisBrokerError), # Retry only specific KIS errors
    reraise=True # Reraise the exception if all retries fail
)

class Orchestrator:
    def _notify_step(self, step: str, status: str):
        """각 단계 시작/완료/오류 시 Discord로 메시지 전송."""
        try:
            # 실제 Discord Bot 호출
            success = asyncio.get_event_loop().run_until_complete(
                send_discord_request(DiscordRequestType.ORDER_CONFIRMATION, payload)
            )
            if success:
                logger.info(f"Successfully sent confirmation request to Discord (or placeholder success).")
                return {
                    "action_type": "user_confirmation_request",
                    "status": "sent",
                    "detail": f"Sent confirmation request for {len(orders_to_confirm)} orders.",
                    "request_id": request_id,
                    "orders_requested": orders_to_confirm
                }
            else:
                raise RuntimeError("Failed to send request to Discord bot.")
        except Exception as e:
             logger.error(f"Failed to send Discord confirmation request: {e}", exc_info=True)
             # Return error status, potentially retry or handle differently
             return {
                 "action_type": "user_confirmation_request",
                 "status": "failed",
        except Exception as e:
             logger.error(f"Failed to upsert execution results: {e}", exc_info=True)

    def __init__(self, broker: KisBroker, db_session_factory, qdrant_client: QdrantClient, finnhub_client: FinnhubClient, memory_rag: MemoryRAG):
        """Orchestrator 초기화

        Args:
            broker: KIS Broker 인스턴스
            db_session_factory: SQLAlchemy 세션 팩토리
            qdrant_client: Qdrant 클라이언트 인스턴스
            finnhub_client: Finnhub 클라이언트 인스턴스
            memory_rag: MemoryRAG 인스턴스
        """
        self.broker = broker
        self.kis = broker
        self.db_session_factory = db_session_factory
        self.qdrant_client = qdrant_client
        self.finnhub_client = finnhub_client # Assign finnhub_client
        self.memory_rag = memory_rag       # Assign memory_rag

        # Initialize OpenAI API Key
        if settings.AZURE_OPENAI_API_KEY:
            self.llm_model_name = settings.LLM_MAIN_TIER_MODEL
            logger.info(f"Orchestrator will use Azure OpenAI deployment: {self.llm_model_name}")
        else:
            logger.warning("OPENAI_API_KEY not set. Orchestrator LLM functionality will be disabled.")

        # Initialize Agents
        # self.kis = KisDeveloper(account_info=settings.KIS_ACCOUNT) # Remove KisDeveloper instantiation
        self.info_crawler = InfoCrawler()
        # Define target symbols - should ideally come from a dynamic source or config
        target_etfs = settings.TARGET_SYMBOLS # Use symbols from config
        # Strategy is now simplified or used differently, initialized later if needed
        # self.strategy = TradingStrategy(broker=self.broker,
        #                                 investment_amount=settings.INVESTMENT_AMOUNT,
        #                                 target_symbols=target_etfs)
        self.risk_guard = RiskGuard(broker=self.broker)
        self.briefing_agent = BriefingAgent() # LLM could be passed here too

        # PortfolioManager 인스턴스화
        self.portfolio_manager = PortfolioManager(self.broker)
        logger.info("Orchestrator initialized all agents and portfolio_manager.")

    # ------------------------------------------------------------------
    # LLM‑CALLABLE HELPER FUNCTIONS (REMOVED - now defined in registry.py)
    # ------------------------------------------------------------------

    def get_balance(self) -> dict:
        """현재 계좌의 예수금·총자산을 조회합니다."""
        return self.broker.get_balance()

    def get_positions(self) -> list[dict]:
        """현재 보유 포지션 목록을 조회합니다."""
        return self.broker.get_positions()

    def get_market_summary(self) -> str:
        """InfoCrawler 로부터 오늘의 시장 요약을 가져옵니다."""
        return self.info_crawler.get_market_summary()

    def run_daily_cycle(self):
        # 노후 메모리 삭제
        try:
            self.memory_rag.gc_old_memories()
            logger.info("Old memories collected and cleaned up.")
        except Exception as e:
            logger.warning(f"Failed to gc old memories: {e}")
        """일일 자동매매 사이클 실행 (AI 주도 쿼리 추천 → 정보 수집 → 추천 → 전략 → 실행)
        """
        self._notify_step("Daily Cycle", "시작")
        logger.info("Starting daily cycle with LLM orchestration...")
        
        if not self.llm_model_name:
            logger.error("LLM model name not set. Cannot run daily cycle.")
            self._notify_step("Daily Cycle", "오류: LLM 미설정")
            return

        try:
            # 0. 오늘의 시장 쿼리: 고정 쿼리 + 동적 쿼리(종목별)
            base_queries = [
                "오늘 미국 주요 ETF(SPY, QQQ, VTI 등) 일간 시세 변동 요인",
                "오늘 SPY·QQQ·VTI 실시간 가격 및 거래량 동향",
                "오늘 미국 배당형 ETF(VYM, DVY 등) 배당 발표 소식",
                "오늘 레버리지·인버스 ETF(TQQQ, SQQQ 등) 수익률 및 리스크",
                "오늘 미국 채권 ETF(TLT, BND 등) 금리 영향 뉴스",
                "오늘 기술 섹터 ETF(XLH, XLK 등) 주요 기업 실적 발표",
                "오늘 섹터별 ETF(금융·에너지·헬스케어) 퍼포먼스 비교",
                "오늘 미국 소형주 ETF(SPY, IWM 등) 시장 심리 및 변동성",
                "오늘 해외 신흥시장 ETF(EEM, VWO 등) 뉴스 요약",
                "오늘 ETF 순자산(AUM) 변동 및 신규 상장 소식",
                "오늘 연준(Fed) 금리 회의 주요 결정 및 시장 반응",
                "오늘 미국 의회(상·하원) 주요 입법 동향(예산·부채한도 등)",
                "오늘 백악관·재무부 정책 발표 및 관료 발언",
                "오늘 미 대선·정당내 경선 관련 최신 뉴스",
                "오늘 미국‑중국 무역협상·관세 이슈 업데이트"
            ]
            current_positions = self._get_current_positions()
            portfolio_names = [pos.get("prdt_name") or pos.get("name") for pos in current_positions if (pos.get("prdt_name") or pos.get("name"))]
            portfolio_queries = [f"{name} 주요 뉴스" for name in portfolio_names]
            all_queries = base_queries + portfolio_queries

            # 1. InfoCrawler: 각 쿼리별 15초 간격으로 검색 및 요약
            self._notify_step("Info Crawler", f"시작 (총 {len(all_queries)}개 쿼리)")
            summaries = []
            for idx, q in enumerate(all_queries):
                logger.info(f"[InfoCrawler] ({idx+1}/{len(all_queries)}) 쿼리: {q}")
                summary = self.info_crawler.get_market_summary(q)
                summaries.append(summary)
                if idx < len(all_queries) - 1:
                    logger.info("[InfoCrawler] API rate limit: 15초 대기...")
                    time.sleep(15)
            self._notify_step("Info Crawler", "완료")
            # summaries(요약본 리스트)로 오늘의 전략 통합 보고서 생성
            self._notify_step("Strategy Report", "시작")
            strategy_report = self._summarize_market_influences(summaries)
            self._notify_step("Strategy Report", "완료")

            # 2. 시장 데이터 수집
            self._notify_step("Market Fetch", "시작")
            market_data = self._fetch_market_data()
            self._notify_step("Market Fetch", "완료")

            # 2.5. 추천 종목 산출 (Recommender 활용)
            self.recommender = Recommender(
                self.finnhub_client,
                target_return=settings.TARGET_RETURN,
                risk_tolerance=settings.RISK_TOLERANCE,
                candidates=settings.CANDIDATE_SYMBOLS
            )
            self._notify_step("Recommender", "시작")
            try:
                rec = self.recommender.recommend()
                recommendations = rec.get('recommendations', [])
                logger.info(f"Recommender output: {recommendations}")
            except Exception as e:
                logger.error(f"Recommender failed: {e}")
                recommendations = []
            self._notify_step("Recommender", "완료")

            # 3. 현재 포지션 조회
            # 3. 포트폴리오 스냅샷
            self._notify_step("Portfolio Snapshot", "시작")
            snapshot = self.portfolio_manager.snapshot()
            self._notify_step("Portfolio Snapshot", "완료")

            # 3.5. 포트폴리오 상황 보고서 생성
            self._notify_step("Portfolio Report", "시작")
            # snapshot["positions"]와 snapshot["performance"] 활용
            portfolio_report = self._summarize_portfolio_status(snapshot["positions"])
            self._notify_step("Portfolio Report", "완료")

            # 4. 과거 메모리 검색
            self._notify_step("Memory RAG", "시작")
            rag_context = self._retrieve_relevant_memory(strategy_report)
            self._notify_step("Memory RAG", "완료")

            # 5. LLM 행동 계획 (추천 종목도 프롬프트에 포함)
            self._notify_step("LLM Action Plan", "시작")
            # --- LLM Action Plan 생성 단계 ---
            # 전략 요약 프롬프트 보강: 투자 원칙, 위험 신호, 포트폴리오 분산 등 강조
            strategy_prompt = (
                "아래는 오늘의 시장 뉴스 요약, 각 보유 종목의 차트 해석, 추천 종목, 과거 메모리입니다. "
                "이 모든 정보를 종합해 오늘의 투자 전략과 주요 근거, 주의할 점을 한글로 10문장 이내로 요약해줘. "
                "반드시 투자 원칙(분산, 리스크 관리, 과도한 집중 회피 등)과 위험 신호도 함께 언급할 것. "
                "전략은 구체적으로, 실전 투자자가 바로 참고할 수 있게 작성해줘."
                f"\n[시장 뉴스 요약]\n{market_data}\n"
                f"[포트폴리오 차트 해석]\n{portfolio_report}\n"
                f"[추천 종목]\n{recommendations}\n"
                f"[과거 메모리]\n{rag_context}\n"
            )
            action_plan_str = self.llm.generate(strategy_prompt)
            self._notify_step("LLM Action Plan", "완료")

            if not action_plan_str:
                logger.warning("LLM did not provide an action plan. Ending cycle.")
                return

            # --- RiskGuard 감사 단계 ---
            self._notify_step("Risk Audit", "리스크 감사 요청")
            risk_guard = self.risk_guard if hasattr(self, 'risk_guard') else None
            risk_audit_result = None
            if risk_guard:
                # 리스크 감사 프롬프트 보강: 구체적 위험 항목, 승인/거부 사유, 개선안 요청
                risk_audit_prompt = (
                    "아래는 오늘의 투자 전략 보고서, 포트폴리오 상황, 시장 데이터, 추천 종목입니다. "
                    "이 전략의 리스크(과도한 집중, 변동성, 손실 위험, 정책 리스크 등)를 구체적으로 평가하고, "
                    "위험 신호가 있다면 상세히 적고, 승인/거부를 반드시 명시해줘. "
                    "거부 시에는 반드시 개선 방안도 제시할 것. "
                    "결과는 JSON 형식으로: {\"approved\": true/false, \"summary\": \"...\", \"improvement\": \"...\"}"
                    f"\n[전략 보고서]\n{action_plan_str}\n"
                    f"[포트폴리오]\n{portfolio_report}\n"
                    f"[시장 데이터]\n{market_data}\n"
                    f"[추천 종목]\n{recommendations}\n"
                )
                risk_audit_result = risk_guard.audit_strategy(risk_audit_prompt)
                self._notify_step("Risk Audit", "감사 완료")
                if hasattr(risk_audit_result, 'approved') and not risk_audit_result.approved:
                    logger.warning("RiskGuard did not approve the strategy. Ending cycle.")
                    self._notify_step("Risk Audit", "거부/위험")
                    return
            else:
                logger.warning("No RiskGuard attached; skipping risk audit.")

            # --- Discord 결재 요청 단계 ---
            self._notify_step("Discord Approval", "전략 보고서 결재 요청")
            notifier = self.discord_notifier if hasattr(self, 'discord_notifier') else None
            if notifier:
                # Discord 결재 메시지 보강: 전략, 리스크 요약, 개선안, 승인/거부 안내
                discord_message = (
                    f"[전략 보고서]\n{action_plan_str}\n\n"
                    f"[포트폴리오]\n{portfolio_report}\n\n"
                    f"[리스크 감사 결과]\n{getattr(risk_audit_result, 'summary', risk_audit_result)}\n"
                    f"[리스크 개선안]\n{getattr(risk_audit_result, 'improvement', '')}\n\n"
                    "위 전략 실행을 승인하시겠습니까?\n[승인] [거부]"
                )
                approval = notifier.send_for_approval(
                    report=discord_message,
                    context={
                        "portfolio_report": portfolio_report,
                        "strategy_report": action_plan_str,
                        "risk_audit": risk_audit_result
                    }
                )
                if not approval:
                    logger.warning("Action plan was rejected or not approved by user. Ending cycle.")
                    self._notify_step("Discord Approval", "거부됨 또는 미승인")
                    return
                self._notify_step("Discord Approval", "승인 완료")
            else:
                logger.warning("No Discord notifier attached; skipping approval step.")

            # --- LLM 실행 함수 생성 단계 ---
            self._notify_step("LLM Trade Command", "실행 함수 생성")
            # 트레이드 명령 프롬프트 보강: 주문 조건, 리스크 경고, 예외 상황 안내
            trade_command_prompt = (
                "아래는 오늘의 전략 보고서, 포트폴리오, 시장 데이터, 추천 종목, 과거 메모리입니다. "
                "이 내용을 바탕으로 실제 매수/매도/관망 등 트레이딩을 위한 함수 호출 예시(심볼, 수량, 가격 등 포함)를 파이썬 코드로만 작성해줘. "
                "예시: kis_api.buy(symbol, qty, price), kis_api.sell(symbol, qty, price) 등. "
                "반드시 아래 조건을 지켜야 한다: 1) 투자 원칙(분산, 과도한 집중 회피, 손실 제한) 위반 금지, 2) 주문 수량/금액은 실제 투자 금액 내에서 합리적으로 산정, 3) 예외 상황(체결 불가, 잔고 부족 등)에는 아무 것도 하지 않는다. "
                "실행이 필요 없는 종목은 아무 것도 하지 말고, 반드시 함수 호출만 나열해줘."
                f"\n[전략 보고서]\n{action_plan_str}\n"
                f"[포트폴리오]\n{portfolio_report}\n"
                f"[시장 데이터]\n{market_data}\n"
                f"[추천 종목]\n{recommendations}\n"
                f"[과거 메모리]\n{rag_context}\n"
            )
            kis_trade_commands = self.llm.generate(trade_command_prompt)
            self._notify_step("LLM Trade Command", "생성 완료")

            # --- 실제 함수 실행 및 체결 처리 단계 ---
            self._notify_step("Trade Execution", "주문 실행")
            trade_results = self._execute_trade_commands(kis_trade_commands)
            self._notify_step("Trade Execution", "완료")

            # --- 하루 요약 및 DB 저장 단계 ---
            self._notify_step("Daily Summary", "요약 생성")
            # 하루 요약 프롬프트 보강: 전략의 성공/실패, 교훈, 내일 개선점 등 강조
            summary_prompt = (
                "아래는 오늘의 전략, 실제 체결 내역, 시장 데이터, 포트폴리오 정보입니다. "
                "이 모든 내용을 바탕으로 오늘 하루의 투자 전략, 실행 결과, 주요 교훈, 내일 개선점을 한글로 구체적으로 요약해줘. "
                "전략의 성공/실패 원인, 시장의 주요 변수, 내일을 위한 조언도 포함할 것."
                f"\n[전략 보고서]\n{action_plan_str}\n"
                f"[체결 내역]\n{trade_results}\n"
                f"[시장 데이터]\n{market_data}\n"
                f"[포트폴리오]\n{portfolio_report}\n"
            )
            daily_summary = self.llm.generate(summary_prompt)
            self._notify_step("Daily Summary", "DB 저장")
            if hasattr(self, 'memory_db'):
                self.memory_db.save_daily_summary(
                    date=self._get_today(),
                    summary=daily_summary,
                    trades=trade_results,
                    strategy=action_plan_str
                )
            self._notify_step("Daily Summary", "완료")

    def _execute_trade_commands(self, kis_trade_commands: str):
        """
        kis_trade_commands(파이썬 코드 문자열)을 안전하게 파싱/실행하여 실제 KIS API로 주문을 실행하고, 결과를 리스트로 반환
        """
        import ast
        results = []
        local_vars = {"kis_api": self.broker}
        try:
            tree = ast.parse(kis_trade_commands)
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    code = compile(ast.Module([node], []), '<string>', 'exec')
                    exec(code, {}, local_vars)
                    results.append(ast.unparse(node))
        except Exception as e:
            logger.error(f"Trade command execution failed: {e}")
            results.append(f"Execution error: {e}")
        return results

    # ── 구현된 주문 실행 메서드 ────────────────────────────────────────
    def _execute_orders(self, orders: list[dict]) -> list[dict]:
        """RiskGuard 통과한 orders를 실제로 실행합니다."""
        executor = TradeExecutor(self.broker)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(executor.execute_batch(orders))

    def _upsert_trade_results(self, execution_results: list[dict]):
        """체결 결과를 MemoryRAG 및 DB에 저장합니다."""
        if self.memory_rag:
            self.memory_rag.save_execution_results(execution_results)
        if hasattr(self, 'memory_db') and self.memory_db:
            try:
                self.memory_db.save_trade_results(execution_results)
            except Exception:
                logger.warning("DB에 체결 결과 저장 실패", exc_info=True)

    def _get_today(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")

    # ... (rest of the code remains the same)
            # -----------------------------------------------------------------

            if success:
                logger.info(f"Successfully sent confirmation request {request_id} to Discord (or placeholder success).")
                return {
                    "action_type": "user_confirmation_request",
                    "status": "sent",
                    "detail": f"Sent confirmation request for {len(orders_to_confirm)} orders.",
                    "request_id": request_id,
                    "orders_requested": orders_to_confirm
                }
            else:
                raise RuntimeError("Failed to send request to Discord bot.")

        except Exception as e:
             logger.error(f"Failed to send Discord confirmation request {request_id}: {e}", exc_info=True)
             # Return error status, potentially retry or handle differently
             return {
                 "action_type": "user_confirmation_request",
                 "status": "failed",
        except Exception as e:
             logger.error(f"Failed to upsert execution results: {e}", exc_info=True)

    def send_notification(self, message: str, is_error: bool = False):
         """Send notification (e.g., via Discord/Slack)."""
         level = "ERROR" if is_error else "INFO"
         log_message = f"NOTIFICATION [{level}]:\n{message[:1000]}..."
         if is_error:
             logger.error(log_message)
         else:
             logger.info(log_message)

         # TODO: Implement actual notification sending (e.g., call Discord bot method)
         # Example: Find the Discord bot instance and call its send method
         # discord_bot = get_discord_bot_instance() # How to get this?
         # if discord_bot:
         #    asyncio.create_task(discord_bot.send_channel_message(message))
         pass # Placeholder for actual sending

# Example Usage (conceptual)
if __name__ == "__main__":
    # This part requires setting up mock/real instances of broker, db, qdrant
    print("Orchestrator requires dependencies (broker, db, qdrant) to run.")
    # mock_broker = ...
    # mock_db = ...
    # mock_qdrant = ...
    # orchestrator = Orchestrator(mock_broker, mock_db, mock_qdrant)
    # orchestrator.run_daily_cycle()
    pass