# Dag 관리·에이전트 호출 흐름

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
# TradingStrategy 모듈 사용이 필요없다면 import 삭제
# from src.agents.strategy import TradingStrategy # <- 이 줄 삭제됨
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
            # Import necessary components locally within the method
            from src.utils.discord_utils import DiscordRequestType
            from src.discord.bot import send_discord_request # Import the async function
            import asyncio
            
            # Ensure there's a running event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # If no loop is running (e.g., called from sync context without setup),
                # start a new one temporarily or handle differently.
                # For simplicity, we log a warning and skip.
                # A better approach might involve passing the loop or using a queue.
                logger.warning("No running event loop found to schedule Discord notification.")
                return
                
            # Schedule the async function call in the event loop
            asyncio.create_task(
                send_discord_request(DiscordRequestType.CYCLE_STATUS, {"step": step, "status": status})
            )
            logger.debug(f"Scheduled Discord notification for step: {step}, status: {status}")
            
        except ImportError as e:
            logger.error(f"Failed to import Discord components for notification: {e}. Is bot.py structured correctly?")
        except Exception as e:
            # Catch potential errors during scheduling or import
            logger.error(f"Error scheduling Discord notification for step {step} ({status}): {e}", exc_info=True)
            pass # Avoid crashing the orchestrator cycle due to notification failure

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

        logger.info("Orchestrator initialized all agents.")

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

            # 2.5. 추천 종목 산출 (Recommender 활용, 쿼리 기반)
            self._notify_step("Recommender", "시작")
            recommender = self.recommender if hasattr(self, 'recommender') else None
            recommendations = []
            if recommender:
                try:
                    rec_result = recommender.recommend(query=query)
                    recommendations = rec_result.get('recommendations', []) if isinstance(rec_result, dict) else rec_result
                    logger.info(f"Recommender output: {recommendations}")
                except Exception as e:
                    logger.error(f"Recommender failed: {e}")
            else:
                logger.warning("No recommender attached to Orchestrator; skipping recommendations.")
            self._notify_step("Recommender", "완료")

            # 3. 현재 포지션 조회
            self._notify_step("Get Positions", "시작")
            current_positions = self._get_current_positions()
            self._notify_step("Get Positions", "완료")

            # 3.5. 포트폴리오 상황 보고서 생성
            self._notify_step("Portfolio Report", "시작")
            portfolio_report = self._summarize_portfolio_status(current_positions)
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

    def _get_today(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")

# --- 중략 ---
    def _summarize_market_influences(self, summaries: list[str]) -> str:
        """
        InfoCrawler에서 수집한 뉴스/이슈 요약 리스트를 종합적으로 분석하여
        오늘 주가에 영향을 줄 수 있는 핵심 요인과 전략을 한글로 요약한다.
        """
        prompt = (
            "다음은 오늘 시장 및 내 포트폴리오 관련 뉴스/이슈 요약입니다.\n"
            "각 요약을 종합해 오늘 주가에 영향을 줄 수 있는 핵심 요인과 시장 전략을 한글로 정리해줘.\n"
            + "\n".join([f"{i+1}. {s}" for i, s in enumerate(summaries)])
        )
        # LLM(Azure OpenAI 등) 호출 예시
        return self.llm.generate(prompt)

    def _summarize_portfolio_status(self, positions: list[dict]) -> str:
        """
        KIS API로 수집한 내 포트폴리오 정보와 각 종목별 차트 분석 결과를 LLM에 전달해
        오늘의 포트폴리오 상황 보고서를 생성한다.
        (1) 각 종목별 차트 해석을 LLM이 직접 수행 (LLM 호출 1회/종목)
        (2) 그 결과와 포지션 정보를 합쳐 전체 포트폴리오 상황을 LLM이 종합 분석 (LLM 호출 1회)
        """
        chart_analysis = []
        for pos in positions:
            symbol = pos.get("symbol") or pos.get("code")
            name = pos.get("prdt_name") or pos.get("name")
            price_history = self._fetch_price_history(symbol)
            # LLM에 차트 해석 프롬프트 전달
            chart_prompt = (
                f"다음은 {name}({symbol})의 최근 가격 히스토리입니다. "
                "차트 변동성, 추세, 위험 신호, 투자 전략을 한글로 간단히 요약해줘.\n"
                f"[가격 데이터]\n{price_history}\n"
            )
            chart_summary = self.llm.generate(chart_prompt)
            chart_analysis.append({
                "name": name,
                "symbol": symbol,
                "chart_summary": chart_summary
            })
        # 전체 포트폴리오 상황 종합 보고서 프롬프트
        prompt = (
            "다음은 내 포트폴리오 보유 종목의 상세 정보와 각 종목별 차트 해석 요약입니다.\n"
            "각 종목별로 주목할 만한 변화, 위험 신호, 투자 전략을 한글로 요약해줘.\n"
            f"[포트폴리오]\n{positions}\n"
            f"[차트 해석 요약]\n{chart_analysis}\n"
        )
        return self.llm.generate(prompt)

    def _fetch_price_history(self, symbol: str):
        """
        Finnhub API를 사용해 심볼별 가격 히스토리(예: 일별 종가 30개) 조회
        """
        import requests
        import os
        api_key = os.getenv("FINNHUB_API_KEY")
        url = "https://finnhub.io/api/v1/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": "D",
            "count": 30,
            "token": api_key
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            # data['c']는 종가(close) 리스트 등
            return data
        else:
            logger.error(f"Finnhub price history fetch failed for {symbol}: {response.text}")
            return []
                self._notify_step("Daily Cycle", "종료: 행동 계획 없음")
                return

            try:
                action_plan = json.loads(action_plan_str)
                logger.info(f"LLM Action Plan: {action_plan}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM action plan JSON: {e}. Plan received: {action_plan_str}")
                self._notify_step("Daily Cycle", "오류: 행동 계획 파싱 실패")
                return

            # 6. 행동 계획 파싱·검증
            self._notify_step("RiskGuard Validation & Confirmation Request", "시작")
            execution_results = self._execute_action_plan(action_plan)
            if any(res.get("action_type") == "user_confirmation_request" and res.get("status") == "sent" for res in execution_results):
                self._notify_step("RiskGuard Validation & Confirmation Request", "완료 (승인 대기 중)")
            elif any(res.get("action_type") == "user_confirmation_request" and res.get("status") == "failed" for res in execution_results):
                self._notify_step("RiskGuard Validation & Confirmation Request", "오류 (승인 요청 실패)")
            else:
                self._notify_step("RiskGuard Validation & Confirmation Request", "완료 (실행할 주문 없음)")

            # 7. Briefing 생성 (추천 결과도 포함)
            self._notify_step("Briefing Generation", "시작")
            briefing_content = self._generate_briefing({
                'execution_results': execution_results,
                'recommendations': recommendations,
                'query': query
            })
            self._notify_step("Briefing Generation", "완료")

            # 8. 결과 저장
            self._notify_step("Save Results", "시작")
            self._upsert_trade_results(execution_results)
            self._notify_step("Save Results", "완료")

            logger.info("Daily cycle completed successfully (pending user confirmation if applicable). LLM orchestration.")
            self._notify_step("Daily Cycle", "완료 (승인 대기 중)" if any(res.get("action_type") == "user_confirmation_request" and res.get("status") == "sent" for res in execution_results) else "완료")
            self.send_notification(briefing_content)

        except Exception as e:
            logger.error(f"Daily cycle failed: {e}", exc_info=True)
            self._notify_step("Daily Cycle", "오류")
            self.send_notification(f"Daily cycle failed: {e}", is_error=True)

    def _get_llm_action_plan(self, market_summary, market_data, current_positions, rag_context, recommendations=None, query=None) -> str | None:
        """LLM에게 현재 상황 정보를 제공하고 행동 계획(JSON)을 요청합니다. (추천 종목/쿼리까지 포함)"""
        logger.info("Asking LLM for the daily action plan...")
        if not self.llm_model_name:
            return None

        # Build the prompt for the LLM
        prompt = f"""
        You are the master AI orchestrator for an ETF autotrading system. Your goal is to maximize returns while managing risk for a small investment amount ({settings.INVESTMENT_AMOUNT:,.0f} KRW).
        Today's Date: {datetime.now().strftime('%Y-%m-%d')}

        Today's Focus Query/Theme: {query}

        Current Market Situation:
        - Summary: {market_summary}
        - Key ETF Prices: {json.dumps({k: v for k, v in market_data.items() if v}, indent=2)}

        Current Portfolio:
        {json.dumps(current_positions, indent=2)}

        Recommendations from the recommender agent (based on query/theme):
        {json.dumps(recommendations, indent=2)}

        Relevant Past Context (from Memory/RAG):
        {rag_context}

        Based on all the above information, your analysis, and current best practices for ETF trading (consider momentum, mean reversion, market sentiment, risk management), decide the course of action for today.

        Your output MUST be a JSON list of actions. Each action should be a dictionary with 'action_type' and necessary parameters.
        Valid 'action_type' values are:
        - 'buy': Execute a buy order. Requires 'symbol' (str), 'quantity' (int). Price will be market price.
        - 'sell': Execute a sell order. Requires 'symbol' (str), 'quantity' (int). Price will be market price.
        - 'hold': No action required for a specific symbol or overall. Can optionally include 'reason' (str).
        - 'briefing': Add a specific insight or note to the daily briefing. Requires 'message' (str).

        Example JSON Output:
        [
          {{"action_type": "sell", "symbol": "069500", "quantity": 5, "reason": "Stop-loss triggered based on yesterday's drop"}},
          {{"action_type": "buy", "symbol": "229200", "quantity": 10, "reason": "Positive momentum signal and favorable market summary"}},
          {{"action_type": "hold", "reason": "Market conditions are uncertain, wait for clearer signals."}},
          {{"action_type": "briefing", "message": "Observed increased volatility in the energy sector ETFs."}}
        ]

        If no trades are recommended, return a list containing only a 'hold' or 'briefing' action, or an empty list [].
        Ensure quantities are integers and symbols are valid KIS ETF codes.
        Be mindful of the total investment amount and avoid over-allocation. Use the RiskGuard agent for final checks implicitly.

        Generate the JSON action plan now:
        """

        try:
            logger.info(f"Requesting Azure OpenAI action plan using {self.llm_model_name}...")
            messages = [
                {"role": "system", "content": "You are an AI assistant that generates JSON action plans for an ETF autotrading system based on provided market context and portfolio data."},
                {"role": "user", "content": prompt}
            ]
            resp_json = azure_chat_completion(
                deployment=self.llm_model_name,
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            text = resp_json["choices"][0]["message"]["content"].strip()
            
            # Extract JSON part (improved robustness)
            json_block = None
            if text.startswith("```json"):
                json_block = text[len("```json"):].split("```")[0].strip()
            elif text.startswith("[") and text.endswith("]"):
                json_block = text
            else:
                start, end = text.find('['), text.rfind(']')
                if start != -1 and end != -1:
                    json_block = text[start:end+1]
            
            if json_block:
                # Validate if it's actually JSON before returning
                try:
                    json.loads(json_block) # Try parsing
                    logger.info(f"Received JSON action plan from Azure OpenAI: {json_block}")
                    return json_block
                except json.JSONDecodeError:
                     logger.error(f"Extracted block is not valid JSON: {json_block}")
                     return None # Failed validation
            else:
                logger.error(f"Could not extract JSON action plan from LLM response: {text}")
                return None

        except Exception as e:
            logger.error(f"Azure OpenAI action plan generation failed: {e}", exc_info=True)
            return None

    def _execute_action_plan(self, action_plan: list) -> list:
        """파싱된 행동 계획을 단계적으로 실행합니다. (사용자 승인 단계 추가)
        """
        logger.info(f"Executing action plan with {len(action_plan)} steps...")
        execution_results = []
        orders_to_request_confirmation = [] # Changed variable name
        briefing_notes = []

        # 1. Parse actions and separate orders from other actions
        for action in action_plan:
            action_type = action.get('action_type')
            if action_type in ['buy', 'sell']:
                symbol = action.get('symbol')
                quantity = action.get('quantity')
                if symbol and isinstance(quantity, int) and quantity > 0:
                     # Add to potential orders - RiskGuard will validate later
                     orders_to_request_confirmation.append({
                         "symbol": symbol,
                         "action": action_type, # 'buy' or 'sell'
                         "quantity": quantity,
                         "price": 0, # Market order based on LLM instruction
                         "reason": action.get('reason', f'LLM directed {action_type}')
                     })
                else:
                    logger.warning(f"Invalid buy/sell action received from LLM: {action}")
                    execution_results.append({"action_type": action_type, "status": "invalid", "detail": action})
            elif action_type == 'briefing':
                message = action.get('message')
                if message:
                    briefing_notes.append(message)
                    execution_results.append({"action_type": action_type, "status": "noted", "detail": message})
            elif action_type == 'hold':
                logger.info(f"LLM directed 'hold': {action.get('reason', 'No specific reason')}")
                execution_results.append({"action_type": action_type, "status": "noted", "detail": action.get('reason')})
            else:
                 logger.warning(f"Unknown action type received from LLM: {action_type}")
                 execution_results.append({"action_type": action_type, "status": "unknown", "detail": action})

        # 2. Validate potential orders with RiskGuard
        if orders_to_request_confirmation:
            logger.info(f"Passing {len(orders_to_request_confirmation)} potential orders to RiskGuard...")
            validated_orders = self.risk_guard.validate_orders(orders_to_request_confirmation)
            logger.info(f"{len(validated_orders)} orders passed RiskGuard validation.")
        else:
            validated_orders = []

        # 3. Request User Confirmation via Discord (INSTEAD of direct execution)
        if validated_orders:
            confirmation_result = self._request_user_confirmation(validated_orders)
            execution_results.append(confirmation_result) # Record the request attempt

            # --- IMPORTANT ---
            # Actual order execution (_execute_orders) should happen *after* receiving
            # user approval (e.g., 'yes') from Discord.
            # This requires a mechanism for the Discord bot to communicate the user's
            # decision back to the Orchestrator (e.g., callback, API call, message queue).
            # Since that mechanism is not yet implemented, we are *NOT* calling
            # self._execute_orders here.
            #
            # Conceptual flow after user clicks 'Yes':
            # 1. Discord bot receives 'Yes' interaction.
            # 2. Bot notifies Orchestrator (e.g., calls an API endpoint on FastAPI).
            # 3. Orchestrator's endpoint handler receives the approved orders and calls:
            #    approved_order_results = self._execute_orders(approved_orders)
            #    self._upsert_trade_results(approved_order_results) # Save results
            #    self.send_notification("User approved orders executed.")
            #
            # For 'No' or 'Hold', log the decision and potentially save context to memory.
            # --- End Conceptual Flow ---

        else:
            logger.info("No orders require user confirmation.")

        # 4. Add briefing notes to results (always happens regardless of orders)
        execution_results.append({"action_type": "briefing_summary", "notes": briefing_notes})

        return execution_results # Return results including confirmation request status

    def _request_user_confirmation(self, orders_to_confirm: list) -> dict:
        """검증된 주문 목록을 Discord 봇으로 보내 사용자 승인을 요청합니다. (실제 전송은 추후 구현)
        """
        request_id = str(uuid.uuid4()) # Unique ID for this confirmation request
        logger.info(f"Requesting user confirmation via Discord for {len(orders_to_confirm)} orders. Request ID: {request_id}")

        # Prepare data payload for Discord bot
        payload = {
            "request_id": request_id,
            "orders": orders_to_confirm
        }

        try:
            # --- TODO: Implement actual communication with Discord Bot ---
            # This function (`send_discord_request`) needs to be implemented in `src/discord/bot.py`
            # or a shared communication module. It should handle:
            # - Formatting the orders into a user-friendly message (e.g., Embed).
            # - Adding Yes/No/Hold buttons.
            # - Sending the message to the appropriate Discord channel/user.
            # - Storing the request_id and orders temporarily to handle the user's response.
            # Example hypothetical call:
            # success = await send_discord_request(type=DiscordRequestType.ORDER_CONFIRMATION, data=payload)
            success = True # Placeholder: Assume request sent successfully
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