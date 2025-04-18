# Dag 관리·에이전트 호출 흐름 

import logging
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# TODO: Import other agents (InfoCrawler, MemoryRAG, Strategy, RiskGuard, Broker, Briefing)

# Import necessary components
from src.config import settings
from src.brokers.kis import KisBroker, KisBrokerError
from src.agents.info_crawler import InfoCrawler
from src.agents.memory_rag import MemoryRAG
from src.agents.strategy import TradingStrategy
from src.agents.risk_guard import RiskGuard
from src.agents.briefing import BriefingAgent
# from src.db.models import SessionLocal # If direct DB session needed here
from qdrant_client import QdrantClient
# from some_notification_client import NotificationClient # e.g., for Slack/Telegram

logger = logging.getLogger(__name__)

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
    def __init__(self, broker: KisBroker, db_session, qdrant_client: QdrantClient):
        """Orchestrator 초기화

        Args:
            broker: KIS Broker 인스턴스
            db_session: SQLAlchemy 세션
            qdrant_client: Qdrant 클라이언트 인스턴스
        """
        self.broker = broker
        self.db_session = db_session
        self.qdrant_client = qdrant_client
        # TODO: Initialize other agents
        # self.info_crawler = InfoCrawler()
        # self.memory_rag = MemoryRAG(qdrant_client, db_session)
        # self.strategy = Strategy()
        # self.risk_guard = RiskGuard()
        # self.briefing = Briefing()
        logger.info("Orchestrator initialized.")

        # Initialize Agents
        # TODO: Pass necessary configurations or dependencies to agents
        self.info_crawler = InfoCrawler() # LLM model could be passed here
        self.memory_rag = MemoryRAG(db_session=db_session) # Uses Qdrant client via settings
        # TODO: Define target symbols from config or elsewhere
        target_etfs = ['069500', '229200', '114800'] # Example: KODEX 200, KOSDAQ 150, KODEX Inverse
        self.strategy = TradingStrategy(broker=self.broker, 
                                        investment_amount=settings.INVESTMENT_AMOUNT, 
                                        target_symbols=target_etfs)
        self.risk_guard = RiskGuard(broker=self.broker)
        self.briefing_agent = BriefingAgent()
        # self.notification_client = NotificationClient(webhook_url=settings.SLACK_WEBHOOK) # Example

    def run_daily_cycle(self):
        """일일 자동매매 사이클 실행"
        """
        logger.info("Starting daily cycle...")
        try:
            # 1. InfoCrawler: 시장 정보 수집 및 요약
            market_summary = self._run_info_crawler()

            # 2. Memory Summarize & Upsert (이전 세션 요약 등)
            self._summarize_and_upsert_memory()

            # 3. Market Fetch (시세 등)
            market_data = self._fetch_market_data()

            # 4. Strategy: 투자 전략 실행
            trading_signals = self._run_strategy(market_data, market_summary)

            # 5. RiskGuard: 주문 리스크 검증
            validated_orders = self._run_risk_guard(trading_signals)

            # 6. Broker Execute: 주문 실행
            order_results = self._execute_orders(validated_orders)

            # 7. Briefing: 결과 브리핑 생성
            briefing_content = self._generate_briefing(order_results)

            # 8. Memory Upsert (금일 거래 결과 등)
            self._upsert_trade_results(order_results)

            logger.info("Daily cycle completed successfully.")
            # TODO: Send success notification (e.g., Slack, Discord)
            print(briefing_content) # 임시 출력

        except Exception as e:
            logger.error(f"Daily cycle failed: {e}", exc_info=True)
            # TODO: Implement retry logic and send failure alert

    def _run_info_crawler(self):
        logger.info("Running Info Crawler...")
        # TODO: Implement info_crawler call
        # return self.info_crawler.get_market_summary()
        return "Market is stable today." # Placeholder

    def _summarize_and_upsert_memory(self):
        logger.info("Summarizing and upserting memory...")
        # TODO: Implement memory summarization and upsert logic
        # self.memory_rag.summarize_recent_sessions()
        pass

    @kis_retry_decorator 
    def _fetch_market_data(self):
        logger.info("Fetching market data...")
        # TODO: Implement market data fetching (e.g., using broker)
        # return self.broker.get_quotes([...]) # 필요한 종목 리스트
        market_data = {}
        # TODO: Get target symbols from strategy or config
        symbols = self.strategy.target_symbols 
        for symbol in symbols:
            try:
                # This call within the loop might need individual retries or batching
                quote = self.broker.get_quote(symbol)
                market_data[symbol] = {
                    'stck_prpr': quote.get('stck_prpr'), # 현재가
                    # Add other relevant fields from quote if needed by strategy
                    'prdy_vrss': quote.get('prdy_vrss'), # 전일대비
                    'prdy_ctrt': quote.get('prdy_ctrt'), # 전일대비율
                }
                logger.debug(f"Fetched quote for {symbol}: {market_data[symbol]}")
                time.sleep(0.1) # Basic rate limiting between KIS calls
            except KisBrokerError as e:
                logger.error(f"Failed to fetch quote for {symbol} after retries: {e}")
                # Decide whether to continue or fail the whole step
                # raise # Re-raise to fail the step if one quote fails
                market_data[symbol] = None # Or mark as failed
            except Exception as e:
                 logger.error(f"Unexpected error fetching quote for {symbol}: {e}", exc_info=True)
                 market_data[symbol] = None
        logger.info(f"Market data fetched for {len(market_data)} symbols.")
        return market_data

    def _run_strategy(self, market_data, market_summary):
        logger.info("Running Strategy...")
        # TODO: Implement strategy call
        # return self.strategy.generate_signals(market_data, market_summary)
        # Placeholder: Buy 1 share of KODEX 200 if stable
        if "stable" in market_summary:
            return [{"symbol": "069500", "action": "buy", "quantity": 1, "price": 35000}]
        return []

    def _run_risk_guard(self, trading_signals):
        logger.info("Running Risk Guard...")
        # TODO: Implement risk_guard call
        # return self.risk_guard.validate_orders(trading_signals)
        return trading_signals # Placeholder: No filtering for now

    @kis_retry_decorator
    def _execute_orders(self, validated_orders):
        logger.info(f"Executing {len(validated_orders)} orders...")
        results = []
        for order in validated_orders:
            order_result = None
            action_desc = f"{order['action']} order for {order['symbol']}"
            try:
                # Determine KIS order parameters
                symbol = order['symbol']
                quantity = order['quantity']
                price = order['price']
                order_type = "01" if price == 0 else "00" # 01: 시장가, 00: 지정가
                buy_sell_code = "02" if order['action'] == 'buy' else "01" # 02: 매수, 01: 매도
                
                # Execute the order using the broker
                kis_response_output = self.broker.order_cash(
                    symbol=symbol, 
                    quantity=quantity, 
                    price=price, 
                    order_type=order_type, 
                    buy_sell_code=buy_sell_code
                )
                # Assume success if no exception
                order_result = {"rt_cd": "0", "msg1": f"{action_desc} submitted successfully.", "output": kis_response_output, "order": order}
                logger.info(f"Order executed: {order_result['msg1']} - Response: {kis_response_output}")
                time.sleep(0.2) # Basic rate limiting between orders

            except KisBrokerError as e:
                logger.error(f"Failed to execute {action_desc}: {e}")
                # self.risk_guard.handle_api_error(e, action_desc) # Log or handle specific KIS errors
                order_result = {"rt_cd": e.response_data.get('rt_cd', '-1') if e.response_data else '-1', 
                                "msg1": f"{action_desc} failed: {e}", 
                                "output": e.response_data, 
                                "order": order}
            except Exception as e:
                logger.error(f"Unexpected error executing {action_desc}: {e}", exc_info=True)
                order_result = {"rt_cd": "-99", "msg1": f"Unexpected error during {action_desc}: {e}", "order": order}
            
            if order_result:
                results.append(order_result)
                
        logger.info(f"Finished executing orders. Results: {len(results)} recorded.")
        return results

    def _generate_briefing(self, order_results):
        logger.info("Generating Briefing...")
        # TODO: Implement briefing generation
        # return self.briefing.create_markdown_report(order_results)
        report = "## Daily Trading Report\n\n"
        success_count = sum(1 for r in order_results if r.get("rt_cd") == "0")
        fail_count = len(order_results) - success_count
        report += f"- Executed Orders: {success_count}\n- Failed Orders: {fail_count}\n\n"
        for result in order_results:
            if result.get("rt_cd") == "0":
                report += f"- Success: {result['msg1']} (Order No: {result['output']['ODNO']})\n"
            else:
                report += f"- Failed: {result['msg1']} - Order: {result.get('order', {})}\n"
        return report # Placeholder

    def _upsert_trade_results(self, order_results):
        logger.info("Upserting trade results to memory...")
        # TODO: Implement upserting trade results to Qdrant/DB
        # self.memory_rag.save_trade_results(order_results)
        pass

    def send_alert(self, message: str):
        """Send alert message (e.g., via Slack/Telegram)."""
        logger.warning(f"ALERT: {message}")
        # try:
        #     if self.notification_client:
        #         self.notification_client.send(f"🚨 Autotrade Alert 🚨\n{message}")
        # except Exception as e:
        #      logger.error(f"Failed to send alert: {e}")
        pass # Placeholder

    def send_notification(self, message: str, is_error: bool = False):
         """Send regular notification or briefing."""
         logger.info(f"NOTIFICATION (Error={is_error}):\n{message[:500]}...")
         # try:
         #     if self.notification_client:
         #         # Format differently for errors vs success briefings?
         #         title = "🚨 Autotrade Error" if is_error else "📊 Autotrade Daily Briefing"
         #         self.notification_client.send(f"{title}\n---\n{message}")
         # except Exception as e:
         #      logger.error(f"Failed to send notification: {e}")
         pass # Placeholder

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