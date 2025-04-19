# Dag 관리·에이전트 호출 흐름

import logging
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import openai # Import OpenAI
import json # For parsing LLM response
import uuid
from datetime import datetime

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
import asyncio # For potential async operations
# from src.agents.kis_developer import KisDeveloper # Removed
# from src.utils.logger import setup_logger # Remove custom logger import

# NEW: function registry
from src.utils.registry import command, COMMANDS

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
    def __init__(self, broker: KisBroker, db_session_factory, qdrant_client: QdrantClient):
        """Orchestrator 초기화

        Args:
            broker: KIS Broker 인스턴스
            db_session_factory: SQLAlchemy 세션 팩토리
            qdrant_client: Qdrant 클라이언트 인스턴스
        """
        self.broker = broker
        self.db_session_factory = db_session_factory
        self.qdrant_client = qdrant_client
        self.llm_model_name = None # Store model name

        # Initialize Finnhub Client
        self.finnhub = FinnhubClient(api_key=settings.FINNHUB_API_KEY)

        # Initialize OpenAI API Key
        if settings.OPENAI_API_KEY:
            # Setting openai.api_key globally, consider client instance if needed
            if not getattr(openai, 'api_key', None):
                openai.api_key = settings.OPENAI_API_KEY
            self.llm_model_name = settings.LLM_MAIN_TIER_MODEL  # Store model name
            logger.info(f"Orchestrator will use OpenAI model: {self.llm_model_name}")
        else:
            logger.warning("OPENAI_API_KEY not set. Orchestrator LLM functionality will be disabled.")

        # Initialize Agents
        # self.kis = KisDeveloper(account_info=settings.KIS_ACCOUNT) # Remove KisDeveloper instantiation
        self.info_crawler = InfoCrawler()
        self.memory_rag = MemoryRAG(db_session_factory=self.db_session_factory) # Pass factory
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
        """일일 자동매매 사이클 실행 (LLM 중심)
        """
        logger.info("Starting daily cycle with LLM orchestration...")
        if not self.llm_model_name: # Check if model name is set
            logger.error("LLM model name not set. Cannot run daily cycle.")
            return

        try:
            # 1. 정보 수집 및 준비 (InfoCrawler, Market Data, Memory)
            market_summary = self._run_info_crawler()
            market_data = self._fetch_market_data()
            current_positions = self._get_current_positions()
            # Retrieve relevant past memories/context using RAG
            rag_context = self._retrieve_relevant_memory(market_summary)

            # 2. LLM 추론 및 행동 계획 수립
            action_plan_str = self._get_llm_action_plan(market_summary, market_data, current_positions, rag_context)

            if not action_plan_str:
                logger.warning("LLM did not provide an action plan. Ending cycle.")
                return

            # 3. 행동 계획 파싱 및 실행
            try:
                # Expecting LLM to return a JSON list of actions
                action_plan = json.loads(action_plan_str)
                logger.info(f"LLM Action Plan: {action_plan}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM action plan JSON: {e}. Plan received: {action_plan_str}")
                # Maybe try a simpler parsing or ask LLM to reformat
                # For now, end cycle on parse failure
                return

            # --- 실행 단계 ---
            # 분리된 함수로 만들어 재사용성 높임
            execution_results = self._execute_action_plan(action_plan)

            # 4. 결과 보고 및 저장
            briefing_content = self._generate_briefing(execution_results)
            self._upsert_trade_results(execution_results) # Save results to memory/DB

            logger.info("Daily cycle completed successfully with LLM orchestration.")
            self.send_notification(briefing_content) # Use new notification method

        except Exception as e:
            logger.error(f"Daily cycle failed: {e}", exc_info=True)
            self.send_notification(f"Daily cycle failed: {e}", is_error=True)

    def _get_llm_action_plan(self, market_summary, market_data, current_positions, rag_context) -> str | None:
        """LLM에게 현재 상황 정보를 제공하고 행동 계획(JSON)을 요청합니다.
        """
        logger.info("Asking LLM for the daily action plan...")
        if not self.llm_model_name:
            return None

        # Build the prompt for the LLM
        prompt = f"""
        You are the master AI orchestrator for an ETF autotrading system. Your goal is to maximize returns while managing risk for a small investment amount ({settings.INVESTMENT_AMOUNT:,.0f} KRW).
        Today's Date: {datetime.now().strftime('%Y-%m-%d')}

        Current Market Situation:
        - Summary: {market_summary}
        - Key ETF Prices: {json.dumps({k: v for k, v in market_data.items() if v}, indent=2)}

        Current Portfolio:
        {json.dumps(current_positions, indent=2)}

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
            logger.info(f"Requesting OpenAI action plan using {self.llm_model_name}...")
            messages = [
                {"role": "system", "content": "You are an AI assistant that generates JSON action plans for an ETF autotrading system based on provided market context and portfolio data."}, # System prompt
                {"role": "user", "content": prompt}
            ]
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY) # Create client
            resp = client.chat.completions.create(
                model=self.llm_model_name, # Use stored model name
                messages=messages,
                temperature=0.7,
                max_tokens=800, # Adjust as needed
                # Consider response_format for JSON mode if using compatible models
                # response_format={"type": "json_object"} 
            )
            text = resp.choices[0].message.content.strip()
            
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
                    logger.info(f"Received JSON action plan from OpenAI: {json_block}")
                    return json_block
                except json.JSONDecodeError:
                     logger.error(f"Extracted block is not valid JSON: {json_block}")
                     return None # Failed validation
            else:
                logger.error(f"Could not extract JSON action plan from LLM response: {text}")
                return None

        except openai.APIError as e:
             logger.error(f"OpenAI API Error during action plan generation: {e}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"OpenAI action plan generation failed: {e}", exc_info=True)
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
                 "detail": f"Error sending confirmation request: {e}",
                 "request_id": request_id,
                 "orders_requested": orders_to_confirm
             }

    def _run_info_crawler(self):
        logger.info("Running Info Crawler...")
        try:
            # Let InfoCrawler handle its own logic, potentially using LLM for summarization
            summary = self.info_crawler.get_market_summary()
            if summary:
                 # Optionally save summary to memory
                 self.memory_rag.save_memory(summary, metadata={"type": "market_summary", "source": "infocrawler"})
                 return summary
            else:
                 return "No market summary available."
        except Exception as e:
             logger.error(f"InfoCrawler failed: {e}", exc_info=True)
             return "Error fetching market summary."

    def _retrieve_relevant_memory(self, current_summary: str, limit: int = 5) -> str:
        """현재 상황과 관련된 과거 메모리를 RAG를 통해 검색합니다.
        """
        logger.info("Retrieving relevant memory context using RAG...")
        try:
            search_results = self.memory_rag.search_similar(current_summary, limit=limit)
            context = ""
            if search_results:
                context += "Found relevant past context:\n"
                for i, hit in enumerate(search_results):
                    # Ensure payload and text exist
                    payload_text = hit.payload.get('text', 'N/A') if hit.payload else 'N/A'
                    context += f"{i+1}. (Score: {hit.score:.3f}) {payload_text[:200]}...\n" # Limit length
            else:
                 context = "No relevant past context found."
            return context.strip()
        except Exception as e:
             logger.error(f"MemoryRAG search failed: {e}", exc_info=True)
             return "Error retrieving context from memory."

    @kis_retry_decorator
    def _get_current_positions(self) -> list:
        """현재 보유 포지션 정보를 가져옵니다.
        """
        logger.info("Fetching current positions...")
        try:
            positions = self.broker.get_positions()
            logger.info(f"Fetched {len(positions)} current positions.")
            return positions
        except KisBrokerError as e:
            logger.error(f"Failed to get current positions after retries: {e}")
            # Depending on policy, maybe return empty list or raise error
            return []
        except Exception as e:
             logger.error(f"Unexpected error getting positions: {e}", exc_info=True)
             return []

    def _summarize_and_upsert_memory(self):
        # This function might be less relevant now if LLM handles context directly
        # Or could be used to summarize *this* cycle's logs after completion
        logger.info("Skipping explicit memory summarization step in LLM-driven cycle.")
        pass

    @kis_retry_decorator
    def _fetch_market_data(self):
        logger.info("Fetching market data...")
        # TODO: Implement market data fetching (e.g., using broker)
        # return self.broker.get_quotes([...]) # 필요한 종목 리스트
        market_data = {}
        # TODO: Get target symbols from strategy or config
        symbols = settings.TARGET_SYMBOLS # <- 수정된 부분
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
        # This method remains largely the same, but is now called CONDITIONALLY
        # after user approval is received.
        logger.info(f"Executing {len(validated_orders)} APPROVED orders...") # Added 'APPROVED'
        results = []
        for order in validated_orders:
            order_result = None
            # Use reason from the order if available, else generate default
            action_desc = order.get('reason', f"{order['action']} order for {order['symbol']}")
            try:
                # Determine KIS order parameters
                symbol = order['symbol']
                quantity = order['quantity']
                price = order['price'] # Should be 0 for market order from LLM plan
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
                order_result = {"action_type": order['action'], "status": "success", "detail": f"{action_desc} submitted successfully.", "kis_response": kis_response_output, "order": order}
                logger.info(f"Order executed: {order_result['detail']} - Response: {kis_response_output}")
                # Consider adding configurable sleep from settings
                time.sleep(settings.ORDER_INTERVAL_SECONDS) # Use setting

            except KisBrokerError as e:
                logger.error(f"Failed to execute {action_desc}: {e}")
                # self.risk_guard.handle_api_error(e, action_desc) # Maybe RiskGuard handles this?
                order_result = {"action_type": order['action'], "status": "failed", "detail": f"{action_desc} failed: {e}", "kis_response": e.response_data, "order": order}
            except Exception as e:
                logger.error(f"Unexpected error executing {action_desc}: {e}", exc_info=True)
                order_result = {"action_type": order['action'], "status": "error", "detail": f"Unexpected error during {action_desc}: {e}", "order": order}

            if order_result:
                results.append(order_result)

        logger.info(f"Finished executing orders. Results: {len(results)} recorded.")
        return results

    def _generate_briefing(self, execution_results):
        logger.info("Generating Briefing based on execution results...")
        try:
            # Pass the structured results to the briefing agent
            return self.briefing_agent.create_report_from_actions(execution_results)
        except Exception as e:
             logger.error(f"Briefing generation failed: {e}", exc_info=True)
             return f"Error generating briefing: {e}"

    def _upsert_trade_results(self, execution_results):
        logger.info("Upserting execution results to memory...")
        try:
            # Let MemoryRAG handle saving the structured results
            self.memory_rag.save_execution_results(execution_results)
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