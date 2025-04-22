# 주문 리스크 필터링 
import logging
from datetime import date, datetime, timedelta
import time
import pandas as pd
import numpy as np

from src.config import settings
from src.brokers.kis import KisBroker, KisBrokerError # Broker 및 에러 타입 임포트

logger = logging.getLogger(__name__)

class RiskGuard:
    def __init__(self, broker: KisBroker):
        """RiskGuard 초기화

        Args:
            broker: KisBroker 인스턴스 (잔고 조회, 시세 조회, 과거 데이터 조회 기능 필요)
        """
        self.broker = broker
        self.max_daily_orders = settings.MAX_DAILY_ORDERS
        self.stop_loss_percent = settings.STOP_LOSS_PERCENT
        self.use_atr_stop_loss = True # ATR 기반 손절 사용 여부 (True 권장) # 고정 비율 손절 (ATR과 병행 또는 택일 가능)
        self.use_atr_stop_loss = True # ATR 기반 손절 사용 여부 (True 권장)
        self.atr_stop_loss_period = 14 # ATR 계산 기간 (손절용)
        self.atr_stop_loss_multiplier = 2.0 # ATR 배수 (예: 2*ATR)
        self.investment_amount = settings.INVESTMENT_AMOUNT # 초기 투자금 (참고용)
        self.daily_order_count = 0
        self.last_checked_date = date.min
        # TODO: Track filled orders and capital usage for more accurate checks
        # self.used_capital = 0.0 # 실시간 잔고 조회로 대체
        self.max_api_retries = 3
        # ── 새 규칙 ─────────────────────────────
        self.max_position_weight = 0.25        # 포지션 1종 최대 25 %
        self.max_portfolio_var  = 0.10         # 1‑일 VaR 10 % 초과 금지
        self.slippage          = 0.001         # 체결 슬리피지 0.1 %
        self.api_backoff_seconds = 5
        logger.info(f"RiskGuard initialized. Max daily orders: {self.max_daily_orders}, Stop loss: {'ATR based' if self.use_atr_stop_loss else f'{self.stop_loss_percent*100}%'}")

    def _reset_daily_counter_if_needed(self):
        """날짜가 바뀌면 일일 주문 카운터 초기화"""
        today = date.today()
        if today != self.last_checked_date:
            logger.info(f"Date changed to {today}. Resetting daily order count.")
            self.daily_order_count = 0
            self.last_checked_date = today

    def validate_orders(self, orders: list) -> list:
        """LLM이 생성한 주문 목록을 검증합니다.

        Args:
            orders: Orchestrator가 생성한 주문 목록 (dict 리스트)
                    Example: [{'symbol': '069500', 'action': 'buy', 'quantity': 10, 'price': 0, 'reason': '...'}, ...]

        Returns:
            검증을 통과한 주문 목록
        """
        self._reset_daily_counter_if_needed() # Ensure daily counter is up-to-date
        logger.info(f"RiskGuard validating {len(orders)} potential orders... Daily count: {self.daily_order_count}")
        validated_orders = []
        available_cash = -1 # Sentinel value for fetch failure
        current_positions_map = {} # symbol -> quantity map

        if not isinstance(orders, list):
            logger.error(f"Invalid input type for orders: {type(orders)}. Expected list.")
            return []

        # --- Fetch current state (balance & positions) ---
        try:
            balance_info = self.broker.get_balance()
            available_cash = balance_info.get('dnca_tot_amt', 0) # d+2 예수금 사용 (실제 사용 가능 금액)
            logger.info(f"Available D+2 cash for validation: {available_cash:,.0f} KRW")

            positions_list = self.broker.get_positions()
            current_positions_map = {pos['pdno']: pos['hldg_qty'] for pos in positions_list if 'pdno' in pos and 'hldg_qty' in pos}
            logger.info(f"Current positions for validation: {current_positions_map}")

        except KisBrokerError as e:
            logger.error(f"CRITICAL: Failed to get balance/positions for RiskGuard validation: {e}. Cannot validate orders reliably.")
            # Halt validation if critical info is missing
            return []
        except Exception as e:
            logger.error(f"CRITICAL: Unexpected error getting balance/positions: {e}", exc_info=True)
            return []

        # --- Validate each order ---
        estimated_cash_needed = 0
        temp_positions = current_positions_map.copy() # Simulate position changes

        for order in orders:
            if not isinstance(order, dict):
                 logger.warning(f"Skipping invalid order format (not a dict): {order}")
                 continue

            symbol = order.get('symbol')
            action = order.get('action')
            quantity = order.get('quantity')
            reason = order.get('reason', 'N/A')

            is_valid = True
            validation_notes = []

            # --- Basic checks ---
            if not symbol or not isinstance(symbol, str):
                is_valid = False; validation_notes.append("Missing/invalid symbol")
            if action not in ['buy', 'sell']:
                is_valid = False; validation_notes.append(f"Invalid action: {action}")
            if not isinstance(quantity, int) or quantity <= 0:
                is_valid = False; validation_notes.append(f"Invalid quantity: {quantity}")

            # --- Risk Limit Checks ---
            # ① 포트폴리오 비중 제한
            current_eval = temp_positions.get(symbol, 0) * current_price
            total_eval   = sum(q*current_price for q in temp_positions.values()) + \
                           (available_cash - estimated_cash_needed)
            if action == 'buy' and total_eval > 0:
                if (current_eval/total_eval) > self.max_position_weight:
                    is_valid = False
                    validation_notes.append(f"Max 1‑symbol weight {self.max_position_weight*100:.0f}% exceeded")

            # ② VaR 한도 초과 확인 (단순 σ·z 방식)
            if action in ['buy', 'sell']:
                projected_var = self._compute_portfolio_var(temp_positions)
                if projected_var > self.max_portfolio_var:
                    is_valid = False
                    validation_notes.append(f"Projected VaR {projected_var:.2%} > limit")
            if self.daily_order_count + len(validated_orders) >= self.max_daily_orders:
                # Check against counter + already validated orders in this batch
                is_valid = False; validation_notes.append(f"Max daily order limit ({self.max_daily_orders}) reached")

            # --- Action-specific checks ---
            if action == 'buy':
                # Estimate cost (Market order - use current price + buffer)
                try:
                    quote = self.broker.get_quote(symbol)
                    current_price = float(quote.get('stck_prpr', 0))
                    if current_price <= 0: raise ValueError("Invalid current price")
                    estimated_cost = quantity * current_price * 1.01 # Add 1% buffer for market order slippage
                    time.sleep(0.05) # Slight delay after quote
                except (KisBrokerError, ValueError, Exception) as e:
                     logger.warning(f"Could not estimate cost for BUY {symbol}: {e}. Order rejected.")
                     is_valid = False; validation_notes.append(f"Could not estimate buy cost ({e})")
                     estimated_cost = float('inf') # Ensure it fails cash check

                # Check against available cash (considering already validated orders)
                if (available_cash - estimated_cash_needed) < estimated_cost:
                    is_valid = False
                    validation_notes.append(f"Insufficient cash (Need: {estimated_cost:,.0f}, Avail: {available_cash - estimated_cash_needed:,.0f})")
                else:
                    estimated_cash_needed += estimated_cost # Reserve cash for this order

                # Simulate position change
                temp_positions[symbol] = temp_positions.get(symbol, 0) + quantity

            elif action == 'sell':
                # Check if we hold enough quantity (considering previous sells in this batch)
                current_holding = temp_positions.get(symbol, 0)
                if current_holding < quantity:
                    is_valid = False
                    validation_notes.append(f"Insufficient holding (Have: {current_holding}, Need: {quantity})")
                else:
                    # Simulate position change
                    temp_positions[symbol] = current_holding - quantity

            # TODO: Add more checks (max position size, diversification etc.)

            # --- Final Decision ---
            if is_valid:
                logger.info(f"Order for {symbol} ({action} {quantity}) PASSED validation.")
                validated_orders.append(order)
                # Do not increment self.daily_order_count here, do it after the loop
            else:
                logger.warning(f"Order for {symbol} ({action} {quantity}) FAILED validation: {', '.join(validation_notes)}. Reason: {reason}")
                # Rollback simulated cash/position changes if needed (or recalculate totals)
                # Simple approach: just log failure, don't add to validated_orders. Cash/position simulation resets on next run.

        # Update the daily counter after processing the batch
        self.daily_order_count += len(validated_orders)

        logger.info(f"RiskGuard validation complete. {len(validated_orders)} orders passed. Daily total: {self.daily_order_count}")
        return validated_orders
    
    # --- 손절 로직 (실행은 Orchestrator 담당) --- 

    # ------------------------------------------------------------------
    # 🔽 새 VaR 계산
    # ------------------------------------------------------------------
    def _compute_portfolio_var(self, positions_map: dict[str, int]) -> float:
        """포지션 map을 받아 단순 ‘σ·1.65’ 1‑일 VaR(10 % 신뢰) 백분율 반환"""
        symbols = list(positions_map.keys())
        if not symbols: return 0.0
        prices, weights, returns = [], [], []
        # 수집
        for sym, qty in positions_map.items():
            try:
                df = self.broker.get_historical_data(sym, period=60)
                daily_ret = df['close'].pct_change().dropna()
                if daily_ret.empty: continue
                returns.append(daily_ret)
                price = float(df['close'].iloc[-1])
                prices.append(price)
            except Exception:
                continue
        if not returns: return 0.0
        import pandas as pd, numpy as np
        ret_df = pd.concat(returns, axis=1).fillna(0)
        cov = ret_df.cov()
        port_var = float(np.sqrt(np.dot(weights := np.array([1/len(symbols)]*len(symbols)),
                                        np.dot(cov * 252, weights.T))))
        return 1.65 * port_var  # 90 % VaR
    def check_stop_loss(self, current_positions: list[dict]) -> list[dict]:
        """현재 보유 포지션에 대한 손절 조건 확인 (ATR 또는 고정 비율)
        
        Args:
            current_positions: 현재 보유 종목 리스트 (from broker.get_positions())
                                예: [{'symbol': '069500', 'quantity': 10, 'average_buy_price': 31000, ...}]
        Returns:
            손절매가 필요한 주문(시장가 매도) 리스트
        """
        logger.info("Checking stop-loss conditions for current positions...")
        stop_loss_signals = []
        
        for position in current_positions:
            # pdno를 symbol로 사용
            symbol = position.get('pdno') # KIS position key
            quantity_str = position.get('hldg_qty', '0') # KIS position key
            avg_price_str = position.get('pchs_avg_pric', '0') # KIS position key
            current_price_str = position.get('prpr', '0') # KIS position key

            try:
                quantity = int(quantity_str)
                avg_price = float(avg_price_str)
                current_price = float(current_price_str)
            except (ValueError, TypeError) as e:
                 logger.warning(f"Invalid position data types for stop-loss: {position}. Error: {e}. Skipping.")
                 continue

            if not symbol or quantity <= 0 or avg_price <= 0:
                logger.debug(f"Skipping stop-loss check for invalid position data: {position}")
                continue

            try:
                if current_price <= 0:
                    logger.warning(f"Could not get valid current price ({current_price}) for stop-loss check: {symbol}. Trying get_quote...")
                    # Try fetching quote as fallback
                    quote = self.broker.get_quote(symbol)
                    current_price = float(quote.get('stck_prpr', 0))
                    if current_price <= 0:
                         logger.warning(f"Fallback get_quote also failed for {symbol}. Skipping stop-loss.")
                         continue
                    time.sleep(0.05) # Delay after quote

                stop_loss_price = 0
                reason = ""

                # 2. 손절 라인 계산 (ATR 또는 고정 비율)
                if self.use_atr_stop_loss:
                    atr = self._calculate_atr_for_stop_loss(symbol)
                    if pd.isna(atr) or atr <= 0:
                         logger.warning(f"Could not calculate ATR for stop-loss ({symbol}). Falling back to fixed percentage.")
                         stop_loss_price = avg_price * (1 - self.stop_loss_percent)
                         reason = f"Fallback Stop-Loss ({self.stop_loss_percent*100:.1f}%) triggered at {current_price:.0f} (Buy Avg: {avg_price:.0f}, Stop Level: {stop_loss_price:.0f})"
                    else:
                         stop_loss_price = avg_price - (self.atr_stop_loss_multiplier * atr)
                         reason = f"ATR Stop-Loss ({self.atr_stop_loss_multiplier:.1f}*ATR={atr:.0f}) triggered at {current_price:.0f} (Buy Avg: {avg_price:.0f}, Stop Level: {stop_loss_price:.0f})"
                else:
                    stop_loss_price = avg_price * (1 - self.stop_loss_percent)
                    reason = f"Fixed Stop-Loss ({self.stop_loss_percent*100:.1f}%) triggered at {current_price:.0f} (Buy Avg: {avg_price:.0f}, Stop Level: {stop_loss_price:.0f})"
                
                # 3. 손절 조건 확인 (현재가 < 손절 라인)
                if current_price < stop_loss_price:
                    logger.warning(f"Stop-Loss Triggered for {symbol}! Price: {current_price:.0f}, Stop Level: {stop_loss_price:.0f}, Avg Price: {avg_price:.0f}")
                    stop_loss_signals.append({
                        "symbol": symbol,
                        "action": "sell",
                        "quantity": quantity, # 보유 수량 전체 매도
                        "price": 0, # 시장가 손절
                        "reason": reason
                    })
                # Optional: Add delay even if ATR wasn't calculated here, if get_quote was used
                # time.sleep(0.05)
            except KisBrokerError as e:
                 logger.error(f"KisBrokerError checking stop-loss for {symbol}: {e}")
                 # 개별 종목 오류 시 계속 진행
            except Exception as e:
                logger.error(f"Error checking stop-loss for {symbol}: {e}", exc_info=True)
            
        if stop_loss_signals:
            logger.info(f"Generated {len(stop_loss_signals)} stop-loss signals: {[s['symbol'] for s in stop_loss_signals]}")
        return stop_loss_signals

    def _calculate_atr_for_stop_loss(self, symbol: str) -> float:
        """손절 계산용 ATR 계산 (Strategy 로직과 유사/독립)"""
        try:
            now = datetime.now()
            # KIS API는 보통 최근 100개 데이터만 제공. 기간 늘려도 소용없을 수 있음.
            # start_date = (now - timedelta(days=self.atr_stop_loss_period + 30)).strftime('%Y%m%d')
            # df_daily = self.broker.get_historical_data(symbol, timeframe='D',
            #                                            start_date=start_date,
            #                                            end_date=now.strftime('%Y%m%d'))
            df_daily = self.broker.get_historical_data(symbol, timeframe='D') # Default to recent data

            if df_daily.empty or len(df_daily) < self.atr_stop_loss_period or not all(col in df_daily.columns for col in ['high', 'low', 'close']):
                logger.warning(f"Insufficient daily data ({len(df_daily)} rows) to calculate ATR({self.atr_stop_loss_period}) for {symbol}")
                return np.nan

            df = df_daily.copy()
            # Ensure numeric types
            for col in ['high', 'low', 'close']:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['high', 'low', 'close'])

            if len(df) < self.atr_stop_loss_period:
                 logger.warning(f"Insufficient valid numeric data ({len(df)} rows) after cleaning for ATR({self.atr_stop_loss_period}) on {symbol}")
                 return np.nan

            df = df.iloc[-self.atr_stop_loss_period-5:] # Use recent data slice for calculation robustness

            df['high_low'] = df['high'] - df['low']
            df['high_close'] = np.abs(df['high'] - df['close'].shift())
            df['low_close'] = np.abs(df['low'] - df['close'].shift())
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

            # Use Simple Moving Average for ATR calculation
            atr = df['tr'].rolling(window=self.atr_stop_loss_period, min_periods=self.atr_stop_loss_period).mean().iloc[-1]

            # Alternate: Use Exponential Moving Average (EMA) for ATR - more responsive
            # atr = df['tr'].ewm(span=self.atr_stop_loss_period, adjust=False).mean().iloc[-1]

            if pd.isna(atr):
                 logger.warning(f"ATR calculation resulted in NaN for {symbol}")
                 return np.nan

            logger.debug(f"Calculated ATR({self.atr_stop_loss_period}) for {symbol}: {atr:.2f}")
            return atr
        except KisBrokerError as e:
            # Historical data 조회 실패 시 ATR 계산 불가
             logger.error(f"KisBrokerError calculating ATR for stop-loss ({symbol}): {e}")
             return np.nan
        except Exception as e:
            logger.error(f"Failed to calculate ATR for stop-loss ({symbol}): {e}", exc_info=True)
            return np.nan

    def handle_api_error(self, error: Exception, context: str) -> dict | None:
        """API 오류 처리 (로깅) 및 오류 정보 반환

        Args:
            error: 발생한 예외 객체
            context: 오류 발생 상황 설명 (e.g., "executing order for 069500")

        Returns:
            오류 정보를 담은 dict 또는 None
        """
        log_level = logging.ERROR
        error_info = {
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context
        }

        if isinstance(error, KisBrokerError):
            # KIS API 특정 오류 코드 포함
            rt_cd = error.response_data.get('rt_cd') if error.response_data else 'N/A'
            msg1 = error.response_data.get('msg1') if error.response_data else 'N/A'
            error_info["kis_rt_cd"] = rt_cd
            error_info["kis_msg1"] = msg1
            message = f"KIS API Error encountered during '{context}': {error} (rt_cd: {rt_cd}, msg1: {msg1})"
            # 특정 코드(예: 일시적 오류)는 Warning 수준으로 로깅할 수도 있음
            # if rt_cd == 'SOME_TEMP_ERROR': log_level = logging.WARNING
        else:
            # 일반적인 오류
            message = f"API Error encountered during '{context}': {error}"

        logger.log(log_level, message, exc_info=True if log_level >= logging.ERROR else False)

        # 오류 정보를 반환하여 상위 호출자(Orchestrator)가 알림 등을 처리하도록 함
        return error_info

# Example Usage 업데이트
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Mock Broker (Updated for new methods) --- 
    class MockBrokerForRisk:
        def get_quote(self, symbol):
             print(f"[MockBroker] Requesting quote for {symbol}")
             # Simulate varying prices
             price = 10000 + abs(hash(symbol)) % 20000
             return {'stck_prpr': str(price)}
             
        def get_balance(self):
             print("[MockBroker] Requesting balance...")
             # Simulate balance slightly changing
             cash = 800000 # Fixed for easier validation testing
             return {'dnca_tot_amt': cash, 'tot_evlu_amt': cash + 300000} # KIS keys

        def get_positions(self):
             print("[MockBroker] Requesting positions...")
             # Return example positions for stop loss check using KIS keys
             return [
                 {'pdno': '069500', 'prdt_name': 'KODEX 200', 'hldg_qty': '10', 'pchs_avg_pric': '36000', 'prpr': '37000'}, # Profit
                 {'pdno': '229200', 'prdt_name': 'KODEX 코스닥150', 'hldg_qty': '5', 'pchs_avg_pric': '12000', 'prpr': '11500'}, # Small loss
                 {'pdno': '114800', 'prdt_name': 'KODEX 인버스', 'hldg_qty': '20', 'pchs_avg_pric': '15000', 'prpr': '13000'},  # Big loss - likely stop loss
                 {'pdno': 'Invalid', 'prdt_name': 'Invalid Data', 'hldg_qty': 'abc', 'pchs_avg_pric': 'xyz', 'prpr': '0'}, # Invalid data test
             ]
             
        def get_historical_data(self, symbol, timeframe='D', start_date=None, end_date=None, period=None):
             print(f"[MockBroker] Requesting {timeframe} data for {symbol} (StopLoss ATR calc)")
             if symbol == 'Invalid': return pd.DataFrame() # Test empty DF
             np.random.seed(abs(hash(symbol)) % (2**32 - 1))
             num_periods = 30
             dates = pd.date_range(end=datetime.now(), periods=num_periods, freq='B')
             price_base = 10000 + abs(hash(symbol)) % 5000
             close = price_base + np.cumsum(np.random.randn(num_periods) * 50)
             high = close + np.random.rand(num_periods) * 100
             low = close - np.random.rand(num_periods) * 100
             high = np.maximum(high, close)
             low = np.minimum(low, close)
             # Simulate KIS returning strings
             return pd.DataFrame({'high': high.astype(str), 'low': low.astype(str), 'close': close.astype(str)}, index=dates)

        def order_cash(self, *args, **kwargs): # Dummy method
            print(f"[MockBroker] Received order_cash call: {kwargs}")
            return {"rt_cd": "0", "msg1": "Order submitted", "ODNO": "mock123"}

        def close(self): pass # Dummy method

    mock_broker = MockBrokerForRisk()
    risk_guard = RiskGuard(broker=mock_broker)

    # --- Test Order Validation (with Real Balance Check) --- 
    print("\n--- Testing Order Validation (with Balance Check) --- ")
    proposed_orders_llm = [
        {'symbol': '069500', 'action': 'buy', 'quantity': 10, 'price': 0, 'reason': 'LLM sees upside'}, # Approx 370k * 1.01 = 374k
        {'symbol': '114800', 'action': 'sell', 'quantity': 5, 'price': 0, 'reason': 'LLM reduces inverse'}, # Sell 보유량 20 > 5 OK
        {'symbol': '229200', 'action': 'buy', 'quantity': 20, 'price': 0, 'reason': 'LLM likes KOSDAQ'}, # Approx 11.5k * 20 * 1.01 = 232k (Total buy: 374+232=606k OK)
        {'symbol': '005930', 'action': 'buy', 'quantity': 5, 'price': 0, 'reason': 'LLM wants Samsung'}, # Approx 80k * 5 * 1.01 = 404k (Total buy: 606+404=1010k > 800k FAIL)
        {'symbol': '373170', 'action': 'sell', 'quantity': 100, 'price': 0, 'reason': 'LLM exits LGES'}, # Sell 보유량 0 < 100 FAIL
        {'symbol': '114800', 'action': 'sell', 'quantity': 16, 'price': 0, 'reason': 'LLM exits more Inverse'}, # Sell 보유량 20-5=15 < 16 FAIL
    ]
    validated = risk_guard.validate_orders(proposed_orders_llm)
    print(f"\nValidated orders: {validated}")
    # Expected: 069500 buy, 114800 sell(5), 229200 buy passed. Rest failed.

    # --- Test Stop Loss Check --- 
    print("\n--- Testing Stop-Loss Check --- ")
    current_positions = mock_broker.get_positions()

    stop_loss_orders = risk_guard.check_stop_loss(current_positions)
    print(f"\nStop-Loss Signals: {stop_loss_orders}")
    # Expected: 114800 likely triggers stop loss sell signal

    # Test with Fixed Percentage Stop Loss
    print("\n--- Testing Fixed Percentage Stop-Loss Check --- ")
    risk_guard.use_atr_stop_loss = False
    stop_loss_orders_fixed = risk_guard.check_stop_loss(current_positions)
    print(f"Fixed Stop-Loss Signals: {stop_loss_orders_fixed}")

    # --- Test API Error Handling ---
    print("\n--- Testing API Error Handling --- ")
    try:
        # Simulate an error from broker
        raise ValueError("Simulated connection error") # Generic error
    except Exception as e:
        error_report = risk_guard.handle_api_error(e, "simulated broker action")
        print(f"Generated error report: {error_report}")

    try:
        # Simulate a KIS specific error
        raise KisBrokerError("Rate limit exceeded", response_data={'rt_cd': 'APBK08040', 'msg_cd': 'APBK08040', 'msg1': '초당 처리건수를 초과하였습니다.'})
    except Exception as e:
        error_report_kis = risk_guard.handle_api_error(e, "simulated KIS rate limit")
        print(f"Generated KIS error report: {error_report_kis}")