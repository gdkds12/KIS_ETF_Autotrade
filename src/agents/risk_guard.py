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
        self.stop_loss_percent = settings.STOP_LOSS_PERCENT # 고정 비율 손절 (ATR과 병행 또는 택일 가능)
        self.use_atr_stop_loss = True # ATR 기반 손절 사용 여부 (True 권장)
        self.atr_stop_loss_period = 14 # ATR 계산 기간 (손절용)
        self.atr_stop_loss_multiplier = 2.0 # ATR 배수 (예: 2*ATR)
        self.investment_amount = settings.INVESTMENT_AMOUNT # 초기 투자금 (참고용)
        self.daily_order_count = 0
        self.last_checked_date = date.min
        # TODO: Track filled orders and capital usage for more accurate checks
        # self.used_capital = 0.0 # 실시간 잔고 조회로 대체
        self.max_api_retries = 3
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
        """제안된 주문 리스트를 검증 (한도, 실제 잔고 등)

        Args:
            orders: 전략 에이전트가 생성한 주문 리스트

        Returns:
            검증을 통과한 주문 리스트
        """
        self._reset_daily_counter_if_needed()
        validated_orders = []
        available_capital = 0
        try:
            # --- 실제 계좌 잔고 조회 --- 
            balance_info = self.broker.get_balance()
            available_capital = balance_info.get('available_cash', 0)
            logger.info(f"Validating {len(orders)} proposed orders. Daily count: {self.daily_order_count}/{self.max_daily_orders}. Available cash: {available_capital:,.0f} KRW")
        except KisBrokerError as e:
             logger.error(f"Failed to get account balance for order validation: {e}. Rejecting all buy orders.")
             # 잔고 조회 실패 시 매수 주문은 모두 거부 (안전 조치)
             available_capital = -1 # 매수 불가 상태 표시
        except Exception as e:
             logger.error(f"Unexpected error getting balance: {e}", exc_info=True)
             available_capital = -1

        for order in orders:
            if self.daily_order_count >= self.max_daily_orders:
                logger.warning(f"Order rejected: Daily order limit ({self.max_daily_orders}) reached. Order: {order}")
                continue

            order_value = 0
            price = order['price'] # 지정가 또는 시장가(0)
            quantity = order['quantity']
            symbol = order['symbol']
            
            # 수량/가격 유효성 검사
            if quantity <= 0 or price < 0:
                 logger.warning(f"Order rejected: Invalid quantity ({quantity}) or price ({price}). Order: {order}")
                 continue
                 
            # 매수 주문 시 금액 계산 및 잔고 확인
            if order['action'] == 'buy':
                # 잔고 조회가 실패했으면 매수 불가
                if available_capital < 0:
                     logger.warning(f"Order rejected: Cannot validate order due to balance check failure. Order: {order}")
                     continue
                     
                current_price_estimate = price
                if price == 0: # 시장가 주문
                    try:
                        quote = self.broker.get_quote(symbol)
                        current_price_estimate = float(quote.get('stck_prpr', 0))
                        if current_price_estimate <= 0:
                             raise ValueError("Invalid current price received from quote")
                        # 시장가 주문 시 슬리피지 감안 (예: 5% 높게)
                        order_value = quantity * current_price_estimate * 1.05 
                        logger.info(f"Market order for {symbol}: Estimated value={order_value:.0f} (Price={current_price_estimate:.0f} + 5% slippage)")
                    except (KisBrokerError, ValueError, Exception) as e:
                        logger.error(f"Order rejected: Could not get valid current price for market order {symbol}: {e}. Order: {order}")
                        continue
                else: # 지정가 주문
                     order_value = quantity * price

                # 실제 가용 현금과 비교
                if order_value > available_capital:
                    logger.warning(f"Order rejected: Insufficient cash. Required: {order_value:,.0f}, Available: {available_capital:,.0f}. Order: {order}")
                    continue

            # 검증 통과 (매도 주문은 잔고 체크 X, 보유량 체크는 Broker API가 처리)
            logger.info(f"Order validated: {order}")
            validated_orders.append(order)
            self.daily_order_count += 1
            if order['action'] == 'buy':
                available_capital -= order_value # 예상 사용 가능 금액 차감

        logger.info(f"Validation complete. {len(validated_orders)} orders passed.")
        return validated_orders
    
    # --- 손절 로직 (실행은 Orchestrator 담당) --- 
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
            symbol = position.get('symbol')
            quantity = position.get('quantity', 0)
            avg_price = position.get('average_buy_price', 0)
            
            if not symbol or quantity <= 0 or avg_price <= 0:
                logger.debug(f"Skipping stop-loss check for invalid position data: {position}")
                continue
                
            try:
                # 1. 현재가 조회
                # quote = self.broker.get_quote(symbol)
                # current_price = float(quote.get('stck_prpr', 0))
                # get_positions()에서 받은 현재가 사용 (더 최신 정보 필요 시 get_quote 호출)
                current_price = position.get('current_price', 0)
                if current_price <= 0:
                    logger.warning(f"Could not get valid current price ({current_price}) for stop-loss check: {symbol}. Skipping.")
                    continue
                
                stop_loss_price = 0
                reason = ""

                # 2. 손절 라인 계산 (ATR 또는 고정 비율)
                if self.use_atr_stop_loss:
                    atr = self._calculate_atr_for_stop_loss(symbol)
                    if pd.isna(atr) or atr <= 0:
                         logger.warning(f"Could not calculate ATR for stop-loss ({symbol}). Falling back to fixed percentage.")
                         stop_loss_price = avg_price * (1 - self.stop_loss_percent)
                         reason = f"Fallback Stop-Loss ({self.stop_loss_percent*100:.1f}%) triggered at {current_price:.0f} (Stop Level: {stop_loss_price:.0f})"
                    else:
                         stop_loss_price = avg_price - (self.atr_stop_loss_multiplier * atr)
                         reason = f"ATR Stop-Loss ({self.atr_stop_loss_multiplier:.1f}*ATR={atr:.0f}) triggered at {current_price:.0f} (Stop Level: {stop_loss_price:.0f})"
                else:
                    stop_loss_price = avg_price * (1 - self.stop_loss_percent)
                    reason = f"Fixed Stop-Loss ({self.stop_loss_percent*100:.1f}%) triggered at {current_price:.0f} (Stop Level: {stop_loss_price:.0f})"
                
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
                time.sleep(0.1) # ATR 계산 등 내부 API 호출 시 대비
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
            start_date = (now - timedelta(days=self.atr_stop_loss_period + 30)).strftime('%Y%m%d')
            df_daily = self.broker.get_historical_data(symbol, timeframe='D', 
                                                       start_date=start_date,
                                                       end_date=now.strftime('%Y%m%d'))
                                                       
            if df_daily.empty or len(df_daily) < self.atr_stop_loss_period or not all(col in df_daily.columns for col in ['high', 'low', 'close']):
                logger.warning(f"Insufficient daily data to calculate ATR({self.atr_stop_loss_period}) for {symbol}")
                return np.nan
                
            df = df_daily.copy()
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = np.abs(df['high'] - df['close'].shift())
            df['low_close'] = np.abs(df['low'] - df['close'].shift())
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            atr = df['tr'].rolling(window=self.atr_stop_loss_period, min_periods=self.atr_stop_loss_period).mean().iloc[-1]
            logger.debug(f"Calculated ATR({self.atr_stop_loss_period}) for {symbol}: {atr:.2f}")
            return atr
        except KisBrokerError as e:
            # Historical data 조회 실패 시 ATR 계산 불가
             logger.error(f"KisBrokerError calculating ATR for stop-loss ({symbol}): {e}")
             return np.nan
        except Exception as e:
            logger.error(f"Failed to calculate ATR for stop-loss ({symbol}): {e}", exc_info=True)
            return np.nan

    def handle_api_error(self, error: Exception, action_description: str):
        """KIS API 오류 처리 (Rate Limit 등)
        
        Args:
            error: 발생한 예외 객체 (KisBrokerError 등)
            action_description: 오류가 발생한 작업 설명 (e.g., "placing order for 069500")
            
        Returns:
            재시도 여부 (True/False) - 현재는 로깅만 수행
        """
        logger.error(f"API Error occurred during {action_description}: {error}")
        
        # KisBrokerError이고 특정 에러 코드(e.g., Rate Limit)인 경우 백오프/재시도 로직 구현 가능
        # if isinstance(error, KisBrokerError) and error.response_data:
        #     error_code = error.response_data.get('msg_cd') # 예시 에러 코드
        #     if error_code == 'APBK08040': # 예: 초당 거래건수 초과
        #         logger.warning(f"Rate limit hit during {action_description}. Backing off for {self.api_backoff_seconds}s.")
        #         time.sleep(self.api_backoff_seconds)
        #         # TODO: 재시도 로직은 Orchestrator 등 상위 레벨에서 관리하는 것이 더 적합할 수 있음
        #         return True # 재시도 가능함을 알림
                
        # 그 외 심각한 오류는 알림 등 처리
        # TODO: Implement alerting (e.g., Slack, Telegram)
        
        return False # 기본적으로 재시도 안 함

# Example Usage 업데이트
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Mock Broker (Updated for new methods) --- 
    class MockBrokerForRisk:
        def get_quote(self, symbol):
             print(f"[MockBroker] Requesting quote for {symbol}")
             return {'stck_prpr': str(35000 + hash(symbol) % 1000)}
             
        def get_balance(self):
             print("[MockBroker] Requesting balance...")
             # Simulate balance slightly changing
             cash = 800000 - time.time() % 10000
             return {'available_cash': cash, 'total_asset_value': cash + 200000}

        def get_positions(self):
             print("[MockBroker] Requesting positions...")
             # Return example positions for stop loss check
             return [
                 {'symbol': '069500', 'name': 'KODEX 200', 'quantity': 10, 'average_buy_price': 36000, 'current_price': 37000}, # Profit
                 {'symbol': '229200', 'name': 'KODEX 코스닥150', 'quantity': 5, 'average_buy_price': 12000, 'current_price': 11500}, # Small loss
                 {'symbol': '114800', 'name': 'KODEX 인버스', 'quantity': 20, 'average_buy_price': 15000, 'current_price': 13000}  # Big loss - likely stop loss
             ]
             
        def get_historical_data(self, symbol, timeframe='D', start_date=None, end_date=None, period=None):
             print(f"[MockBroker] Requesting {timeframe} data for {symbol} (StopLoss ATR calc)")
             np.random.seed(abs(hash(symbol)) % (2**32 - 1))
             num_periods = 30
             dates = pd.date_range(end=datetime.now(), periods=num_periods, freq='B')
             price_base = 10000 + hash(symbol) % 5000
             close = price_base + np.cumsum(np.random.randn(num_periods) * 50)
             high = close + np.random.rand(num_periods) * 100
             low = close - np.random.rand(num_periods) * 100
             high = np.maximum(high, close)
             low = np.minimum(low, close)
             return pd.DataFrame({'high': high, 'low': low, 'close': close}, index=dates)

    mock_broker = MockBrokerForRisk()
    risk_guard = RiskGuard(broker=mock_broker)

    # --- Test Order Validation (with Real Balance Check) --- 
    print("\n--- Testing Order Validation (with Balance Check) --- ")
    proposed_orders = [
        {'symbol': '069500', 'action': 'buy', 'quantity': 10, 'price': 35000}, # 350k
        {'symbol': '114800', 'action': 'buy', 'quantity': 20, 'price': 15000}, # 300k (Total 650k - OK)
        {'symbol': '229200', 'action': 'buy', 'quantity': 10, 'price': 12000}, # 120k (Total 770k - OK)
        {'symbol': '005930', 'action': 'buy', 'quantity': 2, 'price': 80000},  # 160k (Total 930k - Fails as likely > 800k balance)
    ]
    validated = risk_guard.validate_orders(proposed_orders)
    print(f"Validated orders: {validated}")

    # --- Test Stop Loss Check --- 
    print("\n--- Testing Stop-Loss Check --- ")
    current_positions = mock_broker.get_positions()
    # Modify one position's current price to trigger stop loss (if needed)
    for pos in current_positions:
        if pos['symbol'] == '114800': # KODEX Inverse
             pos['current_price'] = 13000 # Force price below typical stop loss
             
    stop_loss_orders = risk_guard.check_stop_loss(current_positions)
    print(f"Stop-Loss Signals: {stop_loss_orders}")
    
    # Test with Fixed Percentage Stop Loss
    print("\n--- Testing Fixed Percentage Stop-Loss Check --- ")
    risk_guard.use_atr_stop_loss = False
    stop_loss_orders_fixed = risk_guard.check_stop_loss(current_positions)
    print(f"Fixed Stop-Loss Signals: {stop_loss_orders_fixed}")

    # --- Test API Error Handling (commented out for now) ---
    # print("\n--- Testing API Error Handling --- ")
    # try:
    #     # Simulate an error from broker
    #     raise ValueError("Simulated connection error") # Generic error
    # except Exception as e:
    #     risk_guard.handle_api_error(e, "simulated broker action")
    # 
    # try:
    #     # Simulate a KIS specific error (requires KisBrokerError defined)
    #     # from src.brokers.kis import KisBrokerError 
    #     # raise KisBrokerError("Rate limit exceeded", response_data={'rt_cd': '-1', 'msg_cd': 'APBK08040', 'msg1': '...'})
    #     pass 
    # except Exception as e:
    #     should_retry = risk_guard.handle_api_error(e, "simulated KIS rate limit")
    #     print(f"Should retry action? {should_retry}")