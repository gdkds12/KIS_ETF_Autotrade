# 모멘텀·리밸런스 전략 

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import KisBroker for type hinting and actual calls
from src.brokers.kis import KisBroker, KisBrokerError 
# from src.config import settings # Settings not directly needed here usually

logger = logging.getLogger(__name__)

class TradingStrategy:
    def __init__(self, broker: KisBroker, investment_amount: float, target_symbols: list):
        """TradingStrategy 초기화

        Args:
            broker: KisBroker 인스턴스 (과거 데이터 조회, 포지션 조회 기능 필요)
            investment_amount: 총 투자 금액
            target_symbols: 매매 대상 ETF 종목 코드 리스트
        """
        self.broker = broker
        self.investment_amount = investment_amount
        self.target_symbols = target_symbols
        self.roc_period = 12 # 12개월 모멘텀
        self.roc_short_period = 1 # 1개월 모멘텀
        self.atr_period = 20 # ATR 계산 기간 (일)
        self.risk_per_trade = 0.02 # 거래당 최대 손실률 (2%)
        self.momentum_threshold = 0 # 모멘텀 스코어 임계값 (0 이상일 때 투자 고려)
        self.rebalance_day = 1 # 매월 리밸런싱 실행일 (1일)
        logger.info(f"TradingStrategy initialized for symbols: {target_symbols}")

    def _calculate_roc(self, series: pd.Series, period: int) -> float:
        """주어진 기간의 Rate of Change (ROC) 계산"""
        if len(series) > period:
            # Ensure division is safe (non-zero denominator)
            if series.iloc[-period - 1] != 0:
                 return (series.iloc[-1] / series.iloc[-period - 1]) - 1
            else:
                 return np.inf # Avoid division by zero
        return 0.0

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """Average True Range (ATR) 계산"""
        if df.empty or len(df) < period or not all(col in df.columns for col in ['high', 'low', 'close']):
            logger.warning("Cannot calculate ATR: Insufficient data or missing columns.")
            return np.nan
        df = df.copy()
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        atr = df['tr'].rolling(window=period, min_periods=period).mean().iloc[-1]
        return atr

    def _get_momentum_score(self, symbol: str) -> tuple[float, float]:
        """종목의 12-1 ROC 모멘텀 스코어와 최신 종가 조회 (실제 데이터 사용)"""
        try:
            # 약 13~14개월치 월봉 데이터 요청 (ROC 계산용)
            # KIS API 제약 확인 필요 (한번에 가져올 수 있는 기간 등)
            now = datetime.now()
            start_date_monthly = (now - timedelta(days=31 * 14)).strftime('%Y%m%d')
            df_monthly = self.broker.get_historical_data(symbol, timeframe='M', 
                                                         start_date=start_date_monthly,
                                                         end_date=now.strftime('%Y%m%d'))
            # Placeholder 제거
            # dates_monthly = pd.date_range(end=datetime.now(), periods=14, freq='M')
            # prices_monthly = np.random.rand(14) * 1000 + 30000
            # df_monthly = pd.DataFrame({'close': prices_monthly}, index=dates_monthly)
            logger.debug(f"Monthly data for {symbol} ({len(df_monthly)} rows):\n{df_monthly.tail()}")

            if df_monthly.empty or 'close' not in df_monthly.columns or len(df_monthly) < self.roc_period + 1:
                logger.warning(f"Insufficient monthly data for {symbol} ({len(df_monthly)} rows) to calculate ROC.")
                return -np.inf, np.nan
            
            roc_12m = self._calculate_roc(df_monthly['close'], self.roc_period)
            roc_1m = self._calculate_roc(df_monthly['close'], self.roc_short_period)
            momentum_score = roc_12m - roc_1m
            latest_close = df_monthly['close'].iloc[-1]

            logger.info(f"Momentum score for {symbol}: {momentum_score:.4f} (12m: {roc_12m:.4f}, 1m: {roc_1m:.4f}) @ {latest_close:.0f}")
            return momentum_score, latest_close

        except KisBrokerError as e:
             logger.error(f"KisBrokerError calculating momentum for {symbol}: {e}")
             return -np.inf, np.nan
        except Exception as e:
            logger.error(f"Failed to calculate momentum for {symbol}: {e}", exc_info=True)
            return -np.inf, np.nan

    def _calculate_position_size(self, symbol: str, price: float) -> int:
        """ATR 기반 리스크 관리 포지션 크기 계산 (실제 데이터 사용)"""
        try:
            # ATR 계산용 일봉 데이터 요청 (atr_period + 여유분)
            now = datetime.now()
            start_date_daily = (now - timedelta(days=self.atr_period + 30)).strftime('%Y%m%d')
            df_daily = self.broker.get_historical_data(symbol, timeframe='D', 
                                                       start_date=start_date_daily,
                                                       end_date=now.strftime('%Y%m%d'))
            # Placeholder 제거
            # dates_daily = pd.date_range(end=datetime.now(), periods=self.atr_period + 5, freq='D')
            # high = np.random.rand(self.atr_period + 5) * 500 + price
            # low = high - np.random.rand(self.atr_period + 5) * 500
            # close = (high + low) / 2
            # df_daily = pd.DataFrame({'high': high, 'low': low, 'close': close}, index=dates_daily)
            logger.debug(f"Daily data for {symbol} ({len(df_daily)} rows) for ATR:\n{df_daily.tail()}")

            atr = self._calculate_atr(df_daily, self.atr_period)
            if pd.isna(atr) or atr <= 0:
                logger.warning(f"Could not calculate valid ATR ({atr}) for {symbol}. Using fallback risk calc.")
                # ATR 계산 불가 시 가격 변동성의 일정 비율 사용 (예: 5%)
                risk_per_share = price * 0.05
            else:
                 # 일반적인 ATR 기반 손절 라인: 가격 - 2 * ATR
                stop_loss_distance = 2 * atr # 주당 예상 손실폭
                risk_per_share = stop_loss_distance
            
            if risk_per_share <= 0:
                logger.warning(f"Calculated risk per share is not positive ({risk_per_share:.2f}) for {symbol}. Cannot size position.")
                return 0

            # 총 투자금 대비 거래당 리스크 금액
            capital_at_risk = self.investment_amount * self.risk_per_trade
            
            # 포지션 크기 = (투자금 * 거래당 리스크 비율) / (주당 리스크 금액)
            position_size = int(capital_at_risk // risk_per_share)
            logger.info(f"Position size for {symbol}: {position_size} shares (Price={price:.0f}, ATR={atr:.2f}, Risk/Share={risk_per_share:.2f}, Capital@Risk={capital_at_risk:.0f})")
            
            return max(0, position_size)
        except KisBrokerError as e:
             logger.error(f"KisBrokerError calculating position size for {symbol}: {e}")
             return 0
        except Exception as e:
             logger.error(f"Failed to calculate position size for {symbol}: {e}", exc_info=True)
             return 0

    def generate_signals(self) -> list:
        """매매 신호 생성 (모멘텀 스코어 기반 종목 선정 및 포지션 크기 결정)"""
        logger.info("Generating trading signals based on momentum and ATR...")
        momentum_scores = {}
        prices = {}

        # 1. Calculate momentum score and get latest price for all target symbols
        for symbol in self.target_symbols:
            score, price = self._get_momentum_score(symbol)
            if not pd.isna(price): # 가격 정보가 유효한 경우에만 포함
                momentum_scores[symbol] = score
                prices[symbol] = price
            else:
                 logger.warning(f"Could not get price for {symbol}, excluding from strategy calculation.")
            time.sleep(0.1) # KIS API 호출 간격

        # 2. Select symbols passing momentum threshold
        eligible_symbols = {s: score for s, score in momentum_scores.items() if score > self.momentum_threshold}
        
        if not eligible_symbols:
            logger.info(f"No symbols passed the momentum threshold ({self.momentum_threshold}). No buy signals generated.")
            return []

        # 모멘텀 스코어 순으로 정렬 (높은 순)
        sorted_symbols = sorted(eligible_symbols.items(), key=lambda item: item[1], reverse=True)
        logger.info(f"Symbols passing momentum threshold: {[(s, round(sc, 4)) for s, sc in sorted_symbols]}")

        # 3. Calculate position size for selected symbols
        target_positions = {}
        # --- 균등 배분 대신 리스크 기반 총합 계산 --- 
        # num_selected = len(sorted_symbols)
        # investment_per_symbol = self.investment_amount / max(1, num_selected)
        total_allocated_capital = 0

        for symbol, score in sorted_symbols:
            price = prices[symbol]
            quantity = self._calculate_position_size(symbol, price)
            time.sleep(0.1) # KIS API 호출 간격
            
            if quantity > 0:
                required_capital = quantity * price
                # Check if adding this position exceeds total investment amount (considering risk factor indirectly)
                # This check might be better placed in RiskGuard or Orchestrator with real-time balance
                if total_allocated_capital + required_capital > self.investment_amount * 1.05: # 약간의 여유 허용
                     logger.warning(f"Skipping signal for {symbol} ({quantity} shares @ {price:.0f}): Estimated capital allocation exceeds total investment.")
                     continue
                
                target_positions[symbol] = {
                    "action": "buy",
                    "quantity": quantity,
                    "price": 0, # 시장가 주문 사용
                    "reason": f"Momentum Score: {score:.4f} > {self.momentum_threshold}"
                }
                total_allocated_capital += required_capital
            else:
                logger.info(f"Calculated position size is 0 for {symbol}. No signal generated.")

        # 4. Generate buy signals list
        signals = [
            {**pos_info, "symbol": symbol} for symbol, pos_info in target_positions.items()
        ]
        
        logger.info(f"Generated {len(signals)} buy signals based on current calculation: {signals}")
        return signals

    def check_rebalance_timing(self) -> bool:
        """월간 리밸런싱 시점인지 확인"""
        today = datetime.now().day
        # TODO: Add logic for handling weekends/holidays (e.g., run on the next trading day)
        is_rebalance_day = (today == self.rebalance_day)
        if is_rebalance_day:
            logger.info(f"Today (Day {today}) is the scheduled rebalancing day ({self.rebalance_day}).")
        return is_rebalance_day

    def generate_rebalance_signals(self) -> list:
        """월간 리밸런싱 실행: 목표 포트폴리오와 현재 포트폴리오 비교하여 매매 신호 생성
           (내부적으로 get_positions 호출)
        Returns:
            리밸런싱에 필요한 매수/매도 신호 리스트
        """
        logger.info("Generating rebalancing signals...")
        if not self.check_rebalance_timing():
            logger.info("Not rebalancing day. No rebalance signals generated.")
            return []

        # 1. Get current positions from broker
        try:
            current_positions_list = self.broker.get_positions()
            # Convert list to dict for easier lookup
            current_positions = {pos['symbol']: pos for pos in current_positions_list if pos.get('quantity', 0) > 0}
            logger.info(f"Current positions for rebalancing: {list(current_positions.keys())}")
        except KisBrokerError as e:
            logger.error(f"Failed to get current positions for rebalancing: {e}. Aborting rebalance.")
            return [] # 포지션 조회 실패 시 리밸런싱 중단
        except Exception as e:
             logger.error(f"Unexpected error getting positions: {e}", exc_info=True)
             return []

        # 2. Generate target portfolio based on current momentum
        target_signals = self.generate_signals() # 현재 시점의 목표 포지션 계산
        target_portfolio = {s['symbol']: s for s in target_signals if s['action'] == 'buy'}
        logger.info(f"Target portfolio for rebalancing: {list(target_portfolio.keys())}")

        rebalance_signals = []

        # 3. Generate sell signals for positions no longer in target portfolio
        for symbol, position in current_positions.items():
            if symbol not in target_portfolio:
                current_quantity = position.get('quantity', 0)
                if current_quantity > 0:
                    logger.info(f"Rebalance Sell (Exit): {symbol} (Quantity: {current_quantity}) - No longer in target portfolio.")
                    rebalance_signals.append({
                        "symbol": symbol,
                        "action": "sell",
                        "quantity": current_quantity, # 보유 수량 전체 매도
                        "price": 0, # 시장가 매도
                        "reason": "Rebalancing: Exit position"
                    })

        # 4. Generate buy/sell signals for positions with changed quantity or new entries
        for symbol, target_pos in target_portfolio.items():
            target_quantity = target_pos['quantity']
            current_quantity = current_positions.get(symbol, {}).get('quantity', 0)
            quantity_diff = target_quantity - current_quantity

            if quantity_diff > 0: # Need to buy more or new entry
                 action_reason = "New entry" if current_quantity == 0 else "Increase position"
                 logger.info(f"Rebalance Buy ({action_reason}): {symbol} (Target: {target_quantity}, Current: {current_quantity}, Buy: {quantity_diff}) - Adjusting position.")
                 rebalance_signals.append({
                    "symbol": symbol,
                    "action": "buy",
                    "quantity": quantity_diff,
                    "price": 0, # 시장가 매수
                    "reason": f"Rebalancing: {action_reason}"
                 })
            elif quantity_diff < 0: # Need to sell some
                sell_quantity = -quantity_diff
                logger.info(f"Rebalance Sell (Reduce): {symbol} (Target: {target_quantity}, Current: {current_quantity}, Sell: {sell_quantity}) - Adjusting position.")
                rebalance_signals.append({
                    "symbol": symbol,
                    "action": "sell",
                    "quantity": sell_quantity,
                    "price": 0, # 시장가 매도
                    "reason": "Rebalancing: Reduce position"
                })
            # else: quantity_diff == 0 -> No change needed for this symbol
        
        # TODO: Add validation pass here or rely on RiskGuard
        # e.g., ensure total sell value + cash >= total buy value?

        logger.info(f"Generated {len(rebalance_signals)} rebalancing signals: {rebalance_signals}")
        return rebalance_signals

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Mock Broker (Updated for get_positions) --- 
    class MockBroker:
        call_count = {'hist_m': 0, 'hist_d': 0, 'pos': 0}
        def get_historical_data(self, symbol, period=None, timeframe='D', start_date=None, end_date=None):
            self.call_count[f'hist_{timeframe.lower()}'] += 1
            print(f"[MockBroker] Requesting {timeframe} data for {symbol} (Start: {start_date}, End: {end_date}, Period: {period})")
            np.random.seed(abs(hash(symbol)) % (2**32 - 1)) # Seed per symbol
            num_periods = 0
            if timeframe == 'M':
                 num_periods = 14 # For ROC
                 dates = pd.date_range(end=datetime.now(), periods=num_periods, freq='M')
                 prices = 30000 + (hash(symbol) % 1000) + np.cumsum(np.random.randn(num_periods) * 500)
                 return pd.DataFrame({'close': prices}, index=dates)
            else: # timeframe == 'D'
                 num_periods = 30 # For ATR
                 dates = pd.date_range(end=datetime.now(), periods=num_periods, freq='B') # Business days
                 price_base = 30000 + hash(symbol) % 1000
                 close = price_base + np.cumsum(np.random.randn(num_periods) * 100)
                 high = close + np.random.rand(num_periods) * 200
                 low = close - np.random.rand(num_periods) * 200
                 # Ensure high >= close >= low
                 high = np.maximum(high, close)
                 low = np.minimum(low, close)
                 return pd.DataFrame({'high': high, 'low': low, 'close': close}, index=dates)
        
        def get_positions(self):
            self.call_count['pos'] += 1
            print("[MockBroker] Requesting current positions...")
            # Simulate different positions based on call count for rebalance test
            if self.call_count['pos'] % 2 == 1:
                return [
                    {'symbol': '069500', 'name': 'KODEX 200', 'quantity': 10, 'average_buy_price': 31000, 'current_price': 32000},
                    {'symbol': '114800', 'name': 'KODEX 인버스', 'quantity': 20, 'average_buy_price': 15000, 'current_price': 14500}
                ]
            else:
                 return [
                     {'symbol': '069500', 'name': 'KODEX 200', 'quantity': 15, 'average_buy_price': 31500, 'current_price': 32500},
                     # '114800' is no longer held
                     {'symbol': '229200', 'name': 'KODEX 코스닥150', 'quantity': 5, 'average_buy_price': 12000, 'current_price': 12500}
                 ]

    # --- Strategy Initialization --- 
    mock_broker = MockBroker()
    strategy = TradingStrategy(
        broker=mock_broker,
        investment_amount=1_000_000, 
        target_symbols=['069500', '114800', '229200']
    )

    # --- Generate Initial Buy Signals --- 
    print("\n--- Generating Initial Buy Signals ---")
    initial_signals = strategy.generate_signals()
    print(f"Initial Buy Signals: {initial_signals}")

    # --- Generate Rebalance Signals (assuming today is rebalance day) --- 
    print("\n--- Generating Rebalance Signals (using get_positions internally) --- ")
    strategy.rebalance_day = datetime.now().day # Force rebalance check to pass
    # current_portfolio = mock_broker.get_positions() # No longer needed as argument
    rebalance_signals = strategy.generate_rebalance_signals()
    print(f"Rebalance Signals: {rebalance_signals}") 