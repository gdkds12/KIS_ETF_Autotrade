import pandas as pd
from datetime import datetime, timedelta
from src.config import settings

class Recommender:
    """
    사용자 프로필과 시장 데이터를 기반으로 US ETF 추천 및 포트폴리오 가중치 계산
    """
    def __init__(
        self,
        finnhub_client,
        target_return: float = settings.TARGET_RETURN,
        risk_tolerance: str = settings.RISK_TOLERANCE,
        candidates: list[str] | None = None
    ):
        self.finnhub = finnhub_client
        self.target_return = target_return
        self.risk_tolerance = risk_tolerance
        # 외부에서 전달하거나 설정에 정의된 후보군 사용
        self.candidates = candidates or settings.CANDIDATE_SYMBOLS
        self.risk_profile_map = {"conservative": 0.05,
                                 "moderate":    0.07,
                                 "aggressive":  0.12}

    def _fetch_price_series(self, symbol: str, period_days: int = 365) -> pd.DataFrame:
        """Finnhub에서 일별 종가 데이터를 가져와 DataFrame으로 반환"""
        to_ts = int(datetime.now().timestamp())
        from_ts = int((datetime.now() - timedelta(days=period_days)).timestamp())
        data = self.finnhub.get_candles(symbol, resolution='D', _from=from_ts, to=to_ts)
        # Finnhub 응답에 'c' 리스트(종가)와 't' 리스트(타임스탬프) 가정
        df = pd.DataFrame({'close': data.get('c', [])}, index=pd.to_datetime(data.get('t', []), unit='s'))
        return df

    def _compute_annualized_return(self, df: pd.DataFrame) -> float:
        """기간 수익률을 연환산 수익률로 변환"""
        if df.empty:
            return 0.0
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        total_return = (end_price / start_price) - 1
        days = (df.index[-1] - df.index[0]).days or 1
        annual_return = (1 + total_return) ** (365 / days) - 1
        return annual_return

    def _compute_volatility(self, df: pd.DataFrame) -> float:
        """일별 수익률 표준편차를 연환산 변동성으로 반환"""
        if df.empty or len(df) < 2:
            return float('inf')
        returns = df['close'].pct_change().dropna()
        daily_vol = returns.std()
        annual_vol = daily_vol * (252 ** 0.5)
        return annual_vol

    def _max_drawdown(self, df: pd.DataFrame) -> float:
        if df.empty: return 0.0
        roll_max = df['close'].cummax()
        dd = (df['close']/roll_max - 1).min()
        return abs(dd)

    def _refresh_candidates(self):
        # TODO: implement dynamic candidate refresh logic
        pass

    def recommend(self) -> dict:
        """
        연환산 수익률이 목표 이상인 ETF 중 Sharpe Ratio 기준 상위 종목 선정 후 가중치 부여

        Returns:
            {
                'recommendations': [
                    {'symbol': str, 'annual_return': float, 'volatility': float, 'sharpe': float},
                    ...
                ],
                'weights': {symbol: weight, ...}
            }
        """
        scored = []
        for sym in self.candidates:
            df = self._fetch_price_series(sym)
            ann_ret = self._compute_annualized_return(df)
            vol = self._compute_volatility(df)
            sharpe = ann_ret / vol if vol > 0 else 0
            max_dd = self._max_drawdown(df)
            if ann_ret >= self.risk_profile_map.get(self.risk_tolerance, self.target_return):
                score = 0.6*sharpe + 0.3*ann_ret - 0.1*max_dd
                scored.append({'symbol': sym, 'annual_return': ann_ret,
                               'volatility': vol, 'max_dd': max_dd,
                               'sharpe': sharpe, 'score': score})

        # Sharpe 비율 기준으로 내림차순 정렬
        scored_sorted = sorted(scored, key=lambda x: x['score'], reverse=True)
        top_n = scored_sorted[:3]
        n = len(top_n)
        if n == 0:
            return {'recommendations': [], 'weights': {}}
        weight = 1 / n
        weights = {item['symbol']: weight for item in top_n}
        return {'recommendations': top_n, 'weights': weights}

    # --------------- Helper ----------------
    @staticmethod
    def _max_drawdown(df: pd.DataFrame) -> float:
        if df.empty: return 0.0
        roll_max = df['close'].cummax()
        dd = (df['close']/roll_max - 1).min()
        return abs(dd)