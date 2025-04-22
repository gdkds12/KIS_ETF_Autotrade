"""
PortfolioManager – 현재 잔고·포지션·예수금 등의 스냅샷 + 유틸
"""
import logging
from typing import Dict, List, Any
from src.brokers.kis import KisBroker

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self, broker: KisBroker):
        self.broker = broker

    def snapshot(self) -> Dict[str, Any]:
        """잔고, 포지션, 그리고 기본 성과 지표(Box)까지 포함"""
        balance   = self.broker.get_balance()
        positions = self.broker.get_positions()
        perf      = self._calc_performance(positions, balance)
        return {"balance": balance, "positions": positions, "performance": perf}

    # ------------------------ Helper ------------------------
    def _calc_performance(self, positions, balance) -> dict:
        import pandas as pd
        port_val = sum(p["current_price"]*p["quantity"] for p in positions)
        cash     = balance.get("available_cash", 0)
        total    = port_val + cash
        daily_ret = self._compute_daily_return(positions)
        return {"total_value": total, "port_value": port_val,
                "cash": cash, "daily_return": daily_ret}

    def _compute_daily_return(self, positions):
        try:
            import numpy as np
            rets = []
            for p in positions:
                df = self.broker.get_historical_data(p["symbol"], period=2)
                if len(df) >= 2:
                    rets.append(df["close"].pct_change().iloc[-1])
            return float(np.nanmean(rets)) if rets else 0.0
        except Exception:
            return 0.0

    def get_position_qty(self, symbol: str) -> int:
        for p in self.broker.get_positions():
            if p["symbol"] == symbol:
                return p["quantity"]
        return 0
