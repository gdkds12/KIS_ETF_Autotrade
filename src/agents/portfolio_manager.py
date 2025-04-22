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
        """잔고·포지션·총액 등을 한 번에 리턴"""
        balance = self.broker.get_balance()
        positions = self.broker.get_positions()
        return {
            "balance": balance,
            "positions": positions
        }

    def get_position_qty(self, symbol: str) -> int:
        for p in self.broker.get_positions():
            if p["symbol"] == symbol:
                return p["quantity"]
        return 0
