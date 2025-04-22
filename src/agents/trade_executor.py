"""
TradeExecutor – ①RiskGuard 통과한 주문을 대기열에 올리고
②사용자 확인 이후 KIS에 실제 발주
"""
import logging
import asyncio
from typing import List, Dict, Any
from src.brokers.kis import KisBroker, KisBrokerError

logger = logging.getLogger(__name__)

class TradeExecutor:
    MAX_RETRIES = 3
    BACKOFF_SEC = 2

    def __init__(self, broker: KisBroker):
        self.broker = broker

    # ------------------------------------------------------------------
    async def execute_batch(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """non-blocking 실행; 내부에서 sleep 대신 asyncio.sleep 사용"""
        results: List[Dict[str, Any]] = []
        for o in orders:
            try:
                res = self.broker.order_cash(
                    symbol=o["symbol"],
                    quantity=o["quantity"],
                    price=0,
                    order_type="01",
                    buy_sell_code="02" if o["action_type"]=="buy" else "01"
                )
                results.append({"order": o, "status": "success", "kis": res})
                await asyncio.sleep(0.2)
            except KisBrokerError as e:
                results.append({"order": o, "status": "failed", "error": str(e)})
        return results
