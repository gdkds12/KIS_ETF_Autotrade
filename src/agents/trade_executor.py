"""
TradeExecutor – ①RiskGuard 통과한 주문을 대기열에 올리고
②사용자 확인 이후 KIS에 실제 발주
"""
import logging, time
from typing import List, Dict, Any
from src.brokers.kis import KisBroker, KisBrokerError

logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self, broker: KisBroker):
        self.broker = broker

    # ------------------------------------------------------------------
    def execute_batch(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """blocking 실행; 호출측에서 to_thread 사용 권장"""
        results = []
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
                time.sleep(0.2)  # API 간격
            except KisBrokerError as e:
                results.append({"order": o, "status": "failed", "error": str(e)})
        return results
