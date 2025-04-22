"""
TradeExecutor – ①RiskGuard 통과한 주문을 대기열에 올리고
②사용자 확인 이후 KIS에 실제 발주
"""
import logging, time
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
        """⚡ 비동기 병렬 실행 + 재시도"""
        import asyncio, random
        semaphore = asyncio.Semaphore(5)  # 동시 5 건 제한

        async def _exec(o):
            async with semaphore:
                for attempt in range(1, self.MAX_RETRIES+1):
                    try:
                        res = await asyncio.to_thread(
                            self.broker.order_cash,
                            symbol=o["symbol"],
                            quantity=o["quantity"],
                            price=0,
                            order_type="01",
                            buy_sell_code="02" if o["action_type"]=="buy" else "01"
                        )
                        return {"order": o, "status": "success", "kis": res}
                    except KisBrokerError as e:
                        if attempt == self.MAX_RETRIES:
                            return {"order": o, "status": "failed", "error": str(e)}
                        await asyncio.sleep(self.BACKOFF_SEC * attempt +
                                            random.uniform(0, 1))

        return await asyncio.gather(*[_exec(o) for o in orders])
