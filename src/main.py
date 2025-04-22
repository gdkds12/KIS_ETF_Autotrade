# FastAPI 애플리케이션 진입점 
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from contextlib import asynccontextmanager
import uvicorn
import logging
import time
import asyncio # Added for background task
from datetime import datetime
from pydantic import BaseModel, Field # For request body validation
from typing import List, Dict, Any # For type hinting
import uuid # For request tracking

from src.config import settings
from src.brokers.kis import KisBroker, KisBrokerError
from src.agents.orchestrator import Orchestrator # Import Orchestrator
from src.db.models import SessionLocal, engine, create_tables # Assuming DB setup
from qdrant_client import QdrantClient
from src.utils.registry import set_orchestrator
from src.agents.finnhub_client import FinnhubClient # 수정: 임포트 경로 변경
from src.agents.memory_rag import MemoryRAG # 추가: Memory RAG 임포트

# --- 로깅 설정 --- 
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Lower logging level for specific modules if needed
# logging.getLogger('src.brokers.kis').setLevel(logging.DEBUG) 
# logging.getLogger('src.agents.strategy').setLevel(logging.DEBUG)
# logging.getLogger('src.agents.risk_guard').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# --- 전역 변수 및 상태 --- 
# 에이전트 및 클라이언트 인스턴스를 저장할 딕셔너리
app_state = {}
background_tasks = set()

# --- Background Trading Loop Logic ---
# This loop is now managed internally by Orchestrator probably, 
# but keep a simple trigger/status check if needed.
# Commenting out the detailed loop here as Orchestrator handles the cycle logic.
# async def background_trading_loop(): ... 

# --- FastAPI Lifespan 관리 --- 
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    # --- Initialize Components (Moved Orchestrator init here) ---
    global app_state # Ensure we modify the global app_state
    try:
        # Create DB tables if they don't exist
        # Consider running migrations separately in production
        create_tables()
        logger.info("Database tables checked/created.")

        # --- Initialize Broker ---
        if not all([settings.APP_KEY, settings.APP_SECRET, settings.CANO, settings.ACNT_PRDT]):
            logger.error("Missing required KIS API credentials or account info in settings.")
            raise RuntimeError("Missing KIS API credentials")
        is_virtual = settings.KIS_VIRTUAL_ACCOUNT
        base_url = settings.KIS_VIRTUAL_URL if is_virtual else settings.BASE_URL
        logger.info(f"Running in {'VIRTUAL' if is_virtual else 'REAL'} mode.")
        broker = KisBroker(
            app_key=settings.APP_KEY,
            app_secret=settings.APP_SECRET,
            base_url=base_url,
            cano=settings.CANO,
            acnt_prdt_cd=settings.ACNT_PRDT,
            virtual_account=is_virtual
        )
        app_state['broker'] = broker
        logger.info("Broker initialized.")

        # --- Initialize Finnhub Client ---
        if settings.FINNHUB_API_KEY:
            finnhub_client = FinnhubClient(token=settings.FINNHUB_API_KEY) # 수정: api_key -> token
            app_state['finnhub_client'] = finnhub_client
            logger.info("Finnhub client initialized.")
        else:
            logger.warning("FINNHUB_API_KEY not found in settings. Finnhub features will be unavailable.")
            app_state['finnhub_client'] = None

        # --- Initialize Qdrant Client --- 
        try:
            qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=10
            )
            # 연결 테스트: collections 가져오기
            _ = qdrant_client.get_collections()
            app_state['qdrant_client'] = qdrant_client
            logger.info("Qdrant client initialized and connected.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}. Orchestrator RAG features might fail.", exc_info=True)
            app_state['qdrant_client'] = None
            
        # --- Initialize Memory RAG --- 
        # Memory RAG는 Qdrant Client가 필요할 수 있음
        if app_state.get('qdrant_client'):
            try:
                memory_rag = MemoryRAG(
                    qdrant_client=app_state['qdrant_client'],
                    # 필요한 다른 설정 전달 (예: embedding model)
                    # embedding_model_name=settings.EMBEDDING_MODEL_NAME 
                )
                app_state['memory_rag'] = memory_rag
                logger.info("Memory RAG initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Memory RAG: {e}. RAG features might fail.", exc_info=True)
                app_state['memory_rag'] = None
        else:
             logger.warning("Qdrant client not available. Memory RAG initialization skipped.")
             app_state['memory_rag'] = None

        # --- Initialize Orchestrator --- 
        # Pass dependencies: broker, db_session_factory, qdrant_client, finnhub_client, memory_rag
        orchestrator = Orchestrator(
            broker=app_state['broker'],
            db_session_factory=SessionLocal, # Pass the factory
            qdrant_client=app_state.get('qdrant_client'), # Use .get() for safety
            finnhub_client=app_state.get('finnhub_client'), # 추가
            memory_rag=app_state.get('memory_rag') # 추가
        )
        app_state['orchestrator'] = orchestrator
        set_orchestrator(orchestrator) # Register orchestrator globally
        logger.info("Orchestrator initialized and registered.")

        # Start background tasks if needed (e.g., Discord bot)
        # discord_bot_task = asyncio.create_task(start_discord_bot(orchestrator))
        # background_tasks.add(discord_bot_task)
        # discord_bot_task.add_done_callback(background_tasks.discard)

        yield # Application is ready to serve requests

    except Exception as e:
        logger.error(f"Error during application startup: {e}", exc_info=True)
        # Ensure partial resources are cleaned up if startup fails mid-way
        if 'broker' in app_state and app_state['broker']:
            # KisBroker.close는 동기 메소드일 수 있으므로 to_thread 사용 고려
            await asyncio.to_thread(app_state['broker'].close) # 수정: close_session -> close, asyncio.to_thread 사용
            logger.info("Broker session closed (on startup error).")
        if 'qdrant_client' in app_state and app_state['qdrant_client']:
             # Qdrant client might not have an explicit close in the sdk
             logger.info("Qdrant client resources released (if applicable).")
        # Re-raise the exception to signal FastAPI lifespan failure
        raise e

    # --- Shutdown Logic --- 
    logger.info("Application shutdown...")
    # Gracefully cancel background tasks if any were started
    # for task in list(background_tasks):
    #     if not task.done(): task.cancel()
    # if background_tasks:
    #      logger.info(f"Waiting for {len(background_tasks)} background tasks to cancel...")
    #      await asyncio.gather(*background_tasks, return_exceptions=True)
    #      logger.info("Background tasks cancelled.")

    if 'broker' in app_state and app_state['broker']:
        # KisBroker.close는 동기 메소드일 수 있으므로 to_thread 사용 고려
        await asyncio.to_thread(app_state['broker'].close) # 수정: close_session -> close, asyncio.to_thread 사용
        logger.info("Broker session closed.")
    if 'qdrant_client' in app_state and app_state['qdrant_client']:
         try:
             # Qdrant client might not have an explicit close, depends on implementation
             # app_state['qdrant_client'].close()
             logger.info("Qdrant client resources released (if applicable).")
         except Exception as q_close_e:
              logger.warning(f"Error closing Qdrant client: {q_close_e}")
                  
    logger.info("Application shutdown complete.")

# --- Pydantic Models for API --- 
class OrderConfirmationPayload(BaseModel):
    request_id: uuid.UUID = Field(..., description="The unique ID of the confirmation request sent to Discord.")
    approved_orders: List[Dict[str, Any]] = Field(..., description="List of order dictionaries approved by the user.")

# --- FastAPI 앱 생성 --- 
app = FastAPI(
    title="KIS ETF Autotrade API",
    description="LLM-driven ETF autotrading system using KIS OpenAPI.",
    version="1.2.0", # Incremented version
    lifespan=lifespan # Lifespan 이벤트 핸들러 등록
)

# --- Dependency Functions (if needed) --- 
def get_orchestrator() -> Orchestrator:
    orchestrator = app_state.get('orchestrator')
    if not orchestrator:
         # This shouldn't happen if lifespan completes successfully
         raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator

# --- API Endpoints --- 

@app.get("/")
async def read_root():
    return {"message": "KIS ETF Autotrade API is running."}

@app.get("/health")
async def health_check(broker: KisBroker = Depends(lambda: app_state.get('broker'))): # Get broker from state
    if not broker:
         return {"status": "error", "detail": "Broker not initialized"}
    try:
        # Check if token is valid (or get a new one implicitly)
        token_valid = await asyncio.to_thread(broker.check_token)
        if token_valid:
            return {"status": "ok", "kis_token": "valid"}
        else:
             # If check_token refreshes, it might be okay, but let's indicate potential issue
             return {"status": "warning", "kis_token": "refreshed or invalid"}
    except KisBrokerError as e:
        logger.error(f"Health check failed during KIS token check: {e}")
        return {"status": "error", "detail": f"KIS Broker Error: {e}"}
    except Exception as e:
         logger.error(f"Health check failed unexpectedly: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/confirm_order")
async def confirm_order_endpoint(payload: OrderConfirmationPayload, background_tasks: BackgroundTasks, orchestrator: Orchestrator = Depends(get_orchestrator)):
    """Endpoint called by Discord bot when user approves orders."""
    request_id = payload.request_id
    approved_orders = payload.approved_orders
    logger.info(f"Received order confirmation callback from Discord for request ID: {request_id} with {len(approved_orders)} orders.")

    if not approved_orders:
        logger.warning(f"Confirmation received for request {request_id}, but no orders were provided for execution.")
        # Acknowledge receipt but indicate no action taken
        return {"status": "received", "message": "No orders to execute.", "request_id": request_id}

    # --- Execute orders in the background --- 
    # Use background tasks to avoid blocking the response to Discord bot
    try:
        # Define the background task function
        async def execute_and_save(orders):
            logger.info(f"Background task started for executing {len(orders)} approved orders (Request ID: {request_id})...")
            try:
                # Orchestrator methods might be synchronous, run in threadpool
                order_exec_results = await asyncio.to_thread(orchestrator._execute_orders, orders)
                logger.info(f"Order execution finished for request {request_id}. Results count: {len(order_exec_results)}")
                
                # Save results (might also be synchronous)
                await asyncio.to_thread(orchestrator._upsert_trade_results, order_exec_results)
                logger.info(f"Execution results saved for request {request_id}.")
                
                # Send notification (if needed, might also be sync/async)
                # await asyncio.to_thread(orchestrator.send_notification, f"User approved orders executed for request {request_id}.")
            except Exception as bg_e:
                 logger.error(f"Error during background order execution/saving for request {request_id}: {bg_e}", exc_info=True)
                 # TODO: Implement more robust error handling/notification for background failures

        background_tasks.add_task(execute_and_save, approved_orders)
        
        logger.info(f"Background task scheduled for order execution (Request ID: {request_id}).")
        # Return success quickly to Discord
        return {"status": "processing", "message": "Order execution started in background.", "request_id": request_id}

    except Exception as e:
        logger.error(f"Failed to schedule background task for order execution (Request ID: {request_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start order execution for request {request_id}")

# --- Manual Trigger Endpoint --- 
@app.post("/trigger_cycle")
async def trigger_cycle_endpoint(background_tasks: BackgroundTasks, orchestrator: Orchestrator = Depends(get_orchestrator)):
    """Manually triggers the Orchestrator's daily cycle in the background."""
    logger.info("Received manual trigger request for the daily cycle.")
    try:
        # Define the background task
        async def run_cycle_task():
            logger.info("Background task started for running the daily cycle...")
            try:
                # Assuming run_daily_cycle handles its own errors internally
                # Run synchronous method in threadpool if it blocks
                await asyncio.to_thread(orchestrator.run_daily_cycle)
                logger.info("Background daily cycle task finished.")
            except Exception as cycle_err:
                 logger.error(f"Error during background daily cycle execution: {cycle_err}", exc_info=True)
                 # TODO: Notify admin/user about the cycle failure?

        background_tasks.add_task(run_cycle_task)
        logger.info("Background task scheduled for running the daily cycle.")
        return {"status": "triggered", "message": "Orchestrator cycle triggered to run in background."}

    except Exception as e:
        logger.error(f"Failed to schedule background task for daily cycle trigger: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to trigger orchestrator cycle")

# --- Main Execution (for running directly) ---
if __name__ == "__main__":
    # Note: Running Uvicorn directly might not be ideal for managing dependencies
    #       and lifespans correctly compared to using `uvicorn src.main:app`
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)