# Qdrant 벡터 저장·검색
import logging
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
# from sqlalchemy.orm import Session # For DB interaction if needed
# from some_embedding_model import get_embedding # Placeholder
import uuid
import openai
from sentence_transformers import SentenceTransformer # 실제 임베딩 모델 라이브러리
from tenacity import retry, stop_after_attempt, wait_fixed # Qdrant 연결 재시도
from sqlalchemy.orm import Session
from sqlalchemy import select # select 함수 임포트
import datetime
# import google.generativeai as genai # Gemini 라이브러리 임포트

# 내부 config 모듈에서 설정 로드
from src.config import settings
from src.db.models import TradingSession, SessionLog # DB 모델 임포트
# Placeholder for LLM summarization function
# from some_llm_library import summarize_text

logger = logging.getLogger(__name__)

# ✅ GPT-4o 또는 o4-mini 계열은 max_completion_tokens, 그 외는 max_tokens 사용 (SDK 최신 버전 기준)
def get_token_param(model: str, limit: int) -> dict:
    if model.startswith("o4") or model.startswith("gpt-4o"):
        return {"max_completion_tokens": limit}
    else:
        return {"max_tokens": limit}

def get_temperature_param(model: str, temperature: float) -> dict:
    if model.startswith("o4") or model.startswith("gpt-4o"):
        return {}  # 기본값 1.0만 지원
    else:
        return {"temperature": temperature}

# --- OpenAI 모델 초기화 (MemoryRAG 용) ---
# Rely on global setting of openai.api_key done elsewhere (e.g., orchestrator, bot setup)
if settings.OPENAI_API_KEY:
    # openai.api_key = settings.OPENAI_API_KEY # Avoid setting globally multiple times
    logger.info(f"MemoryRAG will use OpenAI model for summarization: {settings.LLM_LIGHTWEIGHT_TIER_MODEL}")
else:
    logger.warning("OPENAI_API_KEY not set. Session summarization will use placeholder.")

# --- LLM 요약 함수 (Now using OpenAI) --- 
def summarize_text(text: str, topic: str = None) -> str:
    """LLM을 사용하여 텍스트 요약하기"""
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    system_prompt = (
        "당신은 텍스트 요약 전문가입니다. 다음 텍스트를 명확하고 간결하게 요약해주세요.\n"
        "요약은 원문의 핵심 정보와 중요한 세부 사항을 보존해야 합니다.\n"
        "불필요한 세부사항은 제외하고, 원문의 주요 주제와 논점에 집중하세요.\n"
        "한국어로 3-5문장 분량으로 요약하세요."
    )
    
    user_content = f"다음 텍스트를 요약해주세요:\n\n{text}"
    if topic:
        user_content += f"\n\n이 텍스트는 '{topic}' 주제와 관련이 있습니다."
        
    try:
        resp = client.chat.completions.create(
            model=settings.LLM_SUMMARY_TIER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            **get_temperature_param(settings.LLM_SUMMARY_TIER_MODEL, 0.3),
            **get_token_param(settings.LLM_SUMMARY_TIER_MODEL, 300),
        )
        summary = resp.choices[0].message.content.strip()
        logger.info("Successfully received summary from OpenAI.")
        return summary
    except openai.APIError as e:
        logger.error(f"OpenAI API Error during summarization: {e}", exc_info=True)
        return f"(OpenAI API 오류: {e})"
    except Exception as e:
        logger.error(f"OpenAI summarization failed: {e}", exc_info=True)
        return f"(요약 불가: {e})"

class MemoryRAG:
    def __init__(self, db_session_factory = None):
        """MemoryRAG 초기화

        Args:
            db_session_factory: SQLAlchemy 세션 팩토리 (e.g., SessionLocal). 제공되지 않으면 DB 연동 기능 제한.
        """
        self.qdrant_uri = settings.QDRANT_URL
        self.qdrant_api_key = settings.QDRANT_API_KEY
        self.collection_name = "autotrade_memory"
        self.vector_dim = settings.VECTOR_DIM
        self.embedding_model_name = settings.EMBEDDING_MODEL_NAME
        self.db_session_factory = db_session_factory # 세션 팩토리 저장

        # OpenAI Embedding API 초기화
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY must be set to use OpenAI embeddings")
        openai.api_key = settings.OPENAI_API_KEY
        self.embedding_model = openai
        logger.info(f"Using OpenAI embedding model: {self.embedding_model_name} (vector_dim={self.vector_dim})")

        # Qdrant 클라이언트 초기화
        self.qdrant_client = self._init_qdrant_client()
        self._ensure_collection_exists()
        logger.info(f"MemoryRAG initialized ({'with DB' if db_session_factory else 'without DB'} factory).")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def _init_qdrant_client(self) -> QdrantClient:
        """Qdrant 클라이언트를 초기화하고 연결을 시도합니다."""
        try:
            logger.info(f"Connecting to Qdrant at {self.qdrant_uri}...")
            client = QdrantClient(url=self.qdrant_uri, api_key=self.qdrant_api_key, timeout=10)
            # 연결 테스트: collections 가져오기
            _ = client.get_collections()
            logger.info("Successfully connected to Qdrant.")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _ensure_collection_exists(self):
        """Qdrant에 필요한 컬렉션이 없으면 생성합니다."""
        try:
            collection_info = self.qdrant_client.get_collection(collection_name=self.collection_name)
            existing_dim = collection_info.vectors_config.params.size
            if existing_dim != self.vector_dim:
                logger.warning(f"Qdrant collection '{self.collection_name}' exists but has different vector dimension ({existing_dim}) than model ({self.vector_dim}). Consider recreating collection.")
            logger.info(f"Qdrant collection '{self.collection_name}' exists with dimension {existing_dim}.")
        except Exception as e:
            logger.warning(f"Collection '{self.collection_name}' not found or error checking: {e}. Attempting to create...")
            try:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.vector_dim, distance=models.Distance.COSINE)
                )
                logger.info(f"Collection '{self.collection_name}' created with dimension {self.vector_dim}.")
            except Exception as create_e:
                # 이미 존재해서 409 Conflict 가 떴다면 그냥 무시
                msg = str(create_e)
                if 'already exists' in msg or 'Conflict' in msg:
                    logger.warning(f"Collection '{self.collection_name}' already exists, skipping creation.")
                else:
                    logger.error(f"Failed to create Qdrant collection: {create_e}", exc_info=True)
                    raise RuntimeError(f"Failed to create Qdrant collection: {create_e}") from create_e

    def get_embedding(self, text: str) -> list[float]:
        """주어진 텍스트에 대한 임베딩 벡터를 생성합니다."""
        try:
            resp = self.embedding_model.Embedding.create(
                model=self.embedding_model_name,
                input=text
            )
            return resp["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: '{text[:50]}...': {e}", exc_info=True)
            raise ValueError(f"Embedding failed: {e}") from e

    def save_memory(self, text: str, metadata: dict = None):
        """텍스트와 메타데이터를 벡터화하여 Qdrant에 저장합니다."""
        if not text:
            logger.warning("Attempted to save empty text to memory. Skipping.")
            return
        try:
            vector = self.get_embedding(text)
            doc_id = str(uuid.uuid4())
            payload = metadata if metadata else {}
            payload["text"] = text # 원본 텍스트 저장
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=vector,
                        payload=payload
                    )
                ],
                wait=True
            )
            logger.info(f"Saved memory item with ID: {doc_id}, Metadata: {metadata}")
        except ValueError as e: # 임베딩 실패 시
             logger.error(f"Skipping memory save due to embedding error: {e}")
        except Exception as e:
            # doc_id가 정의되지 않았을 수 있으므로 try 블록 밖에서 사용하지 않음
            logger.error(f"Failed to save memory: {e}", exc_info=True)

    def search_similar(self, query_text: str, limit: int = 5, score_threshold: float = 0.5) -> list:
        """주어진 텍스트와 유사한 메모리를 Qdrant에서 검색합니다."""
        if not query_text:
            logger.warning("Attempted to search with empty query text. Returning empty list.")
            return []
        try:
            query_vector = self.get_embedding(query_text)
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            logger.info(f"Found {len(search_result)} similar items for query '{query_text[:50]}...' with threshold {score_threshold}.")
            return search_result
        except ValueError as e: # 임베딩 실패 시
             logger.error(f"Skipping memory search due to query embedding error: {e}")
             return []
        except Exception as e:
            logger.error(f"Failed to search memory for query '{query_text[:50]}...': {e}", exc_info=True)
            return []

    def summarize_and_upsert(self, session_uuid: str):
        """특정 세션의 로그를 DB에서 조회하여 요약하고 결과를 Qdrant와 DB에 저장합니다.

        Args:
            session_uuid: 요약할 세션의 내부 UUID
        """
        if not self.db_session_factory:
             logger.warning("DB session factory not provided. Cannot summarize session from DB.")
             return
             
        logger.info(f"Attempting to summarize session {session_uuid} from DB...")
        db: Session = self.db_session_factory()
        try:
            # 1. DB에서 세션 및 로그 조회
            stmt = select(TradingSession).where(TradingSession.session_uuid == session_uuid)
            session_obj = db.scalars(stmt).first()
            
            if not session_obj:
                logger.warning(f"TradingSession with UUID {session_uuid} not found in DB.")
                return

            if not session_obj.is_active: # 이미 종료된 세션은 요약하지 않음 (옵션)
                 logger.info(f"Session {session_uuid} is already inactive. Skipping summarization.")
                 # return
                 
            # 로그 조회 (최신 순으로 가져오는 것이 좋을 수 있음)
            log_stmt = select(SessionLog).where(SessionLog.session_id == session_obj.id).order_by(SessionLog.timestamp.asc())
            logs = db.scalars(log_stmt).all()
            
            if not logs:
                logger.warning(f"No logs found for session {session_uuid} (DB ID: {session_obj.id}). Cannot summarize.")
                return

            # 2. 로그 텍스트 조합
            conversation_text = "\n".join([f"{log.actor}: {log.message}" for log in logs])
            logger.info(f"Retrieved {len(logs)} logs for session {session_uuid} for summarization.")

            # 3. LLM 요약 호출 (Now uses OpenAI)
            summary_text = summarize_text(conversation_text)

            # 4. Qdrant에 요약 저장
            self.save_memory(summary_text, metadata={
                "type": "session_summary", 
                "session_uuid": session_uuid,
                "discord_thread_id": session_obj.discord_thread_id,
                "discord_user_id": session_obj.discord_user_id
            })

            # 5. DB에 세션 상태 및 요약 업데이트
            session_obj.summary = summary_text
            session_obj.is_active = False # 세션 비활성화
            session_obj.end_time = datetime.datetime.now(datetime.timezone.utc) # 종료 시간 기록
            db.commit()
            logger.info(f"Session {session_uuid} summarized, saved to Qdrant, and marked inactive in DB.")

        except Exception as e:
            logger.error(f"Error during summarize_and_upsert for session {session_uuid}: {e}", exc_info=True)
            db.rollback() # 오류 발생 시 롤백
        finally:
            db.close()

    def save_trade_results(self, trade_results: list):
        """(Deprecated) 거래 결과를 메모리(Qdrant)에 저장합니다. 대신 save_execution_results 사용 권장."""
        logger.warning("save_trade_results is deprecated. Use save_execution_results instead.")
        # Keep for backward compatibility or simple cases if needed
        saved_count = 0
        for result in trade_results:
            try:
                status = "성공" if result.get("rt_cd") == "0" else "실패"
                order_info = result.get('order', {})
                symbol = order_info.get('symbol', 'N/A')
                action = order_info.get('action', 'N/A').upper()
                quantity = order_info.get('quantity', 'N/A')
                # Simple text representation
                text = f"거래 결과 [{status}]: {action} {symbol} {quantity}주. 메시지: {result.get('msg1', 'N/A')}"
                metadata = {
                    "type": "trade_result",
                    "status": status.lower(),
                    "symbol": symbol,
                    "action": action.lower(),
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                self.save_memory(text, metadata=metadata)
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save single trade result to memory: {result}. Error: {e}", exc_info=True)
        logger.info(f"(Deprecated) Saved {saved_count}/{len(trade_results)} trade results to memory.")

    def save_execution_results(self, execution_results: list):
        """Orchestrator의 실행 결과 리스트를 메모리(Qdrant)에 저장합니다."""
        logger.info(f"Saving {len(execution_results)} execution results to memory...")
        saved_count = 0
        failed_count = 0
        cycle_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for result in execution_results:
            try:
                action_type = result.get('action_type', 'unknown')
                status = result.get('status', 'unknown')
                detail = result.get('detail', 'N/A')
                metadata = {
                    "type": "execution_result",
                    "action_type": action_type,
                    "status": status,
                    "timestamp": cycle_timestamp # Use same timestamp for the whole cycle
                }
                text = f"실행 결과 [{action_type.upper()}/{status.upper()}]: {detail}"

                if action_type in ['buy', 'sell']:
                    order_info = result.get('order', {})
                    metadata['symbol'] = order_info.get('symbol')
                    metadata['quantity'] = order_info.get('quantity')
                    metadata['reason'] = order_info.get('reason')
                    text += f" (Order: {order_info.get('action')} {metadata['symbol']} {metadata['quantity']}주, Reason: {metadata['reason']})"
                elif action_type == 'briefing_summary':
                     metadata['notes'] = result.get('notes')
                     text = f"실행 결과 [LLM 브리핑 요약]: {len(metadata.get('notes', []))}개 노트"
                elif action_type == 'briefing':
                     text = f"실행 결과 [LLM 브리핑 노트]: {detail}"
                elif action_type == 'hold':
                     metadata['reason'] = detail
                     text = f"실행 결과 [HOLD]: {detail}"

                # Add KIS response if available (maybe just key info)
                kis_response = result.get('kis_response')
                if kis_response:
                    metadata['kis_rt_cd'] = kis_response.get('rt_cd')
                    metadata['kis_msg1'] = kis_response.get('msg1')
                    metadata['kis_odno'] = kis_response.get('ODNO')

                # Save to Qdrant
                self.save_memory(text, metadata=metadata)
                saved_count += 1

            except Exception as e:
                logger.error(f"Failed to save execution result to memory: {result}. Error: {e}", exc_info=True)
                failed_count += 1
        
        logger.info(f"Finished saving execution results. Saved: {saved_count}, Failed: {failed_count}")

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    from src.db.models import SessionLocal # SessionLocal 가져오기
    from src.db.models import create_tables

    if not settings.QDRANT_URL or not settings.DATABASE_URL:
        print("QDRANT_URL or DATABASE_URL is not configured. Exiting.")
        exit()

    # DB 테이블 생성 (테스트용)
    create_tables()

    # --- DB에 테스트 세션 및 로그 생성 --- 
    test_session_uuid = str(uuid.uuid4())
    db = SessionLocal()
    try:
        # 기존 테스트 세션 삭제 (옵션)
        existing_session = db.query(TradingSession).filter(TradingSession.session_uuid == test_session_uuid).first()
        if existing_session:
             db.delete(existing_session)
             db.commit()
             
        new_session = TradingSession(
            session_uuid=test_session_uuid,
            discord_thread_id="discord_thread_12345",
            discord_user_id="discord_user_67890"
        )
        db.add(new_session)
        db.flush() # ID를 얻기 위해 flush
        
        log1 = SessionLog(session_id=new_session.id, actor="user", message="KODEX 200 매수해도 될까요?")
        log2 = SessionLog(session_id=new_session.id, actor="ai", message="현재 시장 상황을 고려할 때 매수는 신중해야 합니다.")
        log3 = SessionLog(session_id=new_session.id, actor="user", message="알겠습니다. 감사합니다.")
        db.add_all([log1, log2, log3])
        db.commit()
        print(f"Created test session {test_session_uuid} with ID {new_session.id} and 3 logs in DB.")
    except Exception as db_e:
         print(f"Error setting up test data in DB: {db_e}")
         db.rollback()
         exit()
    finally:
         db.close()
    # --- End DB Setup ---

    try:
        # MemoryRAG 인스턴스 생성 (DB 세션 팩토리 전달)
        memory = MemoryRAG(db_session_factory=SessionLocal)
        
        # --- 테스트: 세션 요약 및 저장 --- 
        print(f"\n--- Summarizing Session {test_session_uuid} --- ")
        memory.summarize_and_upsert(test_session_uuid)
        
        # --- 확인: Qdrant에서 요약 검색 --- 
        summary_query = "KODEX 200 관련 대화"
        summary_results = memory.search_similar(summary_query, limit=1, score_threshold=0.1) # 낮은 임계값으로 설정
        print(f"\nSearch for summary ('{summary_query}') in Qdrant:")
        found = False
        if summary_results:
             for hit in summary_results:
                 if hit.payload.get("type") == "session_summary" and hit.payload.get("session_uuid") == test_session_uuid:
                     print(f"- Found Summary! Score: {hit.score:.4f}")
                     print(f"  Payload: {hit.payload}")
                     found = True
                     break
        if not found:
             print("Summary not found in Qdrant or threshold too high.")
             
        # --- 확인: DB에서 세션 상태 확인 --- 
        db = SessionLocal()
        try:
            updated_session = db.query(TradingSession).filter(TradingSession.session_uuid == test_session_uuid).first()
            if updated_session:
                 print(f"\nSession status in DB: is_active={updated_session.is_active}, end_time={updated_session.end_time}")
                 print(f"Session summary in DB: {updated_session.summary}")
            else:
                 print("Session not found in DB after update attempt.")
        finally:
             db.close()

    except RuntimeError as e:
         print(f"\nInitialization Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the example run: {e}")

    # Verify trade results were saved
    trade_query = "trade results"
    trade_search_results = memory.search_similar(trade_query)
    print(f"\n--- Search Results for '{trade_query}' ---")
    if trade_search_results:
        for hit in trade_search_results:
             if hit.payload.get("type") == "trade_result":
                print(f"- Score: {hit.score:.4f}, ID: {hit.id}, Payload: {hit.payload}")
    else:
         print("No trade results found in memory.") 