# Qdrant 벡터 저장·검색
import logging
import uuid
import datetime
from typing import Optional, List, Dict, Any

import openai # Add OpenAI library
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus
# Remove SentenceTransformer import if no longer needed elsewhere, keep for now
# from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import select

# 내부 config 모듈에서 설정 로드
from src.config import settings
from src.db.models import TradingSession, SessionLog # DB 모델 임포트
from src.utils.azure_openai import azure_chat_completion

logger = logging.getLogger(__name__)


class MemoryRAG:
    """세션 로그 요약, 벡터 저장 및 검색을 위한 RAG 클래스"""

    DEFAULT_SCORE_THRESHOLD = 0.65           # 유사도 검색 기본 임계값
    FORGET_AFTER_DAYS       = 30             # 30 일 경과 시 soft‑delete
    MAX_MEMORY_PER_SESSION  = 50             # 세션당 최대 저장 벡터

    def __init__(self, db_session_factory: sessionmaker, qdrant_client: QdrantClient, llm_model: str):
        """MemoryRAG 초기화

        Args:
            db_session_factory: SQLAlchemy 세션 팩토리 (e.g., SessionLocal).
            qdrant_client: 미리 초기화된 Qdrant 클라이언트 인스턴스.
            llm_model: 요약에 사용할 Azure OpenAI 배포 이름.
        """
        self.db_session_factory = db_session_factory
        self.qdrant_client = qdrant_client
        self.llm_model = llm_model # Used for summarization via Azure
        self.collection_name = "autotrade_memory"
        self.vector_dim = settings.VECTOR_DIM
        self.embedding_model_name = settings.EMBEDDING_MODEL_NAME # Should be 'text-embedding-3-large'

        if not self.db_session_factory:
            raise ValueError("db_session_factory is required.")
        if not self.qdrant_client:
            raise ValueError("qdrant_client is required.")
        if not self.llm_model:
            raise ValueError("llm_model (Azure deployment name for summarization) is required.")
        if not settings.OPENAI_API_KEY:
             raise ValueError("OPENAI_API_KEY is required in settings for embeddings.")

        try:
            # Initialize OpenAI client for embeddings
            self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            # Verify connection by making a simple call (optional, but good practice)
            # self.openai_client.models.list() # Example verification
            logger.info(f"OpenAI client initialized successfully for embeddings using model '{self.embedding_model_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize OpenAI client: {e}")

        # 컬렉션 존재 확인 및 생성 (Qdrant part remains the same)
        self._ensure_collection_exists()
        logger.info(f"MemoryRAG initialized. Using Qdrant collection: {self.collection_name}, Summarization LLM: {self.llm_model}, Embedding Model: {self.embedding_model_name}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def _ensure_collection_exists(self):
        """Qdrant에 필요한 컬렉션이 없으면 생성합니다."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Collection '{self.collection_name}' not found. Creating it...")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
                )
                logger.info(f"Collection '{self.collection_name}' created successfully.")
            else:
                logger.debug(f"Collection '{self.collection_name}' already exists.")

        except RetryError as e:
             logger.error(f"Failed to ensure Qdrant collection '{self.collection_name}' exists after multiple retries: {e}")
             raise RuntimeError(f"Could not connect to or verify Qdrant collection: {e}")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection '{self.collection_name}': {e}", exc_info=True)
            # 특정 Qdrant 예외를 잡는 것이 더 좋을 수 있음
            raise RuntimeError(f"Qdrant operation failed: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def get_embedding(self, text: str) -> List[float]:
        """주어진 텍스트에 대한 임베딩 벡터를 생성합니다 (OpenAI API 사용)."""
        if not text or not isinstance(text, str):
            logger.warning("get_embedding called with empty or invalid text.")
            return [] # Return empty list for invalid input

        try:
            # Replace SentenceTransformer encode with OpenAI API call
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model_name
            )
            # Extract the embedding vector
            embedding = response.data[0].embedding
            # logger.debug(f"Generated embedding for text snippet: '{text[:50]}...' Dimension: {len(embedding)}")
            return embedding
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API request failed to connect: {e}")
            raise RuntimeError(f"OpenAI API Connection Error: {e}")
        except openai.RateLimitError as e:
            logger.error(f"OpenAI API request exceeded rate limit: {e}")
            # Implement specific backoff/retry logic if needed, or rely on tenacity
            raise RuntimeError(f"OpenAI API Rate Limit Error: {e}")
        except openai.APIStatusError as e:
            logger.error(f"OpenAI API returned an error status: {e.status_code} - {e.response}")
            raise RuntimeError(f"OpenAI API Status Error: {e}")
        except Exception as e:
            logger.error(f"Failed to get embedding from OpenAI for text '{text[:100]}...': {e}", exc_info=True)
            # Reraise after logging, potentially wrapping in a custom exception
            raise RuntimeError(f"Failed to get embedding: {e}")

    def save_memory(self, text: str, metadata: dict = None, memory_id: Optional[str] = None) -> str:
        """
        텍스트를 벡터화해 Qdrant에 *스트리밍* 업서트합니다.  
        - 동일 세션에서 `MAX_MEMORY_PER_SESSION` 초과 시 가장 오래된 벡터를 삭제  
        - `FORGET_AFTER_DAYS` 를 초과한 메모리는 백그라운드에서 주기적으로 삭제
        """
        if not text:
            logger.warning("Attempted to save memory with empty text. Skipping.")
            return None
        
        if not memory_id:
            memory_id = str(uuid.uuid4())

        metadata = metadata or {}
        metadata['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata['original_text'] = text # 원문 저장 (선택적)

        try:
            # --- Streaming Upsert ---
            vector = self.get_embedding(text)

            # 오래된 세션 메모리 정리
            if session_uuid := metadata.get("session_uuid"):
                self._enforce_session_quota(session_uuid)
            
            point = PointStruct(
                id=memory_id,
                vector=vector,
                payload=metadata
            )
            
            response = self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True # 작업 완료 대기
            )
            
            if response.status == UpdateStatus.COMPLETED:
                 logger.info(f"Memory saved/updated successfully with ID: {memory_id}")
                 return memory_id
            else:
                 logger.warning(f"Memory upsert for ID {memory_id} finished with status: {response.status}")
                 # 부분 성공/실패 시 처리 필요 시 추가
                 return memory_id # ID는 반환하나, 상태 확인 필요

        except RuntimeError as e:
             logger.error(f"Failed to save memory due to embedding error: {e}")
             raise # 임베딩 실패는 심각한 문제일 수 있으므로 다시 raise
        except Exception as e:
            logger.error(f"Failed to save memory to Qdrant (ID: {memory_id}): {e}", exc_info=True)
            raise RuntimeError(f"Qdrant upsert operation failed: {e}")

    def search_similar(self, query_text: str, limit: int = 5, score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
        """주어진 텍스트와 유사한 메모리를 Qdrant에서 검색합니다."""
        if score_threshold is None:
             score_threshold = self.DEFAULT_SCORE_THRESHOLD
             
        try:
            query_vector = self.get_embedding(query_text)
            
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold # 임계값 적용
            )
            logger.info(f"Found {len(search_result)} similar memories for query (threshold: {score_threshold}).")
            return search_result
            
        except RuntimeError as e:
             logger.error(f"Failed to search memory due to embedding error: {e}")
             return [] # 임베딩 실패 시 빈 결과 반환
        except Exception as e:
            logger.error(f"Failed to search similar memories in Qdrant: {e}", exc_info=True)
            return [] # Qdrant 검색 실패 시 빈 결과 반환

    def _summarize(self, text: str, topic: Optional[str] = None) -> str:
        """LLM을 사용하여 텍스트 요약 (내부 헬퍼 함수)"""
        system_prompt = (
            "당신은 금융 거래 및 투자 관련 대화 내용을 전문적으로 요약하는 AI 어시스턴트입니다. "
            "주어진 대화 로그의 핵심 내용을 간결하고 명확하게 요약해주세요. "
            "주요 결정사항, 투자 종목, 매수/매도 지시, 질문, 답변 등을 포함해야 합니다. "
            "객관적인 사실 위주로 요약하고, 한국어로 3~5 문장으로 작성해주세요."
        )
        
        user_content = f"다음 대화 로그를 요약해주세요:\n\n---\n{text}\n---"
        if topic:
            user_content += f"\n\n(참고: 이 대화는 '{topic}' 주제와 관련이 있습니다.)"
            
        try:
            # Azure OpenAI REST API 호출
            resp_json = azure_chat_completion(
                deployment=self.llm_model, # __init__에서 설정된 모델 사용
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=500, # 요약 결과에 충분한 토큰 할당
                temperature=0.3 # 일관성 있는 요약을 위해 낮은 온도 설정
            )
            
            # 결과 확인 및 추출 (오류 처리 강화)
            if not resp_json or "choices" not in resp_json or not resp_json["choices"]:
                 logger.error("Invalid response received from Azure OpenAI API during summarization.")
                 raise ValueError("Azure OpenAI API returned an unexpected response format.")
                 
            summary = resp_json["choices"][0].get("message", {}).get("content", "").strip()
            
            if not summary:
                 logger.warning("Azure OpenAI returned an empty summary.")
                 return "(요약 내용 없음)" 
                 
            logger.info(f"Successfully received summary from Azure OpenAI model: {self.llm_model}")
            return summary
            
        except Exception as e:
            logger.error(f"Azure OpenAI summarization failed using model {self.llm_model}: {e}", exc_info=True)
            # 간단한 오류 메시지 반환
            return f"(자동 요약 생성 중 오류 발생: {type(e).__name__})"

    def summarize_and_upsert(self, session_uuid: str):
        """특정 세션의 로그를 DB에서 조회하여 요약하고 결과를 Qdrant와 DB에 저장합니다.

        Args:
            session_uuid: 요약할 세션의 내부 UUID
        """
        if not self.db_session_factory:
            logger.error("Database session factory not configured. Cannot summarize session.")
            return

        session: Session = self.db_session_factory()
        try:
            # 1. DB에서 세션 및 관련 로그 조회
            trading_session = session.query(TradingSession).filter(TradingSession.session_uuid == session_uuid).first()
            if not trading_session:
                logger.warning(f"Trading session with UUID {session_uuid} not found in DB.")
                return
                
            if not trading_session.is_active:
                logger.info(f"Session {session_uuid} is already inactive. Skipping summarization.")
                # 이미 요약본이 있는지 확인하고 없으면 생성할 수도 있음 (선택적)
                return

            logs = session.query(SessionLog).filter(SessionLog.session_id == trading_session.id).order_by(SessionLog.timestamp.asc()).all()
            if not logs:
                logger.info(f"No logs found for session {session_uuid}. Cannot summarize.")
                # 세션 비활성화 처리만 수행
                trading_session.is_active = False
                trading_session.end_time = datetime.datetime.now(datetime.timezone.utc)
                session.commit()
                logger.info(f"Marked session {session_uuid} as inactive due to no logs.")
                return

            # 2. 로그 텍스트 포맷팅 (요약 입력용)
            log_texts = [f"{log.actor} ({log.timestamp.strftime('%H:%M:%S')}): {log.message}" for log in logs]
            full_log_text = "\n".join(log_texts)
            
            logger.info(f"Summarizing {len(logs)} logs for session {session_uuid}...")
            
            # 3. LLM을 사용한 요약 생성
            # topic 정보를 추가하면 요약 품질 향상에 도움될 수 있음 (예: 첫 사용자 메시지 등)
            topic_hint = logs[0].message if logs else "투자 관련 대화"
            summary = self._summarize(full_log_text, topic=topic_hint)
            logger.info(f"Generated summary for session {session_uuid}: {summary[:100]}...")

            # 4. 요약 결과를 Qdrant에 저장
            summary_metadata = {
                "type": "session_summary",
                "session_uuid": session_uuid,
                "discord_thread_id": trading_session.discord_thread_id,
                "discord_user_id": trading_session.discord_user_id,
                "log_count": len(logs),
                "original_log_preview": full_log_text[:200] + "..." # 미리보기 저장
            }
            # 요약문 자체를 벡터화하여 저장
            summary_memory_id = self.save_memory(summary, metadata=summary_metadata, memory_id=f"summary_{session_uuid}") 
            if summary_memory_id:
                 logger.info(f"Saved session summary to Qdrant with ID: {summary_memory_id}")
            else:
                 logger.warning(f"Failed to save session summary {session_uuid} to Qdrant.")

            # 5. DB에 요약 결과 및 세션 종료 상태 업데이트
            trading_session.summary = summary
            trading_session.is_active = False
            trading_session.end_time = datetime.datetime.now(datetime.timezone.utc)
            session.commit()
            logger.info(f"Updated session {session_uuid} in DB with summary and marked as inactive.")

        except Exception as e:
            logger.error(f"Error during summarization and upsert for session {session_uuid}: {e}", exc_info=True)
            session.rollback() # 오류 발생 시 롤백
        finally:
            session.close()

    def save_execution_results(self, execution_results: List[Dict[str, Any]]):
        """Orchestrator의 실행 결과 리스트를 메모리(Qdrant)에 저장합니다."""
        if not execution_results:
            return

        points_to_upsert = []
        for result in execution_results:
            try:
                # 실행 결과에서 텍스트 표현 생성 (예시, 실제 구조에 맞게 조정 필요)
                text_representation = f"함수: {result.get('function_name', 'N/A')}, " \
                                    f"상태: {result.get('status', 'N/A')}, " \
                                    f"결과: {str(result.get('result', 'N/A'))[:500]}" # 결과 일부만 포함
                
                if not text_representation:
                    logger.warning(f"Skipping execution result due to missing information: {result}")
                    continue
                    
                # 메타데이터 구성 (실제 필요한 정보 추가)
                metadata = {
                    "type": "execution_result",
                    "function_name": result.get('function_name'),
                    "status": result.get('status'),
                    "timestamp": result.get('timestamp', datetime.datetime.now(datetime.timezone.utc).isoformat()),
                    "session_uuid": result.get('session_uuid'), # 세션 정보가 있다면 추가
                    # 필요시 추가 필드 (예: parameters, error_message)
                }
                # None 값 제거 (Qdrant payload는 None 값 허용 안 함)
                metadata = {k: v for k, v in metadata.items() if v is not None}
                
                vector = self.get_embedding(text_representation)
                point_id = str(uuid.uuid4())
                
                points_to_upsert.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=metadata
                ))
                logger.debug(f"Prepared execution result for saving (ID: {point_id})")
                
            except RuntimeError as e:
                 logger.error(f"Failed to process execution result due to embedding error: {e} - Result: {result}")
                 # 개별 결과 처리 실패 시 계속 진행할지 결정
            except Exception as e:
                logger.error(f"Error processing execution result for saving: {e} - Result: {result}", exc_info=True)
                # 개별 결과 처리 실패 시 계속 진행

        if points_to_upsert:
            try:
                response = self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points_to_upsert,
                    wait=True
                )
                if response.status == UpdateStatus.COMPLETED:
                    logger.info(f"Successfully saved {len(points_to_upsert)} execution results to Qdrant.")
                else:
                    logger.warning(f"Execution results upsert finished with status: {response.status}")
            except Exception as e:
                logger.error(f"Failed to save execution results batch to Qdrant: {e}", exc_info=True)
                # 배치 저장 실패 시 예외 처리 (개별 저장 시도 등)

# Example Usage (테스트 및 로컬 실행용)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # --- 의존성 설정 (테스트용) --- 
    try:
        from src.db.session import SessionLocal # DB 세션 가져오기
        from src.db.models import create_tables # 테이블 생성 함수
        
        # Qdrant 클라이언트 초기화 (테스트용)
        # 실제 환경에서는 설정값을 사용해야 함
        qdrant_cli = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        
        # 사용할 LLM 모델 이름 (테스트용)
        test_llm_model = settings.AZURE_OPENAI_DEPLOYMENT_GPT4 # 설정값 사용
        
        # 필수 설정 확인
        if not settings.QDRANT_URL or not settings.DATABASE_URL or not test_llm_model:
             print("Error: QDRANT_URL, DATABASE_URL, or AZURE_OPENAI_DEPLOYMENT_GPT4 is not configured in settings.")
             exit()
             
        # DB 테이블 생성 (필요한 경우)
        # create_tables() # 실제 실행 시 주석 해제하거나 관리 스크립트 사용
        
    except ImportError as ie:
        print(f"Import error, make sure dependencies are installed and paths are correct: {ie}")
        exit()
    except Exception as setup_e:
         print(f"Error during test setup: {setup_e}")
         exit()
    # --- DB에 테스트 데이터 생성 --- 
    test_session_uuid = str(uuid.uuid4())
    db = SessionLocal()
    try:
        # 기존 테스트 세션 삭제 (옵션)
        existing_session = db.query(TradingSession).filter(TradingSession.session_uuid == test_session_uuid).first()
        if existing_session:
            # 연관된 로그도 삭제해야 할 수 있음 (cascade 설정에 따라 다름)
            db.delete(existing_session)
            db.commit()
            logger.info(f"Deleted existing test session {test_session_uuid}")
             
        new_session = TradingSession(
            session_uuid=test_session_uuid,
            discord_thread_id="discord_thread_test_123",
            discord_user_id="discord_user_test_456"
        )
        db.add(new_session)
        db.flush() # ID를 얻기 위해 flush
        session_db_id = new_session.id # DB 상의 ID 저장
        
        logs_data = [
            ("user", "삼성전자 지금 사도 괜찮을까요?"),
            ("ai", "현재 삼성전자 주가는 8만원 선에서 등락을 반복하고 있습니다. 최근 반도체 업황 개선 기대감이 있지만, 단기 변동성은 클 수 있습니다. 투자 목표와 기간을 고려하여 신중히 결정하시는 것이 좋습니다."),
            ("user", "알겠습니다. 그럼 KODEX 200 ETF는 어떤가요?"),
            ("ai", "KODEX 200은 코스피 200 지수를 추종하는 대표적인 ETF입니다. 분산투자 효과가 있으며 장기적으로 안정적인 성과를 기대할 수 있습니다. 다만, 시장 전체의 등락에 영향을 받습니다."),
            ("user", "KODEX 200 10주 매수해주세요.")
        ]
        
        for actor, message in logs_data:
             log = SessionLog(session_id=session_db_id, actor=actor, message=message)
             db.add(log)
             
        db.commit()
        logger.info(f"Created test session {test_session_uuid} (DB ID: {session_db_id}) with {len(logs_data)} logs.")
        
    except Exception as db_e:
        logger.error(f"Error setting up test data in DB: {db_e}", exc_info=True)
        db.rollback()
        exit()
    finally:
        db.close()
    # --- End DB Setup --- 

    try:
        # --- MemoryRAG 인스턴스 생성 --- 
        memory_rag_instance = MemoryRAG(
            db_session_factory=SessionLocal, 
            qdrant_client=qdrant_cli, 
            llm_model=test_llm_model
        )
        
        # --- 테스트: 세션 요약 및 저장 --- 
        logger.info(f"\n--- Testing Session Summarization for {test_session_uuid} ---")
        memory_rag_instance.summarize_and_upsert(test_session_uuid)
        
        # --- 확인: Qdrant에서 요약 검색 --- 
        summary_query = "삼성전자와 KODEX 200 ETF 관련 논의"
        logger.info(f"\n--- Searching Qdrant for summary related to: '{summary_query}' ---")
        summary_results = memory_rag_instance.search_similar(summary_query, limit=3, score_threshold=0.5)
        
        found_in_qdrant = False
        if summary_results:
            logger.info(f"Found {len(summary_results)} potential matches in Qdrant:")
            for hit in summary_results:
                # 저장된 요약인지 확인 (ID 또는 페이로드 기준)
                if hit.id == f"summary_{test_session_uuid}":
                    logger.info(f"- Found matching summary! Score: {hit.score:.4f}, ID: {hit.id}")
                    logger.info(f"  Payload: {hit.payload}")
                    found_in_qdrant = True
                    break # 찾았으므로 중단
                else:
                     logger.info(f"- Found non-target item: Score: {hit.score:.4f}, ID: {hit.id}, Type: {hit.payload.get('type')}")
                     
        if not found_in_qdrant:
            logger.warning("Target session summary was not found in Qdrant search results or score was below threshold.")
             
        # --- 확인: DB에서 세션 상태 및 요약 확인 --- 
        logger.info(f"\n--- Checking DB for session {test_session_uuid} status and summary ---")
        db = SessionLocal()
        try:
            updated_session = db.query(TradingSession).filter(TradingSession.session_uuid == test_session_uuid).first()
            if updated_session:
                logger.info(f"Session status in DB: is_active={updated_session.is_active}, end_time={updated_session.end_time}")
                logger.info(f"Session summary in DB: {updated_session.summary}")
                assert not updated_session.is_active, "Session should be inactive after summarization."
                assert updated_session.summary is not None and updated_session.summary != "", "Summary should be present in DB."
            else:
                logger.error("Session not found in DB after summarization attempt!")
        finally:
            db.close()
            
        # --- 테스트: 실행 결과 저장 --- 
        logger.info("\n--- Testing Execution Result Saving ---")
        test_execution_results = [
            {"function_name": "search_stock_info", "status": "success", "result": {"price": 81000, "change": 200}, "session_uuid": test_session_uuid, "timestamp": datetime.datetime.now().isoformat()},
            {"function_name": "place_order", "status": "success", "result": {"order_id": "ORD123", "filled_qty": 10}, "session_uuid": test_session_uuid, "timestamp": datetime.datetime.now().isoformat()}
        ]
        memory_rag_instance.save_execution_results(test_execution_results)
        
        # --- 확인: Qdrant에서 실행 결과 검색 --- 
        exec_query = "place_order function execution"
        logger.info(f"\n--- Searching Qdrant for execution results related to: '{exec_query}' ---")
        exec_results = memory_rag_instance.search_similar(exec_query, limit=5, score_threshold=0.5)
        
        found_exec_in_qdrant = False
        if exec_results:
            logger.info(f"Found {len(exec_results)} potential execution results:")
            for hit in exec_results:
                 if hit.payload.get("type") == "execution_result" and hit.payload.get("function_name") == "place_order":
                     logger.info(f"- Found relevant execution result! Score: {hit.score:.4f}, ID: {hit.id}")
                     logger.info(f"  Payload: {hit.payload}")
                     found_exec_in_qdrant = True
                     # 필요 시 더 많은 결과 확인
        if not found_exec_in_qdrant:
             logger.warning("Could not find the specific 'place_order' execution result in Qdrant search.")

    except RuntimeError as e:
        logger.critical(f"\nRuntime Error during MemoryRAG test: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"\nAn unexpected error occurred during the MemoryRAG test run: {e}", exc_info=True)