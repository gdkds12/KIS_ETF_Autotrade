from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.latest_only import LatestOnlyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.state import State
import logging

# --- 로깅 설정 --- 
logger = logging.getLogger(__name__)

# --- Orchestrator 호출 함수 (Placeholder) ---
# 실제로는 Orchestrator 클래스를 임포트하고 메서드를 호출해야 합니다.
# from src.agents.orchestrator import Orchestrator
# from src.config import settings
# from src.db import SessionLocal # Assuming a session factory
# from src.brokers.kis import KisBroker
# from qdrant_client import QdrantClient

def initialize_dependencies():
    """Orchestrator 및 관련 의존성 초기화 (실제 구현 필요)"""
    logger.info("Initializing dependencies (Broker, DB, Qdrant, Orchestrator)...")
    # broker = KisBroker(...)
    # db_session = SessionLocal()
    # qdrant_client = QdrantClient(...)
    # orchestrator = Orchestrator(broker, db_session, qdrant_client)
    # return orchestrator, db_session
    class MockOrchestrator:
        def run_info_crawler(self): logger.info("Running Mock Info Crawler..."); return "Mock Summary"
        def summarize_and_upsert_memory(self): logger.info("Running Mock Memory Summarize/Upsert...")
        def fetch_market_data(self): logger.info("Running Mock Market Fetch..."); return {"069500": {"stck_prpr": "35000"}}
        def run_strategy(self, market_data, market_summary): logger.info("Running Mock Strategy..."); return [{"symbol": "069500", "action": "buy"}]
        def run_risk_guard(self, trading_signals): logger.info("Running Mock Risk Guard..."); return trading_signals
        def execute_orders(self, validated_orders): logger.info("Running Mock Broker Execute..."); return [{"rt_cd": "0", "msg1": "Mock order success", "order": o} for o in validated_orders]
        def generate_briefing(self, order_results): logger.info("Running Mock Briefing..."); return "Mock Briefing Report"
        def upsert_trade_results(self, order_results): logger.info("Running Mock Memory Upsert (Trade Results)...")
        def send_alert(self, message): logger.warning(f"ALERT: {message}")
        def close_connections(self): logger.info("Closing connections (DB, Broker)...")
        
    return MockOrchestrator(), None # Return mock and None for db_session

def cleanup_dependencies(db_session):
    """사용한 DB 세션 등 리소스 정리"""
    logger.info("Cleaning up resources...")
    if db_session:
        # db_session.close()
        logger.info("DB session closed.")
    # orchestrator.close_connections() # Orchestrator에 연결 종료 메서드 추가 고려


# --- Airflow Task Functions --- 

def run_step(step_name: str, orchestrator_method_name: str, orchestrator_kwargs=None, **context):
    """Orchestrator의 특정 단계를 실행하는 공통 함수"""
    logger.info(f"--- Starting Task: {step_name} --- ")
    orchestrator, db_session = None, None
    try:
        orchestrator, db_session = initialize_dependencies()
        method_to_call = getattr(orchestrator, orchestrator_method_name)
        
        # 이전 태스크 결과 가져오기 (XCom 사용)
        task_instance = context['ti']
        required_args = {}
        if orchestrator_method_name == 'run_strategy':
            required_args['market_data'] = task_instance.xcom_pull(task_ids='fetch_market_data', key='return_value')
            required_args['market_summary'] = task_instance.xcom_pull(task_ids='run_info_crawler', key='return_value')
        elif orchestrator_method_name == 'run_risk_guard':
            required_args['trading_signals'] = task_instance.xcom_pull(task_ids='run_strategy', key='return_value')
        elif orchestrator_method_name == 'execute_orders':
            required_args['validated_orders'] = task_instance.xcom_pull(task_ids='run_risk_guard', key='return_value')
        elif orchestrator_method_name == 'generate_briefing' or orchestrator_method_name == 'upsert_trade_results':
             required_args['order_results'] = task_instance.xcom_pull(task_ids='execute_orders', key='return_value')
        
        logger.info(f"Calling orchestrator.{orchestrator_method_name} with args: {required_args.keys()}")
        result = method_to_call(**required_args)
        
        logger.info(f"--- Finished Task: {step_name} --- ")
        # 다음 태스크에서 사용할 결과가 있다면 XCom에 push
        if result is not None:
            return result # Airflow가 자동으로 XCom에 'return_value' 키로 저장
            
    except Exception as e:
        logger.error(f"Error during task {step_name}: {e}", exc_info=True)
        if orchestrator:
            orchestrator.send_alert(f"Airflow Task '{step_name}' Failed: {e}") # 실패 알림
        raise # 태스크 실패 처리
    finally:
        cleanup_dependencies(db_session)


# --- DAG Definition --- 
with DAG(
    dag_id="daily_etf_autotrade_cycle",
    start_date=pendulum.datetime(2024, 7, 1, tz="Asia/Seoul"), # 시작 날짜 (과거로 설정)
    schedule="0 8 * * 1-5", # 매주 월~금요일 08:00 KST 실행
    catchup=False, # 과거 미실행분을 한꺼번에 실행하지 않음
    tags=["etf", "autotrade", "kis"],
    default_args={
        "owner": "airflow",
        "retries": 1, # 실패 시 1번 재시도
        "retry_delay": timedelta(minutes=1),
        "execution_timeout": timedelta(hours=1), # 태스크 최대 실행 시간
        # 'on_failure_callback': notify_failure, # 실패 시 콜백 함수 지정 가능
        # 'sla': timedelta(hours=2), # DAG 실행 완료 목표 시간
    },
    description="KIS OpenAPI를 이용한 일일 ETF 자동매매 사이클",
) as dag:

    # 시작 전 최신 실행인지 확인 (선택 사항)
    latest_only = LatestOnlyOperator(task_id="latest_only")

    # 1. Info Crawler
    task_info_crawler = PythonOperator(
        task_id="run_info_crawler",
        python_callable=run_step,
        op_kwargs={'step_name': 'Info Crawler', 'orchestrator_method_name': 'run_info_crawler'},
    )

    # 2. Memory Summarize & Upsert (Info Crawler와 병렬 또는 순차 실행 가능)
    task_memory_summarize = PythonOperator(
        task_id="summarize_and_upsert_memory",
        python_callable=run_step,
        op_kwargs={'step_name': 'Memory Summarize/Upsert', 'orchestrator_method_name': 'summarize_and_upsert_memory'},
    )

    # 3. Market Data Fetch
    task_market_fetch = PythonOperator(
        task_id="fetch_market_data",
        python_callable=run_step,
        op_kwargs={'step_name': 'Market Fetch', 'orchestrator_method_name': 'fetch_market_data'},
    )

    # 4. Strategy Execution
    task_strategy = PythonOperator(
        task_id="run_strategy",
        python_callable=run_step,
        op_kwargs={'step_name': 'Strategy', 'orchestrator_method_name': 'run_strategy'},
    )

    # 5. Risk Guard Validation
    task_risk_guard = PythonOperator(
        task_id="run_risk_guard",
        python_callable=run_step,
        op_kwargs={'step_name': 'Risk Guard', 'orchestrator_method_name': 'run_risk_guard'},
    )

    # 6. Broker Execute Orders
    task_broker_execute = PythonOperator(
        task_id="execute_orders",
        python_callable=run_step,
        op_kwargs={'step_name': 'Broker Execute', 'orchestrator_method_name': 'execute_orders'},
    )

    # 7. Generate Briefing
    task_briefing = PythonOperator(
        task_id="generate_briefing",
        python_callable=run_step,
        op_kwargs={'step_name': 'Briefing', 'orchestrator_method_name': 'generate_briefing'},
        trigger_rule=TriggerRule.ALL_DONE, # 주문 실행 성공/실패 여부와 관계 없이 실행
    )

    # 8. Memory Upsert (Trade Results)
    task_memory_upsert_trades = PythonOperator(
        task_id="upsert_trade_results",
        python_callable=run_step,
        op_kwargs={'step_name': 'Memory Upsert (Trades)', 'orchestrator_method_name': 'upsert_trade_results'},
        trigger_rule=TriggerRule.ALL_DONE, # 주문 실행 성공/실패 여부와 관계 없이 실행
    )

    # --- 태스크 의존성 정의 --- 
    latest_only >> [task_info_crawler, task_memory_summarize, task_market_fetch]
    
    task_info_crawler >> task_strategy
    task_market_fetch >> task_strategy
    
    task_strategy >> task_risk_guard
    task_risk_guard >> task_broker_execute
    
    task_broker_execute >> task_briefing
    task_broker_execute >> task_memory_upsert_trades

    # 리밸런싱 DAG 트리거 (선택적, 매월 첫 거래일에만 실행되도록 조건 설정 필요)
    # def check_if_rebalance_day(**context):
    #     # TODO: Implement logic to check if today is the rebalance day (e.g., first trading day of month)
    #     return True # or False
    
    # trigger_rebalance_dag = TriggerDagRunOperator(
    #     task_id="trigger_rebalance_dag",
    #     trigger_dag_id="monthly_rebalance_dag", # 별도 리밸런싱 DAG ID
    #     # python_callable=check_if_rebalance_day, # 조건부 실행
    #     conf={"logical_date": "{{ ds }}"},
    #     trigger_rule=TriggerRule.ALL_SUCCESS, # 모든 이전 단계 성공 시
    # )
    # task_broker_execute >> trigger_rebalance_dag # 예시 연결