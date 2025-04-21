from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv
from typing import Optional

# .env 파일 로드 (프로젝트 루트에 있다고 가정)
load_dotenv(override=True)

class Settings(BaseSettings):
    # KIS API Credentials
    APP_KEY: str
    APP_SECRET: str

    # KIS Account Info
    CANO: str # 계좌번호 앞 8자리
    ACNT_PRDT: str = "01" # 계좌상품코드

    # KIS API Base URL
    BASE_URL: str = "https://openapi.koreainvestment.com:9443"
    # KIS 모의투자 도메인 (필요시 사용)
    KIS_VIRTUAL_URL: str = "https://openapivts.koreainvestment.com:29443"
    # 모의투자 계좌 사용 여부 (환경 변수 KIS_VIRTUAL_ACCOUNT=true/false 로 설정 가능)
    KIS_VIRTUAL_ACCOUNT: bool = True

    # Bing Search API Key (Optional when using Azure Foundry for web search)
    BING_API_KEY: str | None = None

    # Discord Bot Configuration
    DISCORD_TOKEN: str
    DISCORD_ORDER_CONFIRMATION_CHANNEL_ID: int | None = None # 주문 승인 메시지를 보낼 채널 ID (Optional)

    # Slack Webhook URL (for alerts)
    SLACK_WEBHOOK: str | None = None # Optional

    # Google Custom Search API
    GOOGLE_API_KEY: str
    GOOGLE_CX: str  # Custom Search Engine ID

    # Database Configuration
    DB_USER: str = "user"
    DB_PASSWORD: str = "password"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "autotrade"
    DATABASE_URL: str | None = None # Construct if not provided
    DB_TYPE: str = "postgresql"

    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "kis_etf_memory"

    # LLM Configuration
    OPENAI_API_KEY: str
    FINNHUB_API_KEY: str
    SERPAPI_API_KEY: str | None = None

    # (Deprecated) Azure AI Foundry 설정 — ARM 환경에서는 사용하지 않음
    PROJECT_CONNECTION_STRING: str | None = None
    BING_CONNECTION_NAME: str | None = None

    # Azure OpenAI 설정
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_DEPLOYMENT_GPT4: str
    AZURE_OPENAI_DEPLOYMENT_GPT35: str

    # Tier별 모델 이름 정의 (사용 목적에 맞게 조정)
    # Using gpt-4o-mini as a placeholder for the non-existent gpt-4.1-nano
    LLM_MAIN_TIER_MODEL: str = "gpt-4.1-nano-2025-04-14"       # Main reasoning model (Orchestrator)
    LLM_LIGHTWEIGHT_TIER_MODEL: str = "gpt-4.1-nano-2025-04-14" # Lightweight tasks (Summarization, etc.)
    # LLM_HIGHEST_TIER_MODEL: str = "gpt-4o" # Alternative if needed
    
    # Embedding Model (Example)
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-large" # <<< 변경: OpenAI 최신 대형 모델
    VECTOR_DIM: int = 12288 # text-embedding-3-large의 기본 임베딩 크기는 3072이나, 실제 반환 크기가 다를 수 있음 (OpenAI 문서 참조)

    # Trading Configuration
    INVESTMENT_AMOUNT: float = 1_000_000 # 총 투자 참고 금액 (LLM Context 용)
    MAX_DAILY_ORDERS: int = 20      # RiskGuard 일일 주문 검증 횟수 한도
    STOP_LOSS_PERCENT: float = 0.07 # 7% 고정 비율 손절 (ATR 손절 미사용 시)
    TARGET_SYMBOLS: list[str] = ["069500", "229200", "114800"] # 기본 분석/거래 대상 ETF
    ORDER_INTERVAL_SECONDS: float = 0.2 # KIS 주문 API 호출 간 최소 간격 (초)
    CYCLE_INTERVAL_MINUTES: int = 15   # Orchestrator 자동 실행 간격 (분) - 실제 스케줄링은 외부에서 관리될 수 있음
    API_ERROR_BACKOFF_SECONDS: int = 5   # KIS API 오류 발생 시 재시도 전 대기 시간 (초)
    # 해외 주식(ETF 포함) 거래 지원 여부
    SUPPORT_OVERSEAS_TRADING: bool = False # True로 설정 시 미국 등 해외 ETF 거래 가능

    # Model configuration
    model_config = SettingsConfigDict(
        env_file='.env',         # .env 파일 경로 명시
        env_file_encoding='utf-8', # 인코딩 설정
        extra='ignore'           # .env에 없는 필드는 무시
    )

    # Calculated fields
    def __init__(self, **values):
        super().__init__(**values)
        if not self.DATABASE_URL:
            # Ensure password is handled correctly if it contains special characters
            # from urllib.parse import quote_plus
            # db_password_escaped = quote_plus(self.DB_PASSWORD)
            self.DATABASE_URL = f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        if not self.QDRANT_URL:
            self.QDRANT_URL = f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

# 설정 객체 생성 (애플리케이션 전역에서 사용)
settings = Settings()

# Example usage:
if __name__ == "__main__":
    print("--- Loaded Settings ---")
    print(f"APP_KEY: {settings.APP_KEY[:4]}...{settings.APP_KEY[-4:]}")
    print(f"APP_SECRET: ****")
    print(f"CANO: {settings.CANO}")
    print(f"ACNT_PRDT: {settings.ACNT_PRDT}")
    print(f"BASE_URL (Real): {settings.BASE_URL}")
    print(f"BASE_URL (Virtual): {settings.KIS_VIRTUAL_URL}")
    print(f"Using Virtual Account: {settings.KIS_VIRTUAL_ACCOUNT}")
    print(f"DISCORD_TOKEN: {settings.DISCORD_TOKEN[:6]}..." if settings.DISCORD_TOKEN else "Not Set")
    print(f"DISCORD_ORDER_CONFIRMATION_CHANNEL_ID: {settings.DISCORD_ORDER_CONFIRMATION_CHANNEL_ID}" if settings.DISCORD_ORDER_CONFIRMATION_CHANNEL_ID else "Not Set")
    print(f"SLACK_WEBHOOK: {settings.SLACK_WEBHOOK[:20]}..." if settings.SLACK_WEBHOOK else "Not Set")
    print(f"DATABASE_URL: {settings.DATABASE_URL}")
    print(f"QDRANT_URL: {settings.QDRANT_URL}")
    print(f"OPENAI_API_KEY Set: {bool(settings.OPENAI_API_KEY)}")
    # print(f"GOOGLE_API_KEY Set: {bool(settings.GOOGLE_API_KEY)}") # Remove print for Google key
    print(f"FINNHUB_API_KEY Set: {bool(settings.FINNHUB_API_KEY)}")
    print(f"SERPAPI_API_KEY Set: {bool(settings.SERPAPI_API_KEY)}")
    # Print updated model names
    print(f"LLM Main Tier (Reasoning): {settings.LLM_MAIN_TIER_MODEL}")
    print(f"LLM Lightweight Tier (Summarization, etc.): {settings.LLM_LIGHTWEIGHT_TIER_MODEL}")
    # ... (print rest of settings) ... 