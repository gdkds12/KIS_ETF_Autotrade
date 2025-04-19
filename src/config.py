from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

# .env 파일 로드 (프로젝트 루트에 있다고 가정)
load_dotenv()

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

    # Discord Bot Configuration
    DISCORD_TOKEN: str
    DISCORD_ORDER_CONFIRMATION_CHANNEL_ID: int | None = None # 주문 승인 메시지를 보낼 채널 ID (Optional)

    # Slack Webhook URL (for alerts)
    SLACK_WEBHOOK: str | None = None # Optional

    # Database Configuration
    DB_USER: str = "user"
    DB_PASSWORD: str = "password"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "autotrade"
    DATABASE_URL: str | None = None # Construct if not provided

    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str | None = None # Construct if not provided
    QDRANT_API_KEY: str | None = None # Optional

    # --- LLM Configuration --- 
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None

    # Tier별 모델 이름 정의
    LLM_HIGHEST_TIER_MODEL: str = "gpt-3.5-turbo" # O3 대응
    LLM_MAIN_TIER_MODEL: str = "gpt-4o-mini"       # O4-mini 대응
    LLM_LIGHTWEIGHT_TIER_MODEL: str = "gemini-1.5-flash-latest" # Gemini Flash 대응
    # LLM_MAIN_TIER_HIGH_MODEL: str = "gpt-4o" # O4-mini-high 대응 (필요시 활성화)
    
    # Embedding Model (Example)
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2" # 예시
    VECTOR_DIM: int = 384 # 위 모델 기준 (실제 모델에 맞게 조정 필요)

    # Trading Configuration
    INVESTMENT_AMOUNT: float = 1_000_000 # 총 투자금 (예: 100만원)
    MAX_DAILY_ORDERS: int = 10
    STOP_LOSS_PERCENT: float = 0.10 # 10% 손절 (예시)

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
    print(f"GOOGLE_API_KEY Set: {bool(settings.GOOGLE_API_KEY)}")
    print(f"LLM Highest Tier: {settings.LLM_HIGHEST_TIER_MODEL}")
    print(f"LLM Main Tier: {settings.LLM_MAIN_TIER_MODEL}")
    print(f"LLM Lightweight Tier: {settings.LLM_LIGHTWEIGHT_TIER_MODEL}")
    print(f"EMBEDDING_MODEL_NAME: {settings.EMBEDDING_MODEL_NAME}")
    print(f"VECTOR_DIM: {settings.VECTOR_DIM}")
    print(f"INVESTMENT_AMOUNT: {settings.INVESTMENT_AMOUNT}") 