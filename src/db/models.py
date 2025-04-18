# SQLAlchemy ORM 모델 
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func # for server_default=func.now()

from src.config import settings # DB URL 가져오기

Base = declarative_base()

class Account(Base):
    """계좌 정보 모델"""
    __tablename__ = 'accounts'

    id = Column(Integer, primary_key=True)
    account_number = Column(String(20), unique=True, nullable=False, index=True) # 계좌번호 (CANO + ACNT_PRDT)
    broker_name = Column(String(50), default="KIS") # 증권사 이름
    principal = Column(Float, default=0.0) # 초기 투자 원금
    balance = Column(Float, default=0.0) # 현재 현금 잔고 (실시간 갱신 필요)
    total_asset_value = Column(Float, default=0.0) # 총 평가 자산 (현금 + 주식 평가액)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    orders = relationship("Order", back_populates="account")
    positions = relationship("Position", back_populates="account")
    sessions = relationship("TradingSession", back_populates="account") # Discord 세션 연결

    def __repr__(self):
        return f"<Account(account_number='{self.account_number}', balance={self.balance})>"

class Order(Base):
    """주문 정보 모델"""
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False, index=True)
    order_number = Column(String(50), unique=True, nullable=False, index=True) # KIS 주문번호 (ODNO)
    broker_order_id = Column(String(50)) # 혹시 모를 KIS 내부 ID (KRX_FWDG_ORD_ORGNO 등)
    symbol = Column(String(20), nullable=False, index=True) # 종목 코드 (PDNO)
    order_type = Column(String(10), nullable=False) # 주문 유형 (e.g., "지정가", "시장가") KIS: ORD_DVSN
    buy_sell_code = Column(String(10), nullable=False) # 매매 구분 (e.g., "매수", "매도")
    quantity = Column(Integer, nullable=False) # 주문 수량 (ORD_QTY)
    price = Column(Float, nullable=False) # 주문 가격 (ORD_UNPR), 시장가는 0
    status = Column(String(20), default="submitted", nullable=False) # 주문 상태 (e.g., "submitted", "filled", "partial_fill", "cancelled", "failed")
    filled_quantity = Column(Integer, default=0) # 체결 수량
    filled_avg_price = Column(Float, default=0.0) # 체결 평균 가격
    error_message = Column(Text) # 주문 실패 시 메시지 (msg1)
    order_time = Column(DateTime(timezone=True)) # 주문 시간 (ORD_TMD 파싱)
    filled_time = Column(DateTime(timezone=True)) # 최종 체결 시간 (필요시)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    account = relationship("Account", back_populates="orders")

    def __repr__(self):
        return f"<Order(order_number='{self.order_number}', symbol='{self.symbol}', status='{self.status}')>"

class Position(Base):
    """보유 종목 (포지션) 모델"""
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Integer, nullable=False) # 보유 수량
    average_buy_price = Column(Float, nullable=False) # 평균 매수 단가
    current_price = Column(Float) # 현재가 (실시간 갱신 필요)
    current_value = Column(Float) # 평가 금액 (quantity * current_price)
    profit_loss = Column(Float) # 평가 손익
    profit_loss_percent = Column(Float) # 평가 손익률
    # TODO: Add fields from KIS get_balance response if needed (e.g., pchs_amt, evlu_amt)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    account = relationship("Account", back_populates="positions")

    def __repr__(self):
        return f"<Position(symbol='{self.symbol}', quantity={self.quantity}, avg_price={self.average_buy_price})>"

class TradingSession(Base):
    """Discord 트레이딩 세션 로그 모델"""
    __tablename__ = 'trading_sessions'

    id = Column(Integer, primary_key=True)
    session_uuid = Column(String(36), unique=True, nullable=False, index=True) # 내부 세션 ID (LLM 등 연동용)
    discord_thread_id = Column(String(50), unique=True, nullable=False, index=True) # Discord 스레드 ID
    discord_user_id = Column(String(50), nullable=False, index=True) # 세션 시작 유저 ID
    account_id = Column(Integer, ForeignKey('accounts.id'), nullable=True) # 연결된 계좌 (선택적)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True)) # 세션 종료 시간 (아카이브 시)
    is_active = Column(Boolean, default=True)
    summary = Column(Text) # 세션 종료 시 요약 내용

    # Relationship
    account = relationship("Account", back_populates="sessions")
    logs = relationship("SessionLog", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TradingSession(thread_id='{self.discord_thread_id}', user_id='{self.discord_user_id}')>"

class SessionLog(Base):
    """세션 내 메시지 로그 모델"""
    __tablename__ = 'session_logs'

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('trading_sessions.id'), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    actor = Column(String(10), nullable=False) # "user" or "ai"
    message = Column(Text, nullable=False)
    # TODO: Add fields for suggested orders, confirmations if needed
    suggested_order_json = Column(Text) # 제안된 주문 정보 (JSON)
    order_confirmed = Column(Boolean) # 사용자가 주문 실행 확인 여부

    # Relationship
    session = relationship("TradingSession", back_populates="logs")

    def __repr__(self):
        return f"<SessionLog(session_id={self.session_id}, actor='{self.actor}', message='{self.message[:30]}...')>"

# --- DB 연결 및 테이블 생성 (애플리케이션 시작 시 또는 Alembic 사용) --- 
engine = create_engine(settings.DATABASE_URL, echo=False) # echo=True for SQL logging

# Session maker for application use
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """데이터베이스에 모든 테이블 생성 (개발/테스트용, 프로덕션은 Alembic 권장)"""
    logger.info(f"Creating database tables if they don't exist at {settings.DB_HOST}...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Tables checked/created successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # This will attempt to create tables if the script is run directly
    # Ensure your database server is running and accessible
    print(f"Database URL: {settings.DATABASE_URL}")
    create_tables()

    # Example of creating a session
    # db = SessionLocal()
    # try:
    #     # Use the session
    #     # new_account = Account(account_number="12345678-01")
    #     # db.add(new_account)
    #     # db.commit()
    #     print("Database session ready.")
    # finally:
    #     db.close() 