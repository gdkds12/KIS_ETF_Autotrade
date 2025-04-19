KIS ETF Autotrade — AI 개발 명세서 & 사용자 매뉴얼 (v1.1)

목적 : KIS OpenAPI 로 100 만원 이하 소액 KRX ETF를 자동 매매. 전 과정 AI가 코드·테스트·문서 생성 및 운영.

0. 전체 구조 요약

┌─ Discord Bot (/trade) ─┐
│   └─ discord/bot.py    │
│                        │
│   Session Manager      │
│    ↕ REST/WebSocket    │
│ Executive Orchestrator │
│    └─ src/agents/      │
│    ↕ gRPC (internal)   │
│ Strategy · Risk Guard  │
│    └─ src/strategy/    │
│    ↕ Broker(KIS)       │
│    └─ src/brokers/kis.py │
└──────── Data Lake ─────┘

일일 루틴 08:00 KST (전 거래일 마감 후)

InfoCrawler → 2. Memory Summarize → 3. Market Fetch → 4. Strategy → 5. RiskGuard → 6. Broker Execute → 7. Briefing → 8. Memory Upsert

디스코드 세션 언제든 /trade 호출 → 실시간 LLM 대화 & 주문 확인

1. 파일 구조 & 핵심 기능

autotrade-etf/
├─ README.md                 # 프로젝트 개요
├─ docker-compose.yml        # Infra(Postgres,Qdrant,Grafana)
├─ .env                     # 환경변수 (APP_KEY,SECRET,CANO...)
├─ src/
│   ├─ main.py              # FastAPI 애플리케이션 진입점
│   ├─ config.py            # Pydantic 설정
│   ├─ brokers/
│   │   └─ kis.py           # KIS OpenAPI 래퍼 (토큰·시세·주문)
│   ├─ agents/
│   │   ├─ orchestrator.py  # Dag 관리·에이전트 호출 흐름
│   │   ├─ info_crawler.py  # 시장 이슈 수집·요약
│   │   ├─ memory_rag.py    # Qdrant 벡터 저장·검색
│   │   ├─ strategy.py      # 모멘텀·리밸런스 전략
│   │   ├─ risk_guard.py    # 주문 리스크 필터링
│   │   └─ briefing.py      # Discord 브리핑 생성
│   ├─ discord/
│   │   └─ bot.py           # Slash 명령·세션 관리자
│   ├─ dags/
│   │   └─ daily_cycle.py   # Airflow DAG 정의
│   ├─ db/
│   │   ├─ models.py        # SQLAlchemy ORM 모델
│   │   └─ migrations/      # Alembic 마이그레이션
│   └─ tests/               # pytest 단위·통합 테스트
└─ requirements.lock         # 의존 잠금

각 파일 핵심 기능 요약

main.py: FastAPI 서버 시작, 라우터 등록

config.py: 환경 변수 로드, API 엔드포인트 설정

kis.py: OAuth2 토큰 발급/재발급, 시세조회(quote), 현금주문(order_cash)

orchestrator.py: 일일 사이클 코디네이터, 실패 시 재시도·알람

info_crawler.py: Fnguide·KRX RSS 크롤링, 요약(Flash 모델)

memory_rag.py: 세션 요약 저장 및 유사도 검색 API

strategy.py: 12-1 ROC, ATR 기반 포지션 계산 및 월간 리밸런스

risk_guard.py: 일일 주문 한도, 손절·API 오류 감지

briefing.py: 트레이드 결과 Markdown 임베드 생성

bot.py: /trade 커맨드, Thread 세션 관리, 메시지 프록시

daily_cycle.py: Airflow 스케줄, 태스크 의존성·SLA 설정

models.py: accounts/orders/positions/sessions 메시지 ERD 정의

tests/: VCR 기반 모킹, 커버리지 85% 목표

2. 사용자 매뉴얼

2.1 설치 & 초기 설정

리포지토리 클론

git clone https://.../autotrade-etf.git
cd autotrade-etf

환경 변수 작성 (.env)

APP_KEY=...
APP_SECRET=...
CANO=12345678
ACNT_PRDT=01
BASE_URL=https://openapi.koreainvestment.com:9443
DISCORD_TOKEN=...
SLACK_WEBHOOK=...

Docker 환경 띄우기

docker-compose up -d

의존성 설치 & 마이그레이션

poetry install
alembic upgrade head

2.2 일일 자동매매 실행

Airflow Web UI (http://localhost:8080) 접속 → daily_cycle DAG 활성화 → 스케줄 자동 실행

수동 실행:

airflow dags trigger daily_cycle

2.3 디스코드 실시간 세션

Discord 서버에 Bot 초대 (적절 권한: CREATE_PUBLIC_THREADS,SEND_MESSAGES)

채널에서 /trade 입력 → 질문 프롬프트 모달

Thread 에서 LLM 응답 확인 및 후속 질문

주문 제안 시 ✅ 버튼 클릭 → 실제 주문 실행

30분 idle 후 자동 요약 및 Thread 아카이브

2.4 결과 확인 & 모니터링

Grafana 대시보드: PnL, NAV 변동, 주문 이력

Telegram 알림: NAV < 800k KRW 시 긴급 알람

로그 조회 (Postgres session_logs): thread_id 기반 대화 기록

2.5 문제 해결 가이드

토큰 만료 오류: kis.py 로그 확인, get_token() 재시도

429 Rate Limit: risk_guard 백오프 동작, 재시도 후에도 잦으면 요청 빈도 조정

DAG 실패: Airflow 로그 → 재시도 → discord/#alert 알람

Last updated: 2025‑04‑18