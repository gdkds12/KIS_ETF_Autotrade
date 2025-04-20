from src.brokers.kis import KisBroker

class AI:
    def __init__(self, kis_broker=None):
        # KisBroker 인스턴스 주입 가능 (테스트/확장성)
        self.kis_broker = kis_broker or KisBroker(
            app_key=None,  # 실제 환경에서는 settings에서 주입
            app_secret=None,
            base_url=None,
            cano=None,
            acnt_prdt_cd=None,
            virtual_account=True
        )

    def process_query(self, user_query: str):
        """AI가 사용자의 질문을 처리하고, 국내/해외를 판단하여 ETF/주식 시세 조회"""
        # 1. 질문 분석하여 해외 여부 판단
        is_foreign = self.determine_foreign(user_query)
        # 2. 종목 코드 추출 (실제 구현은 심볼 추출 로직 필요)
        symbol = self.extract_symbol_from_query(user_query)
        # 3. KisBroker의 get_quote 호출
        result = self.kis_broker.get_quote(symbol, is_foreign)
        return result

    def determine_foreign(self, query: str) -> bool:
        """질문에 '미국', '해외' 등 해외 관련 키워드가 있으면 True 반환"""
        query = query.lower()
        if '미국' in query or '해외' in query or 'us' in query or 'usa' in query:
            return True
        return False

    def extract_symbol_from_query(self, query: str):
        """질문에서 심볼/종목명 추출 (예시: 심플하게 키워드 매칭)"""
        if 'kodex 200' in query.lower():
            return '069500'  # 실제 KODEX 200 심볼 예시
        elif 'spy' in query.upper():
            return 'SPY'
        # 실제 구현은 정규표현식, 사전, LLM 등 활용 가능
        return query.strip()
