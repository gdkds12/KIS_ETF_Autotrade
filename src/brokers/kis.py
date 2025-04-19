# KIS OpenAPI 래퍼 (토큰·시세·주문) 
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import requests
import json
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)

# Placeholder for startup/shutdown logic (e.g., DB connection, background tasks)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize broker
    broker = KisBroker(
        app_key=app.state.app_key,
        app_secret=app.state.app_secret,
        base_url=app.state.base_url,
        cano=app.state.cano,
        acnt_prdt_cd=app.state.acnt_prdt_cd
    )
    yield
    # Clean up broker
    broker.close()

class KisBrokerError(Exception):
    """Custom exception for KIS Broker errors."""
    def __init__(self, message, response_data=None):
        super().__init__(message)
        self.response_data = response_data

class KisBroker:
    # --- TR ID 매핑 테이블 (실전용) ---
    TR_IDS = {
        "get_balance":       "TTTC8434R",
        "get_positions":     "TTTC8434R",
        "get_quote":         "FHKST01010100",
        "order_cash_buy":    "TTTC0801U", # Diff는 매수/매도 순서가 반대였으나, 코드 로직상 buy=02가 맞음
        "order_cash_sell":   "TTTC0802U",
        "get_historical_data": "FHKST03010100" # 추가 (get_historical_data 용)
    }

    def __init__(self, app_key: str, app_secret: str, base_url: str, cano: str, acnt_prdt_cd: str, virtual_account: bool = True):
        """KIS Broker 초기화

        Args:
            app_key: KIS 발급 앱 키
            app_secret: KIS 발급 앱 시크릿
            base_url: KIS API 엔드포인트 (실전/모의 투자 구분)
            cano: 계좌번호 앞 8자리
            acnt_prdt_cd: 계좌상품코드 (01)
            virtual_account: 모의투자 계좌 사용 여부 (True: 모의투자, False: 실전투자)
        """
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = base_url
        self.cano = cano
        self.acnt_prdt_cd = acnt_prdt_cd
        self.virtual_account = virtual_account
        self.access_token = None
        self.token_expires_at = None
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json; charset=utf-8"})
        self._init_headers()
        logger.info(f"KisBroker initialized for {'Virtual' if virtual_account else 'Real'} Trading (URL: {self.base_url}).")

    def _init_headers(self):
        """API 호출에 필요한 기본 헤더 설정 (토큰 제외)"""
        self.session.headers["appkey"] = self.app_key
        self.session.headers["appsecret"] = self.app_secret

    def _update_token_header(self):
        """세션 헤더에 현재 액세스 토큰 추가"""
        if self.access_token:
            self.session.headers["authorization"] = f"Bearer {self.access_token}"
        else:
            self.session.headers.pop("authorization", None)

    def _is_token_valid(self) -> bool:
        """현재 토큰이 유효한지 확인 (만료 시간 5분 전 갱신)"""
        if self.access_token and self.token_expires_at:
            return datetime.now() < self.token_expires_at - timedelta(minutes=5)
        return False

    def _compute_tr_id(self, key: str) -> str:
        """TR_IDS에서 TR ID를 조회하고, 모의투자이면 T->V 치환."""
        base = self.TR_IDS.get(key)
        if not base:
            raise ValueError(f"Unknown TR key: {key}")
        if self.virtual_account and base.startswith("T"):
            # 모의투자용 TR ID 첫 글자 T -> V
            return "V" + base[1:]
        return base

    def get_token(self) -> dict:
        """KIS API 액세스 토큰을 발급/재발급 받습니다."""
        token_url = f"{self.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        try:
            logger.info("Requesting new KIS access token...")
            response = self.session.post(token_url, json=payload, timeout=10)
            response.raise_for_status()
            token_data = response.json()
            if "access_token" in token_data:
                self.access_token = token_data["access_token"]
                expires_in = int(token_data.get("expires_in", 86400))
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                self._update_token_header()
                logger.info(f"New KIS access token obtained. Expires at: {self.token_expires_at}")
                return token_data
            else:
                logger.error(f"Failed to retrieve access token: {token_data}")
                raise KisBrokerError("Access token not found in response", response_data=token_data)
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error while getting token: {e}", exc_info=True)
            raise KisBrokerError(f"HTTP error during token request: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error while getting token: {e}", exc_info=True)
            raise KisBrokerError(f"Unexpected error during token request: {e}") from e

    def check_token(self) -> bool:
        """
        현재 토큰이 유효한지 확인하고,
        만약 만료되었으면 새 토큰을 발급받아 갱신합니다.
        성공하면 True, 실패하면 False를 반환합니다.
        """
        if self._is_token_valid():
            return True
        try:
            self.get_token()
            return True
        except KisBrokerError:
            return False

    def _request(self, method: str, path: str, tr_id: str, params: dict = None, data: dict = None) -> dict:
        """KIS API 요청을 처리하는 내부 메서드"""
        if not self._is_token_valid():
            logger.info("Token expired or invalid, requesting new token.")
            self.get_token()

        url = f"{self.base_url}{path}"
        headers = self.session.headers.copy()
        headers["tr_id"] = tr_id
        
        # Set tr_cont based on virtual_account (needed for real accounts)
        if not self.virtual_account:
             headers["tr_cont"] = "" # 실전투자 연속거래
        else:
            headers.pop("tr_cont", None) # 모의투자는 제거

        try:
            req_params = {'url': url, 'headers': headers, 'timeout': 15} # 기본 타임아웃 설정
            if method.upper() == 'GET':
                req_params['params'] = params
                response = self.session.get(**req_params)
            elif method.upper() == 'POST':
                req_params['json'] = data
                response = self.session.post(**req_params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            logger.debug(f"KIS API Request: {method} {url} TR:{headers['tr_id']} Params:{params} Data:{data}")
            logger.debug(f"KIS API Response Status: {response.status_code}")
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"KIS API Response Data: {response_data}")

            if response_data.get("rt_cd") != "0":
                error_msg = f"KIS API Error (rt_cd={response_data.get('rt_cd', 'N/A')} msg_cd={response_data.get('msg_cd', 'N/A')} TR:{headers['tr_id']}): {response_data.get('msg1', 'Unknown error')}"
                logger.error(error_msg)
                raise KisBrokerError(error_msg, response_data=response_data)

            return response_data

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error during KIS API request ({path}, TR:{headers['tr_id']}): {e}", exc_info=True)
            raise KisBrokerError(f"HTTP error during KIS API request: {e}") from e
        except KisBrokerError as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during KIS API request ({path}, TR:{headers['tr_id']}): {e}", exc_info=True)
            raise KisBrokerError(f"Unexpected error during KIS API request: {e}") from e

    def get_quote(self, symbol: str) -> dict:
        """주식 현재가 시세(체결)를 조회합니다."""
        path = "/uapi/domestic-stock/v1/quotations/inquire-price"
        tr_id = self._compute_tr_id("get_quote") # Use computed TR ID
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol}
        try:
            response_data = self._request("GET", path, tr_id, params=params)
            return response_data.get("output", {})
        except KisBrokerError as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise e

    def order_cash(self, symbol: str, quantity: int, price: int, order_type: str, buy_sell_code: str) -> dict:
        """현금 매수/매도 주문을 실행합니다."""
        path = "/uapi/domestic-stock/v1/trading/order-cash"
        # 매수/매도에 따른 TR key 사용 (02: Buy, 01: Sell)
        key = "order_cash_buy" if buy_sell_code == "02" else "order_cash_sell"
        tr_id = self._compute_tr_id(key) # Use computed TR ID
        payload = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": symbol,
            "ORD_DVSN": order_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
            "ORD_SVR_DVSN_CD": "0"
        }
        try:
            action = "Buy" if buy_sell_code == "02" else "Sell"
            order_desc = f"{action} {quantity} shares of {symbol} at {price if order_type == '00' else 'Market'} price"
            logger.info(f"Executing order: {order_desc}")
            response_data = self._request("POST", path, tr_id, data=payload)
            logger.info(f"Order successful: {response_data.get('msg1')}")
            return response_data.get("output", {})
        except KisBrokerError as e:
            logger.error(f"Failed to execute order ({order_desc}): {e}")
            raise e

    def get_balance(self) -> dict:
        """계좌 잔고 현황을 조회합니다 (현금, 총자산, 총손익 등)."""
        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        tr_id = self._compute_tr_id("get_balance") # Use computed TR ID
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",    # 시간외단일가 여부 (올바른 필드명)
            "OFL_YN": "",           # 공란
            "INQR_DVSN": "01",      # 조회구분 (01: 대출일별, 02: 종목별)
            "UNPR_DVSN": "01",      # 단가구분
            "FUND_STTL_ICLD_YN": "N", # 펀드결제분 포함 여부
            "FNCG_AMT_AUTO_RDPT_YN": "N", # 융자금액 자동상환 여부
            "PRCS_DVSN": "00",      # 처리구분 (00: 전일매매포함)
            "CTX_AREA_FK100": "",   # 연속조회검색조건
            "CTX_AREA_NK100": ""    # 연속조회키
        }
        try:
            logger.info("Requesting account balance...")
            response_data = self._request("GET", path, tr_id, params=params)
            # KIS 잔고 응답은 output1 (주식 잔고 리스트), output2 (펀드 잔고 또는 기타 잔고)
            output1 = response_data.get("output1", [])
            raw_output2 = response_data.get("output2", {})
            # output2가 리스트로 올 경우 첫 번째 요소 사용
            if isinstance(raw_output2, list):
                output2 = raw_output2[0] if raw_output2 else {}
            else:
                output2 = raw_output2 or {}
            
            # 정보 가공 (KIS 응답 구조에 따라 크게 달라짐)
            balance_info = {
                'account_number': f"{self.cano}-{self.acnt_prdt_cd}",
                'available_cash': float(output2.get('dnca_tot_amt', 0)),      # 예수금 총금액
                'total_asset_value': float(output2.get('tot_evlu_amt', 0)),   # 총평가금액
                'total_purchase_amount': float(output2.get('tot_pchs_amt', 0)),# 총매입금액
                'total_pnl': float(output2.get('tot_prfi_amt', 0)),          # 총손익금액 (필드명 확인 필요)
                'total_pnl_percent': float(output2.get('tot_prfi_rt', 0)),    # 총손익률
                # ... 기타 필요한 정보 ...
            }
            logger.info(f"Account balance retrieved: Cash={balance_info['available_cash']}, TotalValue={balance_info['total_asset_value']}")
            return balance_info
        except KisBrokerError as e:
            logger.error(f"Failed to get account balance: {e}")
            raise e
        except Exception as e:
             logger.error(f"Error processing balance data: {e}", exc_info=True)
             raise KisBrokerError(f"Error processing balance data: {e}") from e

    def get_positions(self) -> list[dict]:
        """계좌의 보유 종목 목록을 조회합니다."""
        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        tr_id = self._compute_tr_id("get_positions") # Use computed TR ID
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02", # 조회구분 (02: 종목별)
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "" 
        }
        try:
            logger.info("Requesting account positions...")
            all_positions = []
            while True: # KIS API 페이징 처리 (연속조회키 사용)
                response_data = self._request("GET", path, tr_id, params=params)
                output1 = response_data.get("output1", []) # output1에 주식 잔고 목록이 담겨있다고 가정
                all_positions.extend(output1)
                
                # 연속조회키 (CTX_AREA_NK100) 확인
                ctx_area_nk100 = response_data.get("ctx_area_nk100", "").strip()
                if ctx_area_nk100: # 다음 페이지가 있다면
                    logger.debug(f"Fetching next page of positions using ctx_area_nk100: {ctx_area_nk100}")
                    params["CTX_AREA_FK100"] = params["CTX_AREA_NK100"] # 이전 키 저장 (필요시)
                    params["CTX_AREA_NK100"] = ctx_area_nk100
                    time.sleep(0.1) # 페이지 요청 간 약간의 딜레이
                else:
                    break # 마지막 페이지

            # KIS 응답 필드명에 맞게 데이터 가공
            positions = []
            for item in all_positions:
                position = {
                    'symbol': item.get('pdno'), # 종목코드
                    'name': item.get('prdt_name'), # 종목명
                    'quantity': int(item.get('hldg_qty', 0)), # 보유수량
                    'average_buy_price': float(item.get('pchs_avg_pric', 0)), # 매입평균가격
                    'purchase_amount': float(item.get('pchs_amt', 0)), # 매입금액
                    'current_price': float(item.get('prpr', 0)), # 현재가
                    'evaluation_amount': float(item.get('evlu_amt', 0)), # 평가금액
                    'profit_loss': float(item.get('evlu_pfls_amt', 0)), # 평가손익금액
                    'profit_loss_percent': float(item.get('evlu_pfls_rt', 0)), # 평가손익률
                }
                if position['quantity'] > 0: # 보유 수량이 0 이상인 것만 추가
                     positions.append(position)
                     
            logger.info(f"Retrieved {len(positions)} positions.")
            return positions
        except KisBrokerError as e:
            logger.error(f"Failed to get account positions: {e}")
            raise e
        except Exception as e:
             logger.error(f"Error processing position data: {e}", exc_info=True)
             raise KisBrokerError(f"Error processing position data: {e}") from e

    def get_historical_data(self, symbol: str, timeframe: str = 'D', 
                            start_date: str | None = None, end_date: str | None = None, 
                            period: int = 100, adjust_price: bool = True) -> pd.DataFrame:
        """기간별(일/주/월) 주가 데이터를 조회합니다.

        Args:
            symbol: 종목 코드
            timeframe: 기간 구분 ('D': 일, 'W': 주, 'M': 월)
            start_date: 조회 시작일 (YYYYMMDD, timeframe이 'W'/'M'일 때 우선)
            end_date: 조회 종료일 (YYYYMMDD, 오늘까지)
            period: 조회 기간 (timeframe이 'D'일 때 사용, 최대 100일? KIS 확인 필요)
            adjust_price: 수정주가 반영 여부 (True/False)

        Returns:
            Pandas DataFrame (index=datetime, columns=[open, high, low, close, volume, ...])
        """
        path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        tr_id = self._compute_tr_id("get_historical_data") # Use computed TR ID

        if timeframe not in ['D', 'W', 'M']:
            raise ValueError("Invalid timeframe. Use 'D', 'W', or 'M'.")
        
        today = datetime.now().strftime('%Y%m%d')
        if end_date is None:
             end_date = today
             
        # 시작일 자동 계산 (일봉이고 start_date 없으면 period 기준)
        if timeframe == 'D' and start_date is None:
            start_dt = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=period + 20) # 여유분 추가
            start_date = start_dt.strftime('%Y%m%d')
        elif start_date is None: # 주봉/월봉인데 시작일 없으면 기본값 (예: 1년 전)
             start_dt = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)
             start_date = start_dt.strftime('%Y%m%d')

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": end_date,
            "FID_PERIOD_DIV_CODE": timeframe,
            "FID_ORG_ADJ_PRC": "0" if adjust_price else "1" # 0: 수정주가반영, 1: 미반영
        }

        try:
            logger.info(f"Requesting historical data for {symbol} ({timeframe}, {start_date}~{end_date})...")
            response_data = self._request("GET", path, tr_id, params=params)
            output = response_data.get("output2", []) # output2에 시세 목록 가정
            
            if not output:
                 logger.warning(f"No historical data found for {symbol} in the specified period.")
                 return pd.DataFrame()
                 
            # Pandas DataFrame으로 변환
            # KIS 응답 필드명 확인 필요 (stck_bsop_date, stck_oprc, stck_hgpr, stck_lwpr, stck_clpr, acml_vol 등)
            df = pd.DataFrame(output)
            # 데이터 타입 변환 및 컬럼명 변경
            df = df.rename(columns={
                'stck_bsop_date': 'date', 
                'stck_oprc': 'open', 
                'stck_hgpr': 'high', 
                'stck_lwpr': 'low',
                'stck_clpr': 'close',
                'acml_vol': 'volume'
                # 'acml_tr_pbmn': 'amount' # 거래대금
            })
            # 필요한 컬럼만 선택 (순서 조정 포함)
            cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in cols if col in df.columns]]
            
            # 데이터 타입 변환
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                     df[col] = pd.to_numeric(df[col], errors='coerce')
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            
            df = df.set_index('date')
            df = df.sort_index() # 날짜 오름차순 정렬
            
            logger.info(f"Retrieved {len(df)} historical data points for {symbol}.")
            # TODO: KIS API가 한번에 모든 데이터를 주지 않을 경우, 페이징 처리 로직 추가 필요
            return df

        except KisBrokerError as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise e
        except Exception as e:
             logger.error(f"Error processing historical data for {symbol}: {e}", exc_info=True)
             raise KisBrokerError(f"Error processing historical data: {e}") from e

    def close(self):
        """세션 종료"""
        self.session.close()
        logger.info("KisBroker session closed.")

# 사용 예시 업데이트
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    from src.config import settings # 설정 로드

    logging.basicConfig(level=logging.INFO) # INFO 레벨 로깅
    # logging.getLogger('src.brokers.kis').setLevel(logging.DEBUG) # KIS 상세 로깅 필요시

    if not all([settings.APP_KEY, settings.APP_SECRET, settings.CANO]):
        print("Please set APP_KEY, APP_SECRET, CANO in your .env file or environment variables.")
    else:
        is_virtual = settings.KIS_VIRTUAL_ACCOUNT

        broker = KisBroker(
            app_key=settings.APP_KEY,
            app_secret=settings.APP_SECRET,
            base_url=settings.KIS_VIRTUAL_URL if is_virtual else settings.BASE_URL,
            cano=settings.CANO,
            acnt_prdt_cd=settings.ACNT_PRDT,
            virtual_account=is_virtual
        )

        try:
            # 토큰 발급은 내부적으로 처리되므로 명시적 호출 불필요
            
            # --- 잔고 조회 테스트 --- 
            print("\n--- Account Balance --- ")
            try:
                balance = broker.get_balance()
                print(json.dumps(balance, indent=2, ensure_ascii=False))
            except KisBrokerError as e:
                 print(f"Error getting balance: {e}")
                 if e.response_data:
                     print(f"API Response: {e.response_data}")
            time.sleep(0.5) # API 호출 간격
            
            # --- 보유 종목 조회 테스트 --- 
            print("\n--- Account Positions --- ")
            try:
                positions = broker.get_positions()
                if positions:
                     print(json.dumps(positions[0], indent=2, ensure_ascii=False)) # 첫번째 종목만 출력
                     print(f"... and {len(positions)-1} more positions.")
                else:
                     print("No positions found.")
            except KisBrokerError as e:
                print(f"Error getting positions: {e}")
            time.sleep(0.5)
                
            # --- 과거 데이터 조회 테스트 (KODEX 200 일봉) --- 
            print("\n--- Historical Daily Data (069500, last 5 days) --- ")
            try:
                # 최근 5 거래일 데이터 요청 (period 사용)
                # KisBroker는 내부적으로 period + 여유분 날짜 계산
                hist_daily = broker.get_historical_data("069500", timeframe='D', period=5)
                print(hist_daily.tail())
            except KisBrokerError as e:
                 print(f"Error getting daily historical data: {e}")
            time.sleep(0.5)

            # --- 과거 데이터 조회 테스트 (KODEX 200 월봉) --- 
            print("\n--- Historical Monthly Data (069500, last 6 months) --- ")
            try:
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=180) # 약 6개월 전
                hist_monthly = broker.get_historical_data("069500", timeframe='M', 
                                                        start_date=start_dt.strftime('%Y%m%d'),
                                                        end_date=end_dt.strftime('%Y%m%d'))
                print(hist_monthly.tail())
            except KisBrokerError as e:
                 print(f"Error getting monthly historical data: {e}")

        except KisBrokerError as e:
            print(f"\nAn error occurred during testing: {e}")
        finally:
            broker.close() 