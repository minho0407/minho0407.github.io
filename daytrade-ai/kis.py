"""한국투자증권(KIS Developers) Open API 클라이언트 — 시세 조회 전용 (주문 기능 없음).

공식 예제(github.com/koreainvestment/open-trading-api) 기준
  REST  실전 https://openapi.koreainvestment.com:9443   / 모의 https://openapivts.koreainvestment.com:29443
  WS    실전 ws://ops.koreainvestment.com:21000         / 모의 ws://ops.koreainvestment.com:31000
  접근토큰  POST /oauth2/tokenP   (24시간 유효, 발급은 1분에 1회 제한 → 파일에 캐시해서 재사용)
  웹소켓키  POST /oauth2/Approval
  국내 실시간체결가 H0STCNT0 (tr_key=종목코드 6자리)
  해외 실시간체결가 HDFSCNT0 (tr_key=D+거래소+심볼, 예: DNASAAPL. 미국은 무료 실시간)

환경변수: KIS_APP_KEY, KIS_APP_SECRET, (선택) KIS_PAPER=1 → 모의투자 서버
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta

import requests

REST = {"real": "https://openapi.koreainvestment.com:9443",
        "paper": "https://openapivts.koreainvestment.com:29443"}
WS = {"real": "ws://ops.koreainvestment.com:21000", "paper": "ws://ops.koreainvestment.com:31000"}
WS_PATH = "/tryitout"

# 실시간 체결가 필드 순서 (공식 예제의 columns)
KR_COLS = [
    "MKSC_SHRN_ISCD", "STCK_CNTG_HOUR", "STCK_PRPR", "PRDY_VRSS_SIGN", "PRDY_VRSS", "PRDY_CTRT",
    "WGHN_AVRG_STCK_PRC", "STCK_OPRC", "STCK_HGPR", "STCK_LWPR", "ASKP1", "BIDP1", "CNTG_VOL",
    "ACML_VOL", "ACML_TR_PBMN", "SELN_CNTG_CSNU", "SHNU_CNTG_CSNU", "NTBY_CNTG_CSNU", "CTTR",
    "SELN_CNTG_SMTN", "SHNU_CNTG_SMTN", "CCLD_DVSN", "SHNU_RATE", "PRDY_VOL_VRSS_ACML_VOL_RATE",
    "OPRC_HOUR", "OPRC_VRSS_PRPR_SIGN", "OPRC_VRSS_PRPR", "HGPR_HOUR", "HGPR_VRSS_PRPR_SIGN",
    "HGPR_VRSS_PRPR", "LWPR_HOUR", "LWPR_VRSS_PRPR_SIGN", "LWPR_VRSS_PRPR", "BSOP_DATE",
    "NEW_MKOP_CLS_CODE", "TRHT_YN", "ASKP_RSQN1", "BIDP_RSQN1", "TOTAL_ASKP_RSQN",
    "TOTAL_BIDP_RSQN", "VOL_TNRT", "PRDY_SMNS_HOUR_ACML_VOL", "PRDY_SMNS_HOUR_ACML_VOL_RATE",
    "HOUR_CLS_CODE", "MRKT_TRTM_CLS_CODE", "VI_STND_PRC", "MARKET_CLS_CODE",
]
US_COLS = [
    "RSYM", "SYMB", "ZDIV", "TYMD", "XYMD", "XHMS", "KYMD", "KHMS", "OPEN", "HIGH", "LOW", "LAST",
    "SIGN", "DIFF", "RATE", "PBID", "PASK", "VBID", "VASK", "EVOL", "TVOL", "TAMT", "BIVL", "ASVL",
    "STRN", "MTYP",
]
TR_KR, TR_US = "H0STCNT0", "HDFSCNT0"
MAX_SUBS = 40  # 웹소켓 1세션 최대 구독 수

STATE_DIR = os.environ.get("DAYTRADE_STATE_DIR", os.path.join(os.path.dirname(__file__), "state"))


class KIS:
    def __init__(self, app_key: str | None = None, app_secret: str | None = None, paper: bool | None = None):
        self.app_key = app_key or os.environ["KIS_APP_KEY"]
        self.app_secret = app_secret or os.environ["KIS_APP_SECRET"]
        if paper is None:
            paper = os.environ.get("KIS_PAPER", "") in ("1", "true", "yes")
        self.env = "paper" if paper else "real"
        self.rest, self.ws_url = REST[self.env], WS[self.env] + WS_PATH
        os.makedirs(STATE_DIR, exist_ok=True)
        self._token_file = os.path.join(STATE_DIR, f"kis_token_{self.env}.json")
        self._excd_file = os.path.join(STATE_DIR, "us_exchange.json")
        self._token: str | None = None
        self._approval: str | None = None

    # ---------- 인증 ----------
    def token(self) -> str:
        if self._token:
            return self._token
        try:
            with open(self._token_file) as f:
                c = json.load(f)
            if c.get("app") == self.app_key[-6:] and datetime.fromisoformat(c["exp"]) > datetime.now() + timedelta(minutes=30):
                self._token = c["token"]
                return self._token
        except (OSError, ValueError, KeyError):
            pass
        body = {"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret}
        for attempt in range(3):
            r = requests.post(f"{self.rest}/oauth2/tokenP", json=body, timeout=10)
            d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            if r.status_code == 200 and d.get("access_token"):
                break
            # EGW00133: 토큰 발급 1분당 1회 제한
            print(f"[kis] 토큰 발급 실패 ({r.status_code} {d.get('error_code', '')} {d.get('error_description', '')}) → 65초 후 재시도")
            time.sleep(65)
        else:
            raise RuntimeError("KIS 접근토큰 발급 실패. 앱키/시크릿과 서비스 신청 상태를 확인하세요.")
        exp = d.get("access_token_token_expired") or (datetime.now() + timedelta(seconds=int(d.get("expires_in", 86400)))).strftime("%Y-%m-%d %H:%M:%S")
        self._token = d["access_token"]
        with open(self._token_file, "w") as f:
            json.dump({"token": self._token, "exp": exp.replace(" ", "T"), "app": self.app_key[-6:]}, f)
        os.chmod(self._token_file, 0o600)
        return self._token

    def approval_key(self, refresh: bool = False) -> str:
        if self._approval and not refresh:
            return self._approval
        body = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.app_secret}
        r = requests.post(f"{self.rest}/oauth2/Approval", json=body, timeout=10)
        r.raise_for_status()
        self._approval = r.json()["approval_key"]
        return self._approval

    # ---------- REST 시세 ----------
    def _get(self, path: str, tr_id: str, params: dict) -> dict:
        h = {"content-type": "application/json; charset=utf-8", "authorization": f"Bearer {self.token()}",
             "appkey": self.app_key, "appsecret": self.app_secret, "tr_id": tr_id, "custtype": "P"}
        r = requests.get(f"{self.rest}{path}", headers=h, params=params, timeout=10)
        r.raise_for_status()
        d = r.json()
        if d.get("rt_cd") not in ("0", None):
            raise RuntimeError(f"KIS 오류 {d.get('msg_cd')}: {d.get('msg1')}")
        return d

    def kr_price(self, code: str) -> dict:
        d = self._get("/uapi/domestic-stock/v1/quotations/inquire-price", "FHKST01010100",
                      {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code})
        o = d["output"]
        return {"price": float(o["stck_prpr"]), "open": float(o["stck_oprc"]), "name": o.get("hts_kor_isnm", "")}

    def us_price(self, symbol: str, excd: str) -> dict:
        d = self._get("/uapi/overseas-price/v1/quotations/price", "HHDFS00000300",
                      {"AUTH": "", "EXCD": excd, "SYMB": symbol})
        o = d.get("output") or {}
        return {"price": float(o["last"]) if o.get("last") else None}

    def us_exchange(self, symbol: str) -> str | None:
        """미국 종목의 거래소 코드(NAS/NYS/AMS)를 조회해 캐시."""
        try:
            with open(self._excd_file) as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
        if symbol in cache:
            return cache[symbol]
        for ex in ("NAS", "NYS", "AMS"):
            try:
                if self.us_price(symbol, ex)["price"]:
                    cache[symbol] = ex
                    with open(self._excd_file, "w") as f:
                        json.dump(cache, f)
                    return ex
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.1)  # 초당 호출 제한
        return None


# ---------- 웹소켓 메시지 ----------
def subscribe_msg(approval_key: str, tr_id: str, tr_key: str, on: bool = True) -> str:
    return json.dumps({
        "header": {"approval_key": approval_key, "custtype": "P", "tr_type": "1" if on else "2",
                   "content-type": "utf-8"},
        "body": {"input": {"tr_id": tr_id, "tr_key": tr_key}},
    })


def parse_data(raw: str) -> tuple[str, list[dict]]:
    """'0|H0STCNT0|002|a^b^c^...' → (tr_id, [레코드 dict, ...]). 한 메시지에 여러 건이 붙어 올 수 있음."""
    _, tr_id, cnt, body = raw.split("|", 3)
    cols = KR_COLS if tr_id == TR_KR else US_COLS if tr_id == TR_US else None
    if cols is None:
        return tr_id, []
    vals, n = body.split("^"), len(cols)
    return tr_id, [dict(zip(cols, vals[i * n:(i + 1) * n])) for i in range(int(cnt)) if len(vals) >= (i + 1) * n]


def us_tr_key(symbol: str, excd: str) -> str:
    return f"D{excd}{symbol}"
