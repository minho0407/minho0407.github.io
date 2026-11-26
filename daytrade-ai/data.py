"""데이터 수집: 종목 유니버스 + 일봉/분봉 다운로드 (yfinance 기반, 한국/미국 공통)."""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# 네트워크로 목록을 못 가져올 때 쓰는 기본 유니버스
FALLBACK_KR = [
    "005930.KS", "000660.KS", "373220.KS", "207940.KS", "005380.KS", "000270.KS",
    "068270.KS", "105560.KS", "035420.KS", "055550.KS", "012330.KS", "005490.KS",
    "028260.KS", "035720.KS", "006400.KS", "051910.KS", "003670.KS", "086790.KS",
    "032830.KS", "015760.KS", "034020.KS", "012450.KS", "042660.KS", "009540.KS",
    "010140.KS", "267260.KS", "329180.KS", "011200.KS", "066570.KS", "096770.KS",
    "247540.KQ", "086520.KQ", "196170.KQ", "028300.KQ", "058470.KQ", "277810.KQ",
    "145020.KQ", "403870.KQ", "214150.KQ", "141080.KQ", "039030.KQ", "357780.KQ",
    "263750.KQ", "293490.KQ", "035900.KQ", "041510.KQ", "240810.KQ", "112040.KQ",
]
FALLBACK_US = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "PLTR", "SMCI", "MU", "INTC", "QCOM", "ARM", "MRVL", "COIN", "MSTR", "HOOD",
    "SOFI", "RIVN", "LCID", "NIO", "UBER", "SHOP", "SNOW", "CRWD", "PANW", "NET",
    "DDOG", "MDB", "ROKU", "AFRM", "UPST", "MARA", "RIOT", "IONQ", "RGTI", "SOUN",
    "BA", "JPM", "XOM", "LLY", "UNH", "DIS", "NKE", "PYPL", "SQ", "DKNG",
]


def get_universe(market: str, size: int = 300) -> list[str]:
    """시가총액 상위 종목 리스트 (yfinance 티커 형식)."""
    market = market.lower()
    try:
        import FinanceDataReader as fdr

        if market == "kr":
            out = []
            for mk, suffix in (("KOSPI", ".KS"), ("KOSDAQ", ".KQ")):
                df = fdr.StockListing(mk)
                code_col = "Code" if "Code" in df.columns else "Symbol"
                if "Marcap" in df.columns:
                    df = df.sort_values("Marcap", ascending=False)
                n = size * 2 // 3 if mk == "KOSPI" else size - size * 2 // 3
                # 우선주/스팩 등 제외 (코드 끝자리 0 = 보통주)
                codes = [c for c in df[code_col].astype(str) if c.endswith("0")]
                if "Name" in df.columns:
                    names = dict(zip(df[code_col].astype(str), df["Name"].astype(str)))
                    codes = [c for c in codes if "스팩" not in names.get(c, "")]
                out += [c + suffix for c in codes[:n]]
            if out:
                return out
        elif market == "us":
            df = fdr.StockListing("S&P500")
            syms = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            syms = list(dict.fromkeys(FALLBACK_US + syms))  # 변동성 큰 종목 우선 포함
            if syms:
                return syms[:size]
    except Exception as e:  # noqa: BLE001
        print(f"[universe] 목록 조회 실패 → 기본 목록 사용 ({e.__class__.__name__})")
    return FALLBACK_KR if market == "kr" else FALLBACK_US


def _to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique()
        for t in tickers:
            if t not in lvl0:
                continue
            d = raw[t].copy()
            d["ticker"] = t
            frames.append(d)
    else:
        d = raw.copy()
        d["ticker"] = tickers[0]
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename_axis("date").reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["volume"] > 0]
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def download(tickers: list[str], period: str = "3y", interval: str = "1d",
             batch: int = 80, use_cache: bool = True) -> pd.DataFrame:
    """여러 종목 OHLCV를 long 포맷으로 반환. 같은 날 재실행 시 캐시 사용."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{hashlib.md5(','.join(sorted(tickers)).encode()).hexdigest()[:10]}_{period}_{interval}_{pd.Timestamp.now():%Y%m%d%H}"
    path = os.path.join(CACHE_DIR, key + ".pkl")
    if use_cache and interval == "1d" and os.path.exists(path):
        return pd.read_pickle(path)

    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, period=period, interval=interval, group_by="ticker",
                                  auto_adjust=True, threads=True, progress=False)
                frames.append(_to_long(raw, chunk))
                break
            except Exception as e:  # noqa: BLE001
                print(f"[download] 재시도 {attempt + 1}/3: {e}")
                time.sleep(2 * (attempt + 1))
        print(f"[download] {min(i + batch, len(tickers))}/{len(tickers)} 종목")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise RuntimeError("데이터를 하나도 받지 못했습니다. 인터넷 연결/티커를 확인하세요.")
    if interval == "1d":
        df["date"] = df["date"].dt.tz_localize(None) if df["date"].dt.tz is not None else df["date"]
        df.to_pickle(path)
    return df


def synthetic(n_tickers: int = 60, n_days: int = 700, seed: int = 0) -> pd.DataFrame:
    """네트워크 없이 파이프라인을 시험하기 위한 가짜 데이터.
    '거래량 급증 다음날 소폭 상승' 같은 약한 패턴을 일부러 심어 둠."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    rows = []
    for k in range(n_tickers):
        p, vol_base = 10000 * rng.uniform(0.2, 5), rng.uniform(1e5, 5e6)
        sigma = rng.uniform(0.015, 0.05)
        prev_surge = 0.0
        for d in dates:
            surge = rng.lognormal(0, 0.6)
            drift = 0.006 * (prev_surge > 2.5)  # 심어 둔 패턴
            o = p * (1 + rng.normal(0, sigma / 3))
            c = o * (1 + rng.normal(drift, sigma))
            h = max(o, c) * (1 + abs(rng.normal(0, sigma / 2)))
            lo = min(o, c) * (1 - abs(rng.normal(0, sigma / 2)))
            rows.append((d, f"SYN{k:03d}", o, h, lo, c, vol_base * surge))
            p, prev_surge = c, surge
    return pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"])
