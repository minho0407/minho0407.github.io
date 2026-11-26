"""특징(feature) 생성과 정답(label) 정의.

모든 특징은 '오늘 장 마감' 시점까지의 정보만 사용한다 (미래 정보 누수 없음).
진입은 항상 '다음 거래일 시가'로 가정한다.

- day   모드: 다음날 시가 매수 → 다음날 종가 매도 (당일 청산 단타)
- swing 모드: 다음날 시가 매수 → h거래일째 종가 매도 (1~3일 단타)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "gap", "body", "upper_wick",
    "lower_wick", "range", "close_pos", "vol_ratio", "vol_ratio_5", "log_turnover",
    "atr", "volat_20", "rsi_14", "dist_ma5", "dist_ma20", "dist_ma60", "brk_20",
    "brk_60", "up_streak", "r_ret_1", "r_vol_ratio", "r_turnover", "mkt_ret_1",
    "mkt_ret_5", "rel_ret_1",
]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    o, h, l, c, v = g["open"], g["high"], g["low"], g["close"], g["volume"]
    pc = c.shift(1)
    for n in (1, 3, 5, 10, 20):
        g[f"ret_{n}"] = c / c.shift(n) - 1
    g["gap"] = o / pc - 1
    g["body"] = (c - o) / o
    g["upper_wick"] = (h - np.maximum(o, c)) / c
    g["lower_wick"] = (np.minimum(o, c) - l) / c
    g["range"] = (h - l) / c
    g["close_pos"] = ((c - l) / (h - l).replace(0, np.nan)).fillna(0.5)
    vma20 = v.rolling(20).mean().shift(1)
    g["vol_ratio"] = np.log(v / vma20)
    g["vol_ratio_5"] = np.log(v.rolling(5).mean() / vma20)
    turnover = c * v
    g["turnover_20"] = turnover.rolling(20).mean()
    g["log_turnover"] = np.log(turnover.replace(0, np.nan))
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    g["atr"] = tr.rolling(14).mean() / c
    g["volat_20"] = g["ret_1"].rolling(20).std()
    g["rsi_14"] = _rsi(c)
    for n in (5, 20, 60):
        g[f"dist_ma{n}"] = c / c.rolling(n).mean() - 1
    g["brk_20"] = c / h.rolling(20).max().shift(1) - 1
    g["brk_60"] = c / h.rolling(60).max().shift(1) - 1
    up = (c > pc).astype(int)
    g["up_streak"] = up.groupby((up != up.shift()).cumsum()).cumsum() * up

    # --- 정답: 다음날 시가 진입 ---
    no = o.shift(-1)
    g["y_day"] = c.shift(-1) / no - 1                      # 당일 시가→종가
    g["y_day_high"] = h.shift(-1) / no - 1                 # 당일 최대 상승폭(참고)
    for hz in (1, 2, 3):
        g[f"y_swing_{hz}"] = c.shift(-hz) / no - 1         # hz일째 종가 청산
        # 손절/익절 시뮬레이션용 미래 OHLC (학습 특징에는 쓰지 않음)
        g[f"o{hz}"], g[f"h{hz}"] = o.shift(-hz), h.shift(-hz)
        g[f"l{hz}"], g[f"c{hz}"] = l.shift(-hz), c.shift(-hz)
    return g


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([_per_ticker(g) for _, g in df.groupby("ticker")], ignore_index=True)
    # 시장 전체 / 횡단면(같은 날 종목간 순위) 특징
    by = out.groupby("date")
    out["mkt_ret_1"] = by["ret_1"].transform("mean")
    out["mkt_ret_5"] = by["ret_5"].transform("mean")
    out["rel_ret_1"] = out["ret_1"] - out["mkt_ret_1"]
    out["r_ret_1"] = by["ret_1"].rank(pct=True)
    out["r_vol_ratio"] = by["vol_ratio"].rank(pct=True)
    out["r_turnover"] = by["log_turnover"].rank(pct=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=["dist_ma60", "vol_ratio", "atr"]).reset_index(drop=True)


def target_col(mode: str, horizon: int) -> str:
    return "y_day" if mode == "day" else f"y_swing_{horizon}"
