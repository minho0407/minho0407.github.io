"""장중 스캐너 (규칙 기반). 5분봉으로 '지금' 강한 종목을 점수화한다.

분봉은 yfinance에서 최근 60일까지만 제공돼 ML 학습엔 부족하므로 규칙 점수를 쓴다.
- 상대거래량: 오늘 같은 시각까지 누적 거래량 / 최근 5일 같은 시각 평균
- VWAP 위에 있는지, VWAP 대비 괴리
- 시가범위(첫 30분) 고점 돌파 여부
- 당일 상승률 (너무 많이 오른 종목은 감점)
진입=현재가, 손절=VWAP/첫30분 저점 아래, 익절=손익비 2:1. 승률은 검증되지 않음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import download


def scan(tickers: list[str], top: int = 15) -> pd.DataFrame:
    df = download(tickers, period="10d", interval="5m", use_cache=False)
    df["day"] = df["date"].dt.date
    df["tod"] = df["date"].dt.time
    rows = []
    for t, g in df.groupby("ticker"):
        g = g.sort_values("date")
        days = sorted(g["day"].unique())
        if len(days) < 3:
            continue
        today = g[g["day"] == days[-1]]
        prev = g[g["day"].isin(days[-6:-1])]
        if len(today) < 3:
            continue
        now_tod = today["tod"].iloc[-1]
        cum_today = today["volume"].sum()
        cum_prev = prev[prev["tod"] <= now_tod].groupby("day")["volume"].sum().mean()
        rvol = cum_today / cum_prev if cum_prev else np.nan

        tp = (today["high"] + today["low"] + today["close"]) / 3
        vwap = (tp * today["volume"]).sum() / today["volume"].sum()
        last = today["close"].iloc[-1]
        prev_close = prev[prev["day"] == days[-2]]["close"].iloc[-1] if days[-2] in set(prev["day"]) else today["open"].iloc[0]
        chg = last / prev_close - 1
        orh = today.head(6)["high"].max()  # 첫 30분 고점
        orb = last > orh and len(today) > 6

        score = (np.log(max(rvol, 1e-3)) * 2
                 + 1.5 * (last > vwap) + 20 * (last / vwap - 1)
                 + 1.0 * orb
                 + (10 * chg if chg < 0.15 else 1.5 - 5 * (chg - 0.15)))
        # 손절: VWAP·첫30분 저점 중 현재가 아래의 높은 쪽 (현재가 대비 1~4%로 제한), 익절: 손익비 2:1
        orl = today.head(6)["low"].min()
        cands = [x for x in (vwap, orl) if x < last]
        sl = max(cands) * 0.998 if cands else last * 0.98
        sl = min(max(sl, last * 0.96), last * 0.99)
        tp = last + 2 * (last - sl)
        rows.append(dict(ticker=t, 현재가=last, 등락률=chg, 상대거래량=rvol,
                         VWAP괴리=last / vwap - 1, 시가범위돌파=orb, 점수=score,
                         진입=last, 익절=tp, 손절=sl))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("점수", ascending=False).head(top).reset_index(drop=True)
    out.attrs["day"] = max(df["day"])
    return out
