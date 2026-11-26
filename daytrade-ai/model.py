"""LightGBM 모델 + 워크포워드(walk-forward) 백테스트 + 오늘의 후보 추출.

타깃 설계: '수익률 / ATR'(위험 대비 수익)을 회귀로 예측한다.
  - "X% 이상 오를까?" 분류는 결국 변동성 큰 종목만 고르게 되는 함정이 있어서
    (기대수익이 0이어도 많이 흔들리면 확률이 높게 나옴) 쓰지 않는다.
  - 예측값 × ATR = 예상 수익률(%)로 다시 환산해 보여준다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from features import FEATURES

PARAMS = dict(
    objective="huber", alpha=1.0, n_estimators=300, learning_rate=0.03, num_leaves=15,
    min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=5.0, verbose=-1, n_jobs=-1,
)


def make_model() -> LGBMRegressor:
    return LGBMRegressor(**PARAMS)


def liquidity_filter(df: pd.DataFrame, min_turnover: float, min_price: float) -> pd.DataFrame:
    return df[(df["turnover_20"] >= min_turnover) & (df["close"] >= min_price)]


def _target(df: pd.DataFrame, ycol: str) -> pd.Series:
    return (df[ycol] / df["atr"]).clip(-4, 4)


def walk_forward(df: pd.DataFrame, ycol: str, horizon: int,
                 min_train_days: int = 250, step: int = 20) -> pd.DataFrame:
    """과거로만 학습 → 그 다음 구간 예측을 반복. 반환: 표본외(OOS) 예측이 붙은 행."""
    lab = df.dropna(subset=[ycol]).copy()
    lab["y"] = _target(lab, ycol)
    dates = np.sort(lab["date"].unique())
    if len(dates) < min_train_days + step:
        raise ValueError(f"데이터 기간이 부족합니다 ({len(dates)}일). --period를 늘리세요.")
    embargo = horizon + 1  # 학습 라벨이 테스트 기간과 겹치지 않도록 간격
    preds = []
    for s in range(min_train_days, len(dates), step):
        tr = lab[lab["date"] <= dates[s - embargo]]
        te = lab[lab["date"].isin(dates[s:s + step])].copy()
        if te.empty:
            continue
        m = make_model().fit(tr[FEATURES], tr["y"])
        te["score"] = m.predict(te[FEATURES])
        preds.append(te)
    oos = pd.concat(preds, ignore_index=True)
    oos["exp_ret"] = oos["score"] * oos["atr"]
    return oos


def backtest(oos: pd.DataFrame, ycol: str, cost: float, top: int, horizon: int,
             min_edge: float | None = None) -> dict:
    oos = oos.copy()
    oos["net"] = oos[ycol] - cost
    oos["rank"] = oos.groupby("date")["score"].rank(ascending=False, method="first")
    picks = oos[oos["rank"] <= top]
    if min_edge is not None:
        picks = picks[picks["exp_ret"] - cost >= min_edge]

    base_daily = oos.groupby("date")["net"].mean()  # 비교 기준: 유니버스 전체를 산 경우
    daily = picks.groupby("date")["net"].mean().reindex(base_daily.index).fillna(0)
    sleeves = horizon  # 스윙은 자금을 h등분해 겹쳐 운용 (당일 단타는 1)
    eq = (1 + daily / sleeves).cumprod()
    base_eq = (1 + base_daily / sleeves).cumprod()

    wins, losses = picks.loc[picks["net"] > 0, "net"], picks.loc[picks["net"] <= 0, "net"]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
    # 점수 10분위별 실제 수익 → 모델이 '순서'를 맞추는지 확인하는 핵심 지표
    oos["decile"] = oos.groupby("date")["score"].transform(
        lambda p: pd.qcut(p.rank(method="first"), 10, labels=False) if len(p) >= 10 else np.nan)
    dec = oos.groupby("decile")["net"].agg(["mean", "count"])
    ic = oos.groupby("date").apply(lambda g: g["score"].corr(g[ycol], method="spearman"),
                                   include_groups=False)

    return {
        "기간": f"{oos['date'].min():%Y-%m-%d} ~ {oos['date'].max():%Y-%m-%d}",
        "거래수": len(picks),
        "승률": (picks["net"] > 0).mean(),
        "평균수익(비용차감)": picks["net"].mean(),
        "중앙값수익": picks["net"].median(),
        "손익비(PF)": pf,
        "누적수익": eq.iloc[-1] - 1,
        "최대낙폭(MDD)": (eq / eq.cummax() - 1).min(),
        "비교: 전체종목 평균": oos["net"].mean(),
        "비교: 전체종목 누적": base_eq.iloc[-1] - 1,
        "순위상관(IC)": ic.mean(),
        "_decile": dec,
        "_equity": eq,
        "_picks": picks,
    }


def fit_and_pick(df: pd.DataFrame, ycol: str, top: int):
    lab = df.dropna(subset=[ycol]).copy()
    lab["y"] = _target(lab, ycol)
    m = make_model().fit(lab[FEATURES], lab["y"])
    last = df["date"].max()
    today = df[df["date"] == last].copy()
    today["score"] = m.predict(today[FEATURES])
    today["exp_ret"] = today["score"] * today["atr"]
    today = today.sort_values("score", ascending=False).head(top)
    # ATR 기반 참고용 손절/익절 (진입가 = 다음날 시가 ≈ 오늘 종가 가정)
    today["stop"] = today["close"] * (1 - 1.0 * today["atr"])
    today["target"] = today["close"] * (1 + 1.5 * today["atr"])
    imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return last, today, imp, m
