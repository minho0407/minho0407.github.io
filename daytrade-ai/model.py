"""LightGBM 모델 + 손절/익절 시뮬레이션 + 워크포워드 백테스트 + 승률 보정 + 매매 계획.

타깃 설계: '수익률 / ATR'(위험 대비 수익)을 회귀로 예측해 종목 순위를 매긴다.
  - "X% 오를 확률" 분류는 기대수익과 상관없이 변동성 큰 종목만 고르는 함정이 있음.

승리 확률: 모델 출력이 아니라 '과거 표본외 기간에 비슷한 순위였던 종목이
  실제 매매 규칙(시가 진입·익절·손절·기간 청산)대로 거래했을 때 이긴 비율'이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.isotonic import IsotonicRegression

from features import FEATURES

PARAMS = dict(
    objective="huber", alpha=1.0, n_estimators=300, learning_rate=0.03, num_leaves=15,
    min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=5.0, verbose=-1, n_jobs=-1,
)


@dataclass
class Rule:
    """매매 규칙. 가격 배수는 ATR(14일 평균 변동폭, 종가 대비 비율) 기준."""
    horizon: int = 1       # 1=당일 청산, 2~3=스윙
    tp: float = 1.0        # 익절: 진입가 + tp × ATR
    sl: float = 0.6        # 손절: 진입가 − sl × ATR
    max_gap: float = 0.04  # 시가 갭상승이 이 값 또는 익절폭의 절반을 넘으면 진입 포기
    cost: float = 0.003    # 왕복 비용


def make_model() -> LGBMRegressor:
    return LGBMRegressor(**PARAMS)


def liquidity_filter(df: pd.DataFrame, min_turnover: float, min_price: float) -> pd.DataFrame:
    return df[(df["turnover_20"] >= min_turnover) & (df["close"] >= min_price)]


def simulate(df: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    """일봉으로 규칙대로 거래했을 때의 결과.
    같은 봉에서 손절가·익절가가 모두 닿으면 '손절 먼저'로 보수적으로 처리."""
    entry = df["o1"].to_numpy()
    gap = entry / df["close"].to_numpy() - 1
    atr = df["atr"].to_numpy()
    tp, sl = entry * (1 + rule.tp * atr), entry * (1 - rule.sl * atr)
    exit_px = np.full(len(df), np.nan)
    how = np.full(len(df), "", dtype=object)
    open_ = np.ones(len(df), dtype=bool)
    for k in range(1, rule.horizon + 1):
        o, h, l, c = (df[f"{x}{k}"].to_numpy() for x in "ohlc")
        if k > 1:  # 둘째 날부터는 시가에 이미 손절/익절선을 넘어설 수 있음
            gd, gu = open_ & (o <= sl), open_ & (o >= tp)
            exit_px[gd], how[gd] = o[gd], "손절(갭)"
            exit_px[gu], how[gu] = o[gu], "익절(갭)"
            open_ &= ~(gd | gu)
        hit_sl, hit_tp = open_ & (l <= sl), open_ & ~(l <= sl) & (h >= tp)
        exit_px[hit_sl], how[hit_sl] = sl[hit_sl], "손절"
        exit_px[hit_tp], how[hit_tp] = tp[hit_tp], "익절"
        open_ &= ~(hit_sl | hit_tp)
        if k == rule.horizon:
            exit_px[open_], how[open_] = c[open_], "기간청산"
    skip = gap > gap_limit(rule, atr)
    ret = exit_px / entry - 1 - rule.cost
    ret[skip], how[skip] = np.nan, "갭상승 패스"
    return pd.DataFrame({"sim_ret": ret, "exit": how}, index=df.index)


def gap_limit(rule: Rule, atr):
    """진입 허용 갭: 익절까지 거리의 절반 이상을 시가에 이미 오른 경우는 패스."""
    return np.minimum(rule.max_gap, 0.5 * rule.tp * atr)


def _target(df: pd.DataFrame, ycol: str) -> pd.Series:
    return (df[ycol] / df["atr"]).clip(-4, 4)


def walk_forward(df: pd.DataFrame, ycol: str, rule: Rule,
                 min_train_days: int = 250, step: int = 20) -> pd.DataFrame:
    """과거로만 학습 → 그 다음 구간 예측을 반복. 반환: 표본외(OOS) 예측 + 시뮬레이션 결과."""
    lab = df.dropna(subset=[ycol, f"c{rule.horizon}"]).copy()
    lab["y"] = _target(lab, ycol)
    dates = np.sort(lab["date"].unique())
    if len(dates) < min_train_days + step:
        raise ValueError(f"데이터 기간이 부족합니다 ({len(dates)}일). --period를 늘리세요.")
    embargo = rule.horizon + 1  # 학습 라벨이 테스트 기간과 겹치지 않도록 간격
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
    oos["rank_pct"] = oos.groupby("date")["score"].rank(pct=True)
    return oos.join(simulate(oos, rule))


class Calibrator:
    """그날 유니버스 내 순위(0~1) → 과거 실제 승률·평균수익 (단조 증가 회귀)."""

    def __init__(self, oos: pd.DataFrame):
        t = oos.dropna(subset=["sim_ret"])
        self.win = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
            t["rank_pct"], (t["sim_ret"] > 0).astype(float))
        self.ret = IsotonicRegression(out_of_bounds="clip").fit(t["rank_pct"], t["sim_ret"])

    def __call__(self, rank_pct):
        return self.win.predict(rank_pct), self.ret.predict(rank_pct)


def backtest(oos: pd.DataFrame, rule: Rule, top: int, min_win: float | None = None) -> dict:
    oos = oos.copy()
    oos["rank"] = oos.groupby("date")["score"].rank(ascending=False, method="first")
    picks = oos[(oos["rank"] <= top)].dropna(subset=["sim_ret"])
    if min_win is not None:  # 승률 필터는 '그 시점 이전' 데이터로만 보정해야 공정 → 앞 절반으로 보정, 뒤 절반 평가
        half = np.sort(oos["date"].unique())[len(oos["date"].unique()) // 2]
        cal = Calibrator(oos[oos["date"] < half])
        picks = picks[picks["date"] >= half]
        picks = picks[cal(picks["rank_pct"])[0] >= min_win]
        oos = oos[oos["date"] >= half]

    allr = oos.dropna(subset=["sim_ret"])
    base_daily = allr.groupby("date")["sim_ret"].mean()  # 비교: 같은 규칙으로 유니버스 전체를 산 경우
    daily = picks.groupby("date")["sim_ret"].mean().reindex(base_daily.index).fillna(0)
    eq = (1 + daily / rule.horizon).cumprod()   # 스윙은 자금을 h등분해 겹쳐 운용
    base_eq = (1 + base_daily / rule.horizon).cumprod()

    r = picks["sim_ret"]
    pf = r[r > 0].sum() / abs(r[r <= 0].sum()) if (r <= 0).any() else np.nan
    oos["decile"] = (oos["rank_pct"] * 10).clip(upper=9.999).astype(int)
    dec = oos.dropna(subset=["sim_ret"]).groupby("decile")["sim_ret"].mean()
    ic = oos.groupby("date").apply(lambda g: g["score"].corr(g[f"c{rule.horizon}"] / g["o1"], method="spearman"),
                                   include_groups=False)
    return {
        "기간": f"{oos['date'].min():%Y-%m-%d} ~ {oos['date'].max():%Y-%m-%d}",
        "거래수": len(picks),
        "승률": (r > 0).mean(),
        "평균수익(비용차감)": r.mean(),
        "중앙값수익": r.median(),
        "손익비(PF)": pf,
        "누적수익": eq.iloc[-1] - 1,
        "최대낙폭(MDD)": (eq / eq.cummax() - 1).min(),
        "비교: 전체종목 승률": (allr["sim_ret"] > 0).mean(),
        "비교: 전체종목 평균": allr["sim_ret"].mean(),
        "비교: 전체종목 누적": base_eq.iloc[-1] - 1,
        "순위상관(IC)": ic.mean(),
        "_exit": picks["exit"].value_counts(normalize=True),
        "_decile": dec,
        "_equity": eq,
        "_picks": picks,
    }


def plan(df: pd.DataFrame, ycol: str, rule: Rule, top: int, oos: pd.DataFrame | None = None):
    """전체 데이터로 학습 → 최신 날짜 기준 후보 + 승률/진입/익절/손절."""
    lab = df.dropna(subset=[ycol]).copy()
    lab["y"] = _target(lab, ycol)
    m = make_model().fit(lab[FEATURES], lab["y"])
    last = df["date"].max()
    today = df[df["date"] == last].copy()
    today["score"] = m.predict(today[FEATURES])
    today["rank_pct"] = today["score"].rank(pct=True)
    today = today.sort_values("score", ascending=False).head(top)
    if oos is not None:
        today["win_prob"], today["exp_ret"] = Calibrator(oos)(today["rank_pct"])
    else:
        today["win_prob"] = today["exp_ret"] = np.nan
    ref = today["close"]  # 진입은 다음 거래일 시가. 기준가는 오늘 종가
    today["entry_ref"] = ref
    today["entry_max"] = ref * (1 + gap_limit(rule, today["atr"]))
    today["tp_px"] = ref * (1 + rule.tp * today["atr"])
    today["sl_px"] = ref * (1 - rule.sl * today["atr"])
    imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return last, today, imp
