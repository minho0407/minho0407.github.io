"""단타 종목 선별 AI - 실행 진입점.

예시
  python main.py backtest --market kr --mode day             # 당일 단타 규칙 검증
  python main.py backtest --market us --mode swing --horizon 2
  python main.py pick --market kr --mode day --top 5         # 다음 거래일 후보 + 승률/진입/익절/손절
  python main.py pick --market kr --mode day --notify        # 결과를 텔레그램으로 전송
  python main.py scan --market us --notify                   # 장중 강세 종목 스캔
  python main.py backtest --demo                             # 네트워크 없이 합성 데이터로 시험
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import data
import features
import model
import notify

DEFAULTS = {  # 시장별 기본값: 왕복비용(수수료+세금+슬리피지), 최소 거래대금, 최소 가격
    "kr": dict(cost=0.0030, min_turnover=5e9, min_price=1000),
    "us": dict(cost=0.0010, min_turnover=2e7, min_price=3),
}
TZ = {"kr": "Asia/Seoul", "us": "America/New_York"}
RULE_DEFAULTS = {"day": dict(tp=1.0, sl=0.6), "swing": dict(tp=1.5, sl=1.0)}
OUT_DIR = notify.OUT_DIR


def pct(x):
    return f"{x * 100:+.2f}%" if pd.notna(x) else "-"


def make_rule(args) -> model.Rule:
    d = RULE_DEFAULTS[args.mode]
    return model.Rule(horizon=args.horizon if args.mode == "swing" else 1,
                      tp=args.tp if args.tp is not None else d["tp"],
                      sl=args.sl if args.sl is not None else d["sl"],
                      max_gap=args.max_gap, cost=args.cost)


def mode_name(args) -> str:
    return "당일 단타" if args.mode == "day" else f"{args.horizon}일 스윙"


def load(args) -> pd.DataFrame:
    if args.demo:
        print("[demo] 합성 데이터 사용 (실제 시장 아님)")
        return features.build(data.synthetic())
    tickers = (open(args.tickers).read().split() if args.tickers
               else data.get_universe(args.market, args.universe))
    print(f"[data] {args.market.upper()} {len(tickers)}종목, 기간 {args.period}")
    df = features.build(data.download(tickers, period=args.period))
    d = DEFAULTS[args.market]
    return model.liquidity_filter(df, args.min_turnover or d["min_turnover"], d["min_price"])


def print_report(r: dict, title: str) -> None:
    print(f"\n===== {title} =====")
    for k, v in r.items():
        if k.startswith("_"):
            continue
        if k in ("기간", "거래수"):
            s = v
        elif "승률" in k:
            s = f"{v:.1%}"
        elif k in ("손익비(PF)", "순위상관(IC)"):
            s = f"{v:.3f}"
        else:
            s = pct(v)
        print(f"{k:>16}: {s}")
    print("\n청산 방식 비율:", ", ".join(f"{k} {v:.0%}" for k, v in r["_exit"].items()))
    print("\n모델 점수 10분위별 실제 평균수익 (0=최하위 … 9=최상위, 아래로 갈수록 커지면 모델이 작동하는 것)")
    print(r["_decile"].map(pct).to_string())


def cmd_backtest(args):
    df, rule = load(args), make_rule(args)
    ycol = features.target_col(args.mode, rule.horizon)
    oos = model.walk_forward(df, ycol, rule)
    r = model.backtest(oos, rule, args.top, args.min_win)
    print_report(r, f"워크포워드 백테스트: {args.market.upper()} / {mode_name(args)} / 매일 상위 {args.top}종목 "
                    f"/ 익절 {rule.tp}ATR · 손절 {rule.sl}ATR")
    os.makedirs(OUT_DIR, exist_ok=True)
    f = os.path.join(OUT_DIR, f"backtest_{args.market}_{args.mode}.csv")
    r["_picks"][["date", "ticker", "score", "rank_pct", "sim_ret", "exit"]].to_csv(f, index=False)
    print(f"\n거래 내역 저장: {f}")


def cmd_pick(args):
    df, rule = load(args), make_rule(args)
    ycol = features.target_col(args.mode, rule.horizon)
    print("[pick] 승률 보정을 위해 워크포워드 검증 실행 중...")
    oos = model.walk_forward(df, ycol, rule)
    bt = model.backtest(oos, rule, args.top)
    last, picks, imp = model.plan(df, ycol, rule, args.top, oos)
    if args.min_win is not None:
        picks = picks[picks["win_prob"] >= args.min_win]
    stale = not args.demo and last.date() < pd.Timestamp.now(tz=TZ[args.market]).date()

    names = {}
    if args.market == "kr" and not args.demo:
        try:
            import FinanceDataReader as fdr
            lst = fdr.StockListing("KRX")
            names = dict(zip(lst["Code"].astype(str), lst["Name"]))
        except Exception:  # noqa: BLE001
            pass
    picks["name_kr"] = picks["ticker"].str.split(".").str[0].map(names).fillna("")

    show = pd.DataFrame({
        "티커": picks["ticker"], "종목명": picks["name_kr"],
        "승률": picks["win_prob"].map(lambda p: f"{p:.0%}"),
        "기대수익": picks["exp_ret"].map(pct),
        "기준가": picks["entry_ref"].round(2), "진입상한": picks["entry_max"].round(2),
        "익절가": picks["tp_px"].round(2), "손절가": picks["sl_px"].round(2),
        "거래량배수": np.exp(picks["vol_ratio"]).map(lambda v: f"{v:.1f}x"),
    })
    print(f"\n===== {last:%Y-%m-%d} 장 마감 기준 → 다음 거래일 {mode_name(args)} 후보 =====")
    print("진입: 다음 거래일 시가 매수 (시가가 진입상한 초과면 패스) / 익절·손절은 체결가 기준 같은 %")
    print(show.to_string(index=False) if len(show) else "조건을 만족하는 종목이 없습니다.")
    summary = (f"최근 검증: 상위{args.top} 승률 {bt['승률']:.0%}, 평균 {pct(bt['평균수익(비용차감)'])}"
               f" (전체종목 {pct(bt['비교: 전체종목 평균'])})")
    print("\n" + summary)
    print("모델이 중요하게 본 특징 Top 8:", ", ".join(imp.head(8).index))
    os.makedirs(OUT_DIR, exist_ok=True)
    f = os.path.join(OUT_DIR, f"picks_{args.market}_{args.mode}_{last:%Y%m%d}.csv")
    show.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"저장: {f}")
    if args.notify and stale and not args.force:
        print(f"[notify] 최신 데이터가 {last:%Y-%m-%d} (오늘 아님: 휴장일/데이터 지연) → 전송 생략")
        return
    if args.notify:
        payload = {
            "kind": "plan", "market": args.market, "mode": mode_name(args), "asof": f"{last:%Y-%m-%d}",
            "generated": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M KST"),
            "summary": summary, "demo": args.demo,
            "backtest": {"win": bt["승률"], "avg": bt["평균수익(비용차감)"], "base_avg": bt["비교: 전체종목 평균"],
                         "ic": bt["순위상관(IC)"], "trades": bt["거래수"]},
            "picks": [dict(ticker=r.ticker, name=r.name_kr, win_prob=float(r.win_prob),
                           exp_ret=float(r.exp_ret), entry_ref=float(r.entry_ref),
                           entry_max=float(r.entry_max), tp=float(r.tp_px), sl=float(r.sl_px),
                           vol_x=float(np.exp(r.vol_ratio)))
                      for r in picks.itertuples()],
        }
        notify.send(notify.format_plan(args.market, mode_name(args), last, picks, summary),
                    f"{args.market}_{args.mode}", payload)


def cmd_scan(args):
    from intraday import scan
    tickers = (open(args.tickers).read().split() if args.tickers
               else data.get_universe(args.market, args.universe))
    out = scan(tickers, args.top)
    if out.empty:
        print("장중 데이터가 없습니다 (장 시작 전이거나 휴장일).")
        return
    for c in ("등락률", "VWAP괴리"):
        out[c] = out[c].map(pct)
    out["상대거래량"] = out["상대거래량"].map(lambda v: f"{v:.1f}x")
    out["점수"] = out["점수"].round(2)
    print(out.to_string(index=False))
    day = out.attrs.get("day")
    if args.notify and day and day < pd.Timestamp.now(tz=TZ[args.market]).date() and not args.force:
        print(f"[notify] 분봉 최신일이 {day} (오늘 아님: 휴장일) → 전송 생략")
        return
    if args.notify:
        payload = {"kind": "scan", "market": args.market,
                   "generated": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M KST"),
                   "rows": out.to_dict(orient="records")}
        notify.send(notify.format_scan(args.market, out), f"{args.market}_scan", payload)


def main():
    p = argparse.ArgumentParser(description="단타 종목 선별 AI")
    p.add_argument("command", choices=["backtest", "pick", "scan"])
    p.add_argument("--market", choices=["kr", "us"], default="kr")
    p.add_argument("--mode", choices=["day", "swing"], default="day",
                   help="day=다음날 시가 진입 당일 청산, swing=시가 진입 후 horizon일 보유")
    p.add_argument("--horizon", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--top", type=int, default=5, help="하루에 고를 종목 수")
    p.add_argument("--tp", type=float, default=None, help="익절 = 진입가 + tp×ATR (기본 day 1.0, swing 1.5)")
    p.add_argument("--sl", type=float, default=None, help="손절 = 진입가 − sl×ATR (기본 day 0.6, swing 1.0)")
    p.add_argument("--max-gap", type=float, default=0.04, help="시가 갭상승이 이 값(또는 익절폭의 절반)을 넘으면 진입 포기")
    p.add_argument("--cost", type=float, default=None, help="왕복 거래비용 (기본: kr 0.3%%, us 0.1%%)")
    p.add_argument("--min-win", type=float, default=None, help="과거 승률이 이 값 미만이면 제외 (예: 0.55)")
    p.add_argument("--min-turnover", type=float, default=None, help="20일 평균 거래대금 하한")
    p.add_argument("--universe", type=int, default=300, help="시총 상위 몇 종목을 볼지")
    p.add_argument("--period", default="4y", help="학습 데이터 기간 (yfinance 형식)")
    p.add_argument("--tickers", help="티커 목록 파일 (공백/줄바꿈 구분)")
    p.add_argument("--notify", action="store_true", help="결과를 텔레그램으로 전송")
    p.add_argument("--force", action="store_true", help="휴장일이어도 전송")
    p.add_argument("--demo", action="store_true", help="합성 데이터로 실행")
    args = p.parse_args()
    if args.cost is None:
        args.cost = DEFAULTS[args.market]["cost"]
    {"backtest": cmd_backtest, "pick": cmd_pick, "scan": cmd_scan}[args.command](args)


if __name__ == "__main__":
    main()
