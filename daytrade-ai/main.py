"""단타 종목 선별 AI - 실행 진입점.

예시
  python main.py backtest --market kr --mode day            # 당일 단타 모델 검증
  python main.py backtest --market us --mode swing --horizon 2
  python main.py pick --market kr --mode day --top 10       # 내일 단타 후보
  python main.py scan --market us                           # 장중 실시간 스캐너
  python main.py backtest --demo                            # 네트워크 없이 합성 데이터로 시험
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import data
import features
import model

DEFAULTS = {  # 시장별 기본값: 왕복비용(수수료+세금+슬리피지), 최소 거래대금, 최소 가격
    "kr": dict(cost=0.0030, min_turnover=5e9, min_price=1000),
    "us": dict(cost=0.0010, min_turnover=2e7, min_price=3),
}
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def pct(x):
    return f"{x * 100:+.2f}%" if pd.notna(x) else "-"


def load(args) -> pd.DataFrame:
    if args.demo:
        print("[demo] 합성 데이터 사용 (실제 시장 아님)")
        raw = data.synthetic()
    else:
        tickers = (open(args.tickers).read().split() if args.tickers
                   else data.get_universe(args.market, args.universe))
        print(f"[data] {args.market.upper()} {len(tickers)}종목, 기간 {args.period}")
        raw = data.download(tickers, period=args.period)
    df = features.build(raw)
    d = DEFAULTS[args.market]
    if args.demo:
        return df
    return model.liquidity_filter(df, args.min_turnover or d["min_turnover"], d["min_price"])


def cmd_backtest(args):
    df = load(args)
    ycol = features.target_col(args.mode, args.horizon)
    hz = args.horizon if args.mode == "swing" else 1
    oos = model.walk_forward(df, ycol, hz)
    r = model.backtest(oos, ycol, args.cost, args.top, hz, args.min_edge)
    name = "당일 단타(다음날 시가→종가)" if args.mode == "day" else f"{args.horizon}일 스윙(시가 진입→{args.horizon}일째 종가)"
    print(f"\n===== 워크포워드 백테스트: {args.market.upper()} / {name} / 매일 상위 {args.top}종목 =====")
    for k, v in r.items():
        if k.startswith("_"):
            continue
        if k in ("기간", "거래수"):
            s = v
        elif k == "승률":
            s = f"{v:.1%}"
        elif k in ("손익비(PF)", "순위상관(IC)"):
            s = f"{v:.3f}"
        else:
            s = pct(v)
        print(f"{k:>16}: {s}")
    print("\n모델 점수 10분위별 실제 평균수익 (0=최하위 … 9=최상위, 아래로 갈수록 커지면 모델이 작동하는 것)")
    print(r["_decile"]["mean"].map(pct).to_string())
    os.makedirs(OUT_DIR, exist_ok=True)
    f = os.path.join(OUT_DIR, f"backtest_{args.market}_{args.mode}.csv")
    r["_picks"][["date", "ticker", "score", "exp_ret", ycol, "net"]].to_csv(f, index=False)
    print(f"\n거래 내역 저장: {f}")


def cmd_pick(args):
    df = load(args)
    ycol = features.target_col(args.mode, args.horizon)
    last, picks, imp, _ = model.fit_and_pick(df, ycol, args.top)
    names = {}
    if args.market == "kr" and not args.demo:
        try:
            import FinanceDataReader as fdr
            lst = fdr.StockListing("KRX")
            names = dict(zip(lst["Code"].astype(str), lst["Name"]))
        except Exception:  # noqa: BLE001
            pass
    picks["종목명"] = picks["ticker"].str.split(".").str[0].map(names).fillna("")
    show = pd.DataFrame({
        "티커": picks["ticker"], "종목명": picks["종목명"], "종가": picks["close"].round(2),
        "예상수익": picks["exp_ret"].map(pct),
        "당일등락": picks["ret_1"].map(pct), "거래량배수": np.exp(picks["vol_ratio"]).map(lambda v: f"{v:.1f}x"),
        "손절(참고)": picks["stop"].round(2), "익절(참고)": picks["target"].round(2),
    })
    mode = "당일 단타" if args.mode == "day" else f"{args.horizon}일 스윙"
    print(f"\n===== {last:%Y-%m-%d} 장 마감 기준 → 다음 거래일 {mode} 후보 Top {args.top} =====")
    print(show.to_string(index=False))
    print("\n모델이 중요하게 본 특징 Top 8:", ", ".join(imp.head(8).index))
    os.makedirs(OUT_DIR, exist_ok=True)
    f = os.path.join(OUT_DIR, f"picks_{args.market}_{args.mode}_{last:%Y%m%d}.csv")
    show.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"저장: {f}")


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


def main():
    p = argparse.ArgumentParser(description="단타 종목 선별 AI")
    p.add_argument("command", choices=["backtest", "pick", "scan"])
    p.add_argument("--market", choices=["kr", "us"], default="kr")
    p.add_argument("--mode", choices=["day", "swing"], default="day",
                   help="day=다음날 시가→종가, swing=시가 진입 후 horizon일 보유")
    p.add_argument("--horizon", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--top", type=int, default=10, help="하루에 고를 종목 수")
    p.add_argument("--cost", type=float, default=None, help="왕복 거래비용 (기본: kr 0.3%%, us 0.1%%)")
    p.add_argument("--min-edge", type=float, default=None,
                   help="예상수익-비용이 이 값 미만이면 매수 안 함 (예: 0.002)")
    p.add_argument("--min-turnover", type=float, default=None, help="20일 평균 거래대금 하한")
    p.add_argument("--universe", type=int, default=300, help="시총 상위 몇 종목을 볼지")
    p.add_argument("--period", default="4y", help="학습 데이터 기간 (yfinance 형식)")
    p.add_argument("--tickers", help="티커 목록 파일 (공백/줄바꿈 구분)")
    p.add_argument("--demo", action="store_true", help="합성 데이터로 실행")
    args = p.parse_args()
    if args.cost is None:
        args.cost = DEFAULTS[args.market]["cost"]
    {"backtest": cmd_backtest, "pick": cmd_pick, "scan": cmd_scan}[args.command](args)


if __name__ == "__main__":
    main()
