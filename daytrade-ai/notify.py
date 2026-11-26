"""결과 전달: 텔레그램 메시지 + output/ 폴더에 최신 리포트 저장.

텔레그램을 쓰려면 환경변수 두 개를 설정한다.
  TELEGRAM_BOT_TOKEN : @BotFather 에서 만든 봇 토큰
  TELEGRAM_CHAT_ID   : 내 채팅 ID (봇에게 아무 말이나 보낸 뒤
                       https://api.telegram.org/bot<토큰>/getUpdates 에서 chat.id 확인)
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _fmt_px(x: float, market: str) -> str:
    return f"{x:,.0f}" if market == "kr" else f"{x:,.2f}"


def format_plan(market: str, mode_name: str, last, picks, backtest_summary: str = "") -> str:
    lines = [f"📈 [{market.upper()}] {mode_name} 후보 ({last:%m/%d} 마감 기준)"]
    if backtest_summary:
        lines.append(backtest_summary)
    for i, r in enumerate(picks.itertuples(), 1):
        name = f" {r.name_kr}" if getattr(r, "name_kr", "") else ""
        wp = f"{r.win_prob:.0%}" if r.win_prob == r.win_prob else "-"
        er = f"{r.exp_ret * 100:+.2f}%" if r.exp_ret == r.exp_ret else "-"
        lines.append(
            f"\n{i}. {r.ticker}{name}\n"
            f"   승률 {wp} · 기대 {er}\n"
            f"   진입 시가 (≤{_fmt_px(r.entry_max, market)} 일 때만, 기준 {_fmt_px(r.entry_ref, market)})\n"
            f"   익절 {_fmt_px(r.tp_px, market)} (+{(r.tp_px / r.entry_ref - 1) * 100:.1f}%)"
            f" · 손절 {_fmt_px(r.sl_px, market)} ({(r.sl_px / r.entry_ref - 1) * 100:.1f}%)")
    lines.append("\n※ 익절·손절은 실제 체결가 기준 같은 % 적용. 승률은 과거 백테스트 실적이며 보장 아님.")
    return "\n".join(lines)


def format_scan(market: str, out) -> str:
    lines = [f"⚡ [{market.upper()}] 장중 강세 종목 (규칙 기반, 승률 미검증)"]
    for i, r in enumerate(out.itertuples(), 1):
        lines.append(f"{i}. {r.ticker} {r.현재가:,.2f} ({r.등락률}) 거래량 {r.상대거래량}"
                     f" | 진입 {r.진입:,.2f} 익절 {r.익절:,.2f} 손절 {r.손절:,.2f}")
    return "\n".join(lines)


def send(text: str, tag: str, payload: dict | None = None) -> None:
    """텍스트/JSON 저장 + 텔레그램 전송. SITE_DATA_DIR 환경변수가 있으면 거기에도 JSON 저장(웹페이지용)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"latest_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    if payload is not None:
        for d in filter(None, (OUT_DIR, os.environ.get("SITE_DATA_DIR"))):
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{tag}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        print("[notify] 텔레그램 미설정 → output/ 에만 저장")
        return
    for i in range(0, len(text), 3900):  # 텔레그램 메시지 길이 제한 4096자
        body = urllib.parse.urlencode({"chat_id": chat, "text": text[i:i + 3900]}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body)
        with urllib.request.urlopen(req, timeout=20) as resp:
            ok = json.load(resp).get("ok")
        print("[notify] 텔레그램 전송", "성공" if ok else "실패")
