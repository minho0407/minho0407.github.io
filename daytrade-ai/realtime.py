"""실시간 감시 + 즉시 알림 (한국투자증권 웹소켓).

흐름
  1) 장 시작 전: GitHub Actions가 사이트에 올린 후보(daytrade/data/*.json)를 받아옴
  2) 장중: 후보 종목 실시간 체결가를 웹소켓으로 수신
  3) 이벤트마다 텔레그램으로 즉시 알림
       🟢 진입   : 시가가 진입상한 이하 → 시가 기준 익절/손절가 확정
       ⚪ 패스   : 시가가 진입상한 초과(갭상승)
       🎯 익절   : 현재가 ≥ 익절가
       🛑 손절   : 현재가 ≤ 손절가
       ⏰ 마감전 : 당일 단타(또는 스윙 마지막 날) 마감 15분 전 청산 안내
       📋 요약   : 장 마감 후 오늘 결과
  스윙 포지션은 state/positions.json 에 저장되어 다음 날에도 이어서 감시.

주문은 하지 않습니다. 알림만 보냅니다.

실행
  python realtime.py            # 상시 실행 (서버에서 systemd로 띄우기)
  python realtime.py --check    # 연결 점검: 키 발급 → 웹소켓 접속 → 삼성전자 체결가 수신 → 텔레그램 테스트
  python realtime.py --status   # 현재 감시 중인 포지션 보기
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import urllib.request
from datetime import date, datetime, time as dtime
from zoneinfo import ZoneInfo

import websockets

import kis as K
from notify import telegram

log = logging.getLogger("realtime")
KST, NY = ZoneInfo("Asia/Seoul"), ZoneInfo("America/New_York")
SESSIONS = {
    "kr": dict(tz=KST, pre=dtime(8, 45), open=dtime(9, 0), warn=dtime(15, 15), close=dtime(15, 30)),
    "us": dict(tz=NY, pre=dtime(9, 15), open=dtime(9, 30), warn=dtime(15, 45), close=dtime(16, 0)),
}
PLAN_TAGS = {"kr": ["kr_day", "kr_swing"], "us": ["us_day", "us_swing"]}
PICKS_URL = os.environ.get("PICKS_URL", "https://minho0407.github.io/daytrade/data")
STATE_FILE = os.path.join(K.STATE_DIR, "positions.json")
MAX_PLAN_AGE_DAYS = 5  # 후보 발표일로부터 이 기간이 지나도 진입 못 했으면 폐기


def fmt(px: float, market: str) -> str:
    return f"{px:,.0f}" if market == "kr" else f"{px:,.2f}"


def pctf(x: float) -> str:
    return f"{x * 100:+.2f}%"


class Monitor:
    def __init__(self, client: K.KIS | None, send=telegram):
        self.client, self.send = client, send
        self.pos: dict[str, dict] = self._load()
        self.traded_today: dict[str, date] = {}  # 시장별 오늘 체결 수신 여부(휴장일 판별)

    # ---------- 시계 (테스트에서 교체 가능) ----------
    def now(self, market: str) -> datetime:
        return datetime.now(SESSIONS[market]["tz"])

    # ---------- 상태 저장 ----------
    def _load(self) -> dict:
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save(self) -> None:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.pos, f, ensure_ascii=False, indent=1)
        os.replace(tmp, STATE_FILE)

    def alert(self, text: str) -> None:
        log.info("ALERT %s", text.replace("\n", " | "))
        self.send(text)

    # ---------- 후보 불러오기 ----------
    def fetch_plan(self, tag: str) -> dict | None:
        url = f"{PICKS_URL}/{tag}.json?t={int(datetime.now().timestamp())}"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            log.warning("후보 불러오기 실패 %s: %s", tag, e)
            return None

    def load_plans(self, market: str, plans: dict[str, dict] | None = None) -> None:
        today = self.now(market).date()
        for tag in PLAN_TAGS[market]:
            d = (plans or {}).get(tag) if plans is not None else self.fetch_plan(tag)
            if not d or d.get("demo"):
                continue
            asof = date.fromisoformat(d["asof"])
            if asof >= today or (today - asof).days > MAX_PLAN_AGE_DAYS:
                continue  # 오늘 마감 기준 후보(=내일용)이거나 너무 오래된 후보
            # 더 새 후보가 나오면 예전 '대기' 후보는 폐기
            for p in self.pos.values():
                if p["tag"] == tag and p["status"] == "waiting" and p["asof"] < d["asof"]:
                    p["status"] = "expired"
            for r in d["picks"]:
                pid = f"{tag}:{r['ticker']}:{d['asof']}"
                if pid in self.pos:
                    continue
                code = r["ticker"].split(".")[0]
                self.pos[pid] = dict(
                    id=pid, tag=tag, market=market, ticker=r["ticker"], code=code, key=None,
                    name=r.get("name") or code, mode=d.get("mode", tag), horizon=int(d.get("horizon", 1)),
                    asof=d["asof"], entry_ref=r["entry_ref"], entry_max=r["entry_max"],
                    tp_pct=r["tp"] / r["entry_ref"] - 1, sl_pct=1 - r["sl"] / r["entry_ref"],
                    win_prob=r.get("win_prob"), status="waiting", days=0, last_day=None,
                    entry=None, tp=None, sl=None, last=None, exit=None, reason=None, ret=None, warned=None)
        self.save()

    def resolve_keys(self, market: str) -> None:
        for p in self.active(market):
            if p["key"]:
                continue
            if market == "kr":
                p["key"] = p["code"]
            else:
                ex = self.client.us_exchange(p["code"]) if self.client else None
                if ex:
                    p["key"] = K.us_tr_key(p["code"], ex)
                else:
                    p["status"], p["reason"] = "expired", "거래소 조회 실패"
                    log.warning("%s 거래소 코드 조회 실패 → 감시 제외", p["code"])
        self.save()

    def active(self, market: str) -> list[dict]:
        return [p for p in self.pos.values() if p["market"] == market and p["status"] in ("waiting", "entered")]

    # ---------- 체결 이벤트 ----------
    def on_tick(self, market: str, key: str, price: float, open_: float, high: float, low: float,
                hhmmss: str) -> None:
        s = SESSIONS[market]
        if not (s["open"].strftime("%H%M%S") <= hhmmss < s["close"].strftime("%H%M%S")):
            return  # 정규장 외 체결 무시
        today = self.now(market).date()
        self.traded_today[market] = today
        changed = False
        for p in self.active(market):
            if p["key"] != key:
                continue
            p["last"] = price
            if p["status"] == "waiting":
                changed |= self._try_enter(p, open_ or price, high, low, today)
            elif p["status"] == "entered":
                if p["last_day"] != today.isoformat():  # 스윙: 새 거래일
                    p["days"] += 1
                    p["last_day"] = today.isoformat()
                    changed = True
                changed |= self._check_exit(p, price, price, at_level=False)
        if changed:
            self.save()

    def _try_enter(self, p: dict, open_: float, high: float, low: float, today: date) -> bool:
        m = p["market"]
        if open_ > p["entry_max"]:
            p["status"], p["reason"] = "skipped", "갭상승"
            self.alert(f"⚪ [{m.upper()}] 진입 패스 · {p['name']} ({p['code']})\n"
                       f"시가 {fmt(open_, m)} > 진입상한 {fmt(p['entry_max'], m)} (갭상승)")
            return True
        p.update(status="entered", entry=open_, tp=open_ * (1 + p["tp_pct"]), sl=open_ * (1 - p["sl_pct"]),
                 days=1, last_day=today.isoformat())
        wp = f" · 과거승률 {p['win_prob']:.0%}" if p.get("win_prob") is not None else ""
        self.alert(f"🟢 [{m.upper()}] 진입 · {p['name']} ({p['code']}) · {p['mode']}{wp}\n"
                   f"시가 {fmt(open_, m)} 매수\n"
                   f"🎯 익절 {fmt(p['tp'], m)} ({pctf(p['tp_pct'])})\n"
                   f"🛑 손절 {fmt(p['sl'], m)} ({pctf(-p['sl_pct'])})")
        # 감시를 늦게 시작했을 경우: 장 시작 후 고가/저가로 이미 닿았는지 확인 (손절 우선, 백테스트와 동일)
        self._check_exit(p, low or open_, high or open_, at_level=True)
        return True

    def _check_exit(self, p: dict, low: float, high: float, at_level: bool) -> bool:
        """at_level=True: 장중 고가/저가로 판정 → 익절/손절 '가격'에 체결됐다고 봄.
        False: 실시간 체결가로 판정 → 그 체결가로 보고 (갭으로 뚫린 경우 포함)."""
        m = p["market"]
        if low <= p["sl"]:
            self._close(p, p["sl"] if at_level else low, "손절")
            self.alert(f"🛑 [{m.upper()}] 손절가 이탈 · {p['name']} ({p['code']})\n"
                       f"현재 {fmt(p['exit'], m)} ({pctf(p['ret'])}) → 매도")
            return True
        if high >= p["tp"]:
            self._close(p, p["tp"] if at_level else high, "익절")
            self.alert(f"🎯 [{m.upper()}] 익절가 도달 · {p['name']} ({p['code']})\n"
                       f"현재 {fmt(p['exit'], m)} ({pctf(p['ret'])}) → 매도")
            return True
        return False

    def _close(self, p: dict, px: float, reason: str) -> None:
        p.update(status="closed", exit=px, reason=reason, ret=px / p["entry"] - 1,
                 closed=self.now(p["market"]).date().isoformat())

    # ---------- 시간 이벤트 ----------
    def _due_today(self, p: dict) -> bool:
        return p["status"] == "entered" and p["days"] >= p["horizon"]

    def on_clock(self, market: str) -> None:
        now = self.now(market)
        s, today = SESSIONS[market], now.date().isoformat()
        if now.time() >= s["warn"]:
            for p in self.active(market):
                if self._due_today(p) and p["warned"] != today and p["last"]:
                    p["warned"] = today
                    r = p["last"] / p["entry"] - 1
                    self.alert(f"⏰ [{market.upper()}] 마감 15분 전 청산 · {p['name']} ({p['code']})\n"
                               f"현재 {fmt(p['last'], market)} ({pctf(r)}) · {p['mode']} 보유기간 종료")
                    self.save()

    def end_session(self, market: str) -> None:
        today = self.now(market).date()
        traded = self.traded_today.get(market) == today
        if not traded:
            log.info("[%s] 오늘 체결 없음(휴장일) → 상태 유지", market)
            return
        for p in self.active(market):
            if self._due_today(p) and p["last"]:
                self._close(p, p["last"], "기간청산")
            elif p["status"] == "waiting" and p["asof"] < today.isoformat():
                p["status"], p["reason"] = "expired", "체결 없음(거래정지 등)"
        self.save()
        done = [p for p in self.pos.values() if p["market"] == market and p.get("closed") == today.isoformat()]
        holding = [p for p in self.active(market) if p["status"] == "entered"]
        if not done and not holding:
            return
        lines = [f"📋 [{market.upper()}] {today:%m/%d} 마감 요약"]
        for p in done:
            lines.append(f"· {p['name']} {p['reason']} {pctf(p['ret'])}")
        if done:
            avg = sum(p["ret"] for p in done) / len(done)
            wins = sum(p["ret"] > 0 for p in done)
            lines.append(f"청산 {len(done)}건 · 승 {wins} · 평균 {pctf(avg)} (비용 전)")
        for p in holding:
            lines.append(f"· 보유중 {p['name']} {p['days']}/{p['horizon']}일차 {pctf(p['last'] / p['entry'] - 1)}")
        self.alert("\n".join(lines))
        # 오래된 기록 정리 (30일)
        cutoff = (today.toordinal() - 30)
        self.pos = {k: v for k, v in self.pos.items()
                    if v["status"] in ("waiting", "entered") or date.fromisoformat(v["asof"]).toordinal() >= cutoff}
        self.save()

    # ---------- 웹소켓 ----------
    def handle(self, raw: str) -> str | None:
        """수신 메시지 처리. PINGPONG이면 돌려보낼 문자열 반환."""
        if raw[:1] in ("0", "1"):
            tr_id, recs = K.parse_data(raw)
            for r in recs:
                try:
                    if tr_id == K.TR_KR:
                        self.on_tick("kr", r["MKSC_SHRN_ISCD"], float(r["STCK_PRPR"]), float(r["STCK_OPRC"]),
                                     float(r["STCK_HGPR"]), float(r["STCK_LWPR"]), r["STCK_CNTG_HOUR"])
                    elif tr_id == K.TR_US:
                        self.on_tick("us", r["RSYM"], float(r["LAST"]), float(r["OPEN"] or 0),
                                     float(r["HIGH"] or 0), float(r["LOW"] or 0), r["XHMS"])
                except (KeyError, ValueError) as e:
                    log.debug("파싱 실패 %s: %s", e, r)
            return None
        try:
            d = json.loads(raw)
        except ValueError:
            return None
        hdr = d.get("header", {})
        if hdr.get("tr_id") == "PINGPONG":
            return raw
        body = d.get("body", {})
        if body.get("rt_cd") not in (None, "0"):
            log.warning("구독 오류 %s %s: %s", hdr.get("tr_id"), hdr.get("tr_key"), body.get("msg1"))
        else:
            log.info("%s %s: %s", hdr.get("tr_id"), hdr.get("tr_key"), body.get("msg1"))
        return None

    async def run_session(self, market: str) -> None:
        s = SESSIONS[market]
        self.load_plans(market)
        self.resolve_keys(market)
        keys = sorted({p["key"] for p in self.active(market) if p["key"]})[:K.MAX_SUBS]
        if not keys:
            log.info("[%s] 감시할 후보 없음 → 5분 후 다시 확인", market)
            await asyncio.sleep(300)
            return
        log.info("[%s] 감시 시작: %s", market, ", ".join(keys))
        tr_id = K.TR_KR if market == "kr" else K.TR_US
        backoff = 3
        while self.now(market).time() < s["close"]:
            try:
                approval = self.client.approval_key(refresh=True)
                async with websockets.connect(self.client.ws_url, ping_interval=None, open_timeout=15) as ws:
                    for k in keys:
                        await ws.send(K.subscribe_msg(approval, tr_id, k))
                        await asyncio.sleep(0.05)
                    backoff = 3
                    while self.now(market).time() < s["close"]:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            raw = None
                        if raw:
                            echo = self.handle(raw if isinstance(raw, str) else raw.decode())
                            if echo:
                                await ws.send(echo)
                                await ws.pong(echo.encode())
                        self.on_clock(market)
                        if not self.active(market):
                            log.info("[%s] 모든 후보 처리 완료", market)
                            break
                    break
            except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as e:
                log.warning("[%s] 웹소켓 끊김(%s) → %d초 후 재접속", market, e.__class__.__name__, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
        while self.now(market).time() < s["close"]:  # 조기 종료해도 마감까지 기다렸다 요약
            await asyncio.sleep(30)
        self.end_session(market)

    async def forever(self) -> None:
        self.alert("✅ 실시간 감시 시작 (한국투자증권)")
        while True:
            ran = False
            for m, s in SESSIONS.items():
                now = self.now(m)
                if now.weekday() < 5 and s["pre"] <= now.time() < s["close"]:
                    try:
                        await self.run_session(m)
                    except Exception as e:  # noqa: BLE001 - 상시 실행이므로 죽지 않게
                        log.exception("[%s] 세션 오류", m)
                        self.alert(f"⚠️ 실시간 감시 오류 [{m.upper()}]: {e.__class__.__name__} — 1분 후 재시도")
                        await asyncio.sleep(60)
                    ran = True
            if not ran:
                await asyncio.sleep(30)


async def check(market: str, code: str) -> None:
    c = K.KIS()
    print(f"[1/4] 서버: {c.env} ({c.rest})")
    key = c.approval_key()
    print(f"[2/4] 웹소켓 접속키 발급 OK ({key[:6]}…)")
    if market == "kr":
        print(f"      REST 현재가: {c.kr_price(code)}")
        tr_id, tr_key = K.TR_KR, code
    else:
        ex = c.us_exchange(code)
        print(f"      거래소: {ex}, REST 현재가: {c.us_price(code, ex) if ex else '-'}")
        tr_id, tr_key = K.TR_US, K.us_tr_key(code, ex or "NAS")
    async with websockets.connect(c.ws_url, ping_interval=None, open_timeout=15) as ws:
        await ws.send(K.subscribe_msg(key, tr_id, tr_key))
        print(f"[3/4] 웹소켓 접속 OK, {tr_key} 구독 → 30초간 체결 수신 (장중에만 들어옵니다)")
        n, end = 0, asyncio.get_event_loop().time() + 30
        while asyncio.get_event_loop().time() < end and n < 5:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=end - asyncio.get_event_loop().time())
            except asyncio.TimeoutError:
                break
            if raw[:1] in ("0", "1"):
                _, recs = K.parse_data(raw)
                for r in recs:
                    n += 1
                    print("      체결:", {k: r.get(k) for k in ("STCK_CNTG_HOUR", "STCK_PRPR", "XHMS", "LAST") if k in r})
            else:
                print("      응답:", raw[:200])
    print("[4/4] 텔레그램 테스트:", "성공" if telegram("✅ 실시간 감시 연결 점검 메시지") else "미설정/실패")


def load_env(path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")) -> None:
    """같은 폴더의 .env 를 읽어 환경변수로 (이미 설정된 값은 유지)."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if v.strip():
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


def main():
    load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="실시간 감시 + 텔레그램 알림 (한국투자증권)")
    ap.add_argument("--check", action="store_true", help="연결 점검")
    ap.add_argument("--status", action="store_true", help="감시 중인 포지션 출력")
    ap.add_argument("--market", choices=["kr", "us"], default="kr")
    ap.add_argument("--code", default=None, help="점검용 종목 (기본: 005930 / AAPL)")
    a = ap.parse_args()
    if a.status:
        for p in Monitor(None, send=lambda t: None).pos.values():
            print(f"{p['status']:>8} {p['market']} {p['name']:<12} {p['mode']:<8} 진입 {p['entry']} "
                  f"익절 {p['tp']} 손절 {p['sl']} 결과 {p['reason'] or ''} {p['ret'] if p['ret'] is not None else ''}")
        return
    if a.check:
        asyncio.run(check(a.market, a.code or ("005930" if a.market == "kr" else "AAPL")))
        return
    asyncio.run(Monitor(K.KIS()).forever())


if __name__ == "__main__":
    main()
