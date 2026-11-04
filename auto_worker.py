"""
Trading AI autonomous Paper Trading worker.
GitHub Actions invokes this job every 5 minutes. During each job this process
polls open positions once per minute; market signal scans are independent and
run only when their 15-minute interval is due.
No real orders are ever sent.
"""
import os
import time
import threading
import traceback
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import ccxt
import yfinance as yf

from core.scanner import scan_universe
from core.experiment import (
    DB, init_experiment, active_experiment, update_prices, open_top_trades,
    experiment_report, finish_if_due, log_event, get_worker_state, set_worker_state, get_open_positions
)

MONITOR_INTERVAL = 60
JOB_RUNTIME_SECONDS = int(os.getenv("WORKER_RUNTIME_SECONDS", "240"))
SIGNAL_SCAN_INTERVAL = int(os.getenv("SIGNAL_SCAN_INTERVAL_SECONDS", "900"))
MAX_OPEN = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
SCAN_TIMEFRAME = os.getenv("SCAN_TIMEFRAME", "1h")
SCAN_LIMIT = 180

def telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("Telegram secrets are not configured")
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": message},
            timeout=15,
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        print("Telegram error:", str(exc)[:180])
        return False

def state_get(key):
    return get_worker_state(key)

def state_set(key, value):
    set_worker_state(key, value)

def open_positions():
    return [(r.get("venue"), r.get("symbol"), r.get("asset_class")) for r in get_open_positions()]

def fetch_minute_prices():
    """Fetch the current 1m OHLCV candle for every open position."""
    rows = []
    exchanges = {}
    positions = open_positions()
    for venue, symbol, asset_class in positions:
        try:
            venue_key = str(venue or "").upper()
            if venue_key in ("YAHOO", "GOLD") or "GOLD" in str(asset_class).upper():
                # Yahoo 1m history is a delayed market-data proxy, not broker execution data.
                ticker = "XAUUSD=X"
                hist = yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=False)
                if hist is None or hist.empty:
                    ticker = "GC=F"
                    hist = yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=False)
                if hist is None or hist.empty:
                    raise RuntimeError("Yahoo Finance did not return 1m gold candles")
                candle = hist.iloc[-1]
                bar_time = hist.index[-1]
                if getattr(bar_time, "tzinfo", None) is None:
                    bar_time = bar_time.tz_localize("UTC")
                rows.append({
                    "venue": venue_key, "symbol": symbol,
                    "high": float(candle["High"]), "low": float(candle["Low"]),
                    "close": float(candle["Close"]), "bar_time": bar_time.to_pydatetime()
                })
            else:
                ex = exchanges.get(venue_key)
                if ex is None:
                    exchange_id = venue_key.lower()
                    if not hasattr(ccxt, exchange_id):
                        raise RuntimeError(f"Unknown CCXT venue {venue_key}")
                    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 20000})
                    exchanges[venue_key] = ex
                candles = ex.fetch_ohlcv(symbol, timeframe="1m", limit=2)
                if not candles:
                    raise RuntimeError("Exchange returned no 1m candles")
                ts, op, hi, lo, close, volume = candles[-1]
                rows.append({
                    "venue": venue_key, "symbol": symbol,
                    "high": float(hi), "low": float(lo), "close": float(close),
                    "bar_time": pd.to_datetime(ts, unit="ms", utc=True)
                })
        except Exception as exc:
            print(f"MINUTE MONITOR ERROR {venue} {symbol}: {str(exc)[:180]}")
    return rows

def monitor_open_positions():
    exp = active_experiment()
    if not exp or exp["status"] != "ACTIVE":
        return 0, 0
    prices = fetch_minute_prices()
    closed = update_prices(prices) if prices else []
    for trade in closed:
        message = (
            f"🧪 Paper CLOSE: {trade['symbol']} {trade['side']} · {trade['reason']} · "
            f"exit {trade['exit']:.8g} · net P&L ${trade['net_pnl']:.2f}"
        )
        telegram(message)
        log_event("CLOSE", message)
    return len(prices), len(closed)

def scan_signals_if_due():
    exp = active_experiment()
    if not exp or exp["status"] != "ACTIVE":
        return False
    now = datetime.now(timezone.utc)
    previous = state_get("last_signal_scan_at")
    if previous:
        try:
            if (now - datetime.fromisoformat(previous)).total_seconds() < SIGNAL_SCAN_INTERVAL:
                return False
        except Exception:
            pass

    scan_df, errors = scan_universe(("binance", "bybit"), 30, 1, SCAN_TIMEFRAME, SCAN_LIMIT)
    if scan_df is None or scan_df.empty:
        detail = "; ".join(errors[:8])
        message = "⚠️ Trading AI: сканер сигналов не получил рыночные данные."
        telegram(message + "\\n" + detail[:2000])
        log_event("SCAN_ERROR", message + " " + detail)
        # Don't update last scan timestamp; retry on next monitor cycle.
        return False

    strong = scan_df[
        (scan_df.side != "WAIT") &
        (scan_df.score >= 5) &
        (scan_df.mtf_confirmed.fillna(False) == True)
    ].sort_values("opportunity_score", ascending=False).head(5).copy()

    opened = open_top_trades(strong, max_open=MAX_OPEN)
    for trade in opened:
        message = (
            f"🧪 Paper OPEN: {trade['symbol']} {trade['side']} · entry {trade['entry']:.8g} · "
            f"SL {trade['sl']:.8g} · TP {trade['tp']:.8g} · risk ${trade['risk']:.2f}"
        )
        telegram(message)
        log_event("OPEN", message)

    state_set("last_signal_scan_at", now.isoformat())
    message = (
        f"🔎 Signal scan ({now.strftime('%Y-%m-%d %H:%M UTC')}): "
        f"инструментов {len(scan_df)}; MTF подтверждено "
        f"{int(scan_df.mtf_confirmed.fillna(False).sum())}; TOP-5 {len(strong)}; "
        f"открыто {len(opened)}; ошибок источников {len(errors)}."
    )
    telegram(message)
    log_event("SCAN", message)
    return True

def finish_if_needed():
    if finish_if_due():
        summary, _, _ = experiment_report()
        message = (
            f"🏁 Испытание завершено. Закрыто сделок: {summary['closed_trades']}; "
            f"Win Rate: {summary['win_rate']:.1f}%; Net P&L: ${summary['net_pnl']:.2f}; "
            f"ROI: {summary['roi_pct']:.2f}%."
        )
        telegram(message)
        log_event("FINISH", message)
        return True
    return False

def main():
    init_experiment()
    exp = active_experiment()
    if not exp or exp["status"] != "ACTIVE":
        print("No active experiment; minute monitor will not start.")
        return

    started = time.monotonic()
    next_monitor = started
    scan_thread = None
    while time.monotonic() - started < JOB_RUNTIME_SECONDS:
        if finish_if_needed():
            return

        now_mono = time.monotonic()
        if now_mono >= next_monitor:
            checked, closed = monitor_open_positions()
            print(f"Minute monitor: checked={checked}, closed={closed}")
            next_monitor = now_mono + MONITOR_INTERVAL

        # Signal discovery runs in a separate thread. Slow exchange scanning must
        # never block the once-per-minute checks of already-open positions.
        if scan_thread is None or not scan_thread.is_alive():
            scan_thread = threading.Thread(
                target=scan_signals_if_due,
                name="signal-scanner",
                daemon=True,
            )
            scan_thread.start()

        finish_if_needed()
        remaining = JOB_RUNTIME_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            break
        time.sleep(min(5, remaining))

    if scan_thread is not None and scan_thread.is_alive():
        print("Signal scanner still running at worker slice end; next Actions slice will retry if scan wasn't persisted.")

    print("Worker slice finished; GitHub Actions will persist SQLite state.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        details = traceback.format_exc()[-2500:]
        print(details)
        telegram("🚨 Trading AI worker error:\\n" + details[-1200:])
        raise
