"""
Scheduled Paper Trading worker.
Run by GitHub Actions every 15 minutes; no real orders are ever sent.
"""
import os
import traceback
from datetime import datetime, timezone
import pandas as pd
import requests
import ccxt
import yfinance as yf

from core.scanner import scan_universe
from core.experiment import (
    init_experiment, active_experiment, update_prices, open_top_trades,
    experiment_report, finish_if_due, log_event
)

TIMEFRAME = os.getenv("SCAN_TIMEFRAME", "1h")
MAX_OPEN = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", "15"))

def telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("Telegram secrets are not configured")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": message},
            timeout=20,
        )
        r.raise_for_status()
        return True
    except Exception as exc:
        print("Telegram error:", str(exc)[:200])
        return False

def fetch_open_position_candles():
    """Fetch the latest 1h OHLCV for every open position, even if it left TOP-30."""
    import sqlite3
    from core.experiment import DB
    rows = []
    with sqlite3.connect(DB) as conn:
        positions = conn.execute(
            "SELECT DISTINCT venue, symbol, asset_class FROM experiment_trades WHERE status='OPEN'"
        ).fetchall()
    exchanges = {}
    for venue, symbol, asset_class in positions:
        try:
            if venue in ("BINANCE", "BYBIT"):
                ex = exchanges.get(venue)
                if ex is None:
                    ex = getattr(ccxt, venue.lower())({"enableRateLimit": True, "timeout": 20000})
                    exchanges[venue] = ex
                candles = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=2)
                if candles:
                    ts, op, hi, lo, cl, vol = candles[-1]
                    rows.append({"venue": venue, "symbol": symbol, "high": hi, "low": lo,
                                 "close": cl, "bar_time": pd.to_datetime(ts, unit="ms", utc=True)})
            elif venue == "YAHOO" or "GOLD" in str(asset_class):
                hist = yf.Ticker("XAUUSD=X").history(period="5d", interval="1h")
                if hist is not None and not hist.empty:
                    last = hist.iloc[-1]
                    rows.append({"venue": "YAHOO", "symbol": symbol, "high": float(last["High"]),
                                 "low": float(last["Low"]), "close": float(last["Close"]),
                                 "bar_time": hist.index[-1].to_pydatetime()})
        except Exception as exc:
            print(f"Position price fetch failed {venue} {symbol}: {str(exc)[:160]}")
    return rows

def main():
    init_experiment()
    exp = active_experiment()
    if not exp or exp["status"] != "ACTIVE":
        print("No active 10-day experiment; worker exits without scanning.")
        return
    now = datetime.now(timezone.utc)
    if now >= datetime.fromisoformat(exp["ends_at"]):
        if finish_if_due():
            summary, _, _ = experiment_report()
            msg = (f"🏁 Испытание завершено. Закрыто сделок: {summary['closed_trades']}; "
                   f"Win Rate: {summary['win_rate']:.1f}%; Net P&L: ${summary['net_pnl']:.2f}; "
                   f"ROI: {summary['roi_pct']:.2f}%. Подробный отчёт — в приложении.")
            telegram(msg)
        return

    print(f"Starting scheduled scan at {now.isoformat()}")
    scan_df, errors = scan_universe(("binance", "bybit"), 30, 1, TIMEFRAME, 180)
    if scan_df is None or scan_df.empty:
        message = "⚠️ Trading AI: не удалось получить рыночные данные ни от основного источника, ни от резервных бирж."
        detail = "; ".join(errors[:8])
        telegram(message + "\\n" + detail[:2500])
        log_event("SCAN_ERROR", message + " " + detail)
        return

    # Fetch live OHLCV for all existing positions, not only assets still in TOP-30.
    price_rows = fetch_open_position_candles()
    if price_rows:
        closed = update_prices(price_rows)
    else:
        closed = []

    for t in closed:
        msg = (f"🧪 Paper CLOSE: {t['symbol']} {t['side']} · {t['reason']} · "
               f"exit {t['exit']:.8g} · net P&L ${t['net_pnl']:.2f}")
        telegram(msg); log_event("CLOSE", msg)

    strong = scan_df[
        (scan_df.side != "WAIT") &
        (scan_df.score >= 5) &
        (scan_df.mtf_confirmed.fillna(False) == True)
    ].sort_values("opportunity_score", ascending=False).head(5).copy()

    opened = open_top_trades(strong, max_open=MAX_OPEN)
    for t in opened:
        msg = (f"🧪 Paper OPEN: {t['symbol']} {t['side']} · entry {t['entry']:.8g} · "
               f"SL {t['sl']:.8g} · TP {t['tp']:.8g} · risk ${t['risk']:.2f}")
        telegram(msg); log_event("OPEN", msg)

    summary_msg = (
        f"🔎 Trading AI auto-scan ({now.strftime('%Y-%m-%d %H:%M UTC')}): "
        f"инструментов {len(scan_df)}; MTF подтверждено "
        f"{int(scan_df.mtf_confirmed.fillna(False).sum())}; TOP-5 сигналов {len(strong)}; "
        f"открыто {len(opened)}; закрыто {len(closed)}; ошибок источников {len(errors)}."
    )
    telegram(summary_msg); log_event("SCAN", summary_msg)

    if finish_if_due():
        summary, _, _ = experiment_report()
        msg = (f"🏁 Испытание завершено. Закрыто сделок: {summary['closed_trades']}; "
               f"Win Rate: {summary['win_rate']:.1f}%; Net P&L: ${summary['net_pnl']:.2f}; "
               f"ROI: {summary['roi_pct']:.2f}%. Подробный отчёт — в приложении.")
        telegram(msg)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        details = traceback.format_exc()[-2500:]
        print(details)
        telegram("🚨 Trading AI auto-scan error:\n" + details[-1200:])
        raise
