import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

DB = Path(__file__).resolve().parents[1] / "trading_ai_journal.sqlite3"

def _conn():
    c = sqlite3.connect(DB, timeout=20)
    c.execute("""CREATE TABLE IF NOT EXISTS scanner_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, venue TEXT NOT NULL,
        asset_class TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
        side TEXT NOT NULL, score REAL NOT NULL, entry REAL, stop_loss REAL, take_profit REAL,
        regime TEXT, strategy TEXT, signal_key TEXT UNIQUE, status TEXT DEFAULT 'OPEN',
        exit_price REAL, pnl_pct REAL, notes TEXT)""")
    c.commit()
    return c

def record_signal(row):
    c = _conn()
    try:
        c.execute("""INSERT OR IGNORE INTO scanner_signals
        (created_at,venue,asset_class,symbol,timeframe,side,score,entry,stop_loss,take_profit,regime,strategy,signal_key,notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            datetime.now(timezone.utc).isoformat(), row.get('venue',''), row.get('asset_class',''), row['symbol'],
            row.get('timeframe','1h'), row['side'], float(row.get('score',0)), row.get('entry'), row.get('stop_loss'),
            row.get('take_profit'), row.get('regime',''), row.get('strategy','scanner'), row['signal_key'], row.get('notes','')))
        inserted = c.total_changes > 0
        c.commit()
        return inserted
    finally: c.close()

def list_signals(limit=2000):
    c = _conn()
    try: return pd.read_sql_query("SELECT * FROM scanner_signals ORDER BY id DESC LIMIT ?", c, params=(limit,))
    finally: c.close()

def close_signal(signal_id, exit_price):
    c = _conn()
    try:
        row = c.execute("SELECT side,entry FROM scanner_signals WHERE id=?", (int(signal_id),)).fetchone()
        if not row: return
        side, entry = row
        pnl = ((float(exit_price)/float(entry))-1)*100 * (1 if side=='LONG' else -1)
        c.execute("UPDATE scanner_signals SET status='CLOSED',exit_price=?,pnl_pct=? WHERE id=?", (float(exit_price),pnl,int(signal_id)))
        c.commit()
    finally: c.close()

def performance_summary(df):
    if df.empty: return {"signals":0,"closed":0,"win_rate":0.0,"avg_pnl":0.0,"total_pnl":0.0}
    closed=df[df.status=='CLOSED'].copy()
    if closed.empty: return {"signals":len(df),"closed":0,"win_rate":0.0,"avg_pnl":0.0,"total_pnl":0.0}
    p=pd.to_numeric(closed.pnl_pct,errors='coerce').dropna()
    return {"signals":len(df),"closed":len(p),"win_rate":round(float((p>0).mean()*100),1),"avg_pnl":round(float(p.mean()),3),"total_pnl":round(float(p.sum()),3)}
