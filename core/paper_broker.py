import sqlite3
import pandas as pd
from pathlib import Path
DB=Path("trading_ai_paper.db")
def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        symbol TEXT, side TEXT, entry REAL, stop_loss REAL, take_profit REAL,
        quantity REAL, risk_amount REAL, status TEXT DEFAULT 'OPEN',
        exit_price REAL, pnl REAL, exit_reason TEXT)""")
        cols=[x[1] for x in c.execute("PRAGMA table_info(trades)").fetchall()]
        if "exit_reason" not in cols:
            c.execute("ALTER TABLE trades ADD COLUMN exit_reason TEXT")
def create_paper_trade(symbol,plan):
    with sqlite3.connect(DB) as c:
        c.execute("""INSERT INTO trades(symbol,side,entry,stop_loss,take_profit,quantity,risk_amount,status)
        VALUES(?,?,?,?,?,?,?,'OPEN')""",(symbol,plan["side"],plan["entry"],plan["stop_loss"],plan["take_profit"],plan["quantity"],plan["risk_amount"]))
def _close(c,trade_id,exit_price,reason):
    row=c.execute("SELECT side,entry,quantity FROM trades WHERE id=? AND status='OPEN'",(trade_id,)).fetchone()
    if not row: return
    side,entry,qty=row
    pnl=(exit_price-entry)*qty*(1 if side=="LONG" else -1)
    c.execute("UPDATE trades SET status='CLOSED',exit_price=?,pnl=?,exit_reason=? WHERE id=?",
              (exit_price,pnl,reason,trade_id))
def close_paper_trade(trade_id,exit_price):
    with sqlite3.connect(DB) as c: _close(c,trade_id,exit_price,"MANUAL / EMERGENCY")
def monitor_paper_trades(symbol,high,low,last_close):
    # Conservative same-candle rule: if both stop and target are touched, assume stop first.
    closed=[]
    with sqlite3.connect(DB) as c:
        rows=c.execute("SELECT id,side,stop_loss,take_profit FROM trades WHERE status='OPEN' AND symbol=?",(symbol,)).fetchall()
        for tid,side,sl,tp in rows:
            if side=="LONG" and low<=sl:
                _close(c,tid,sl,"STOP LOSS"); closed.append((tid,"STOP LOSS",sl))
            elif side=="LONG" and high>=tp:
                _close(c,tid,tp,"TAKE PROFIT"); closed.append((tid,"TAKE PROFIT",tp))
            elif side=="SHORT" and high>=sl:
                _close(c,tid,sl,"STOP LOSS"); closed.append((tid,"STOP LOSS",sl))
            elif side=="SHORT" and low<=tp:
                _close(c,tid,tp,"TAKE PROFIT"); closed.append((tid,"TAKE PROFIT",tp))
    return closed
def list_trades(open_only=False):
    with sqlite3.connect(DB) as c:
        q="SELECT * FROM trades"
        if open_only: q+=" WHERE status='OPEN'"
        q+=" ORDER BY id DESC"
        return pd.read_sql_query(q,c)
