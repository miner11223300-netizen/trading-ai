import sqlite3
import os
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd

DB = Path("trading_ai_paper.db")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
START_CAPITAL = 2000.0
RISK_PCT = 0.05
FEE_RATE = 0.001

class _PGCursor:
    def __init__(self, cursor):
        self.cur = cursor
    def execute(self, sql, params=()):
        if "SELECT changes()" in sql:
            self._changes = getattr(self, "_last_rowcount", 0)
            return self
        sql = sql.replace("INSERT OR REPLACE INTO experiment", "INSERT INTO experiment")
        sql = sql.replace("?", "%s")
        self.cur.execute(sql, params)
        self._last_rowcount = self.cur.rowcount
        return self
    def fetchone(self):
        if hasattr(self, "_changes"):
            value = self._changes
            del self._changes
            return {"changes()": value}
        row = self.cur.fetchone()
        return row
    def fetchall(self): return self.cur.fetchall()
    @property
    def rowcount(self): return self.cur.rowcount

class _PGConnection:
    def __init__(self):
        import psycopg2
        from psycopg2.extras import RealDictCursor
        self.conn = psycopg2.connect(DATABASE_URL, connect_timeout=15, sslmode="require")
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
        self.wrapper = _PGCursor(self.cur)
    def __enter__(self): return self
    def __exit__(self, typ, value, tb):
        if typ is None: self.conn.commit()
        else: self.conn.rollback()
        self.cur.close(); self.conn.close()
    def execute(self, sql, params=()):
        if "PRAGMA table_info" in sql:
            self.cur.execute("""SELECT column_name AS name FROM information_schema.columns
                WHERE table_schema='public' AND table_name='experiment_trades'""")
            return self.cur
        return self.wrapper.execute(sql, params)

@contextmanager
def _conn():
    if DATABASE_URL:
        c = _PGConnection()
        try:
            yield c
        finally:
            pass
    else:
        c = sqlite3.connect(DB)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

def init_experiment():
    with _conn() as c:
        if DATABASE_URL:
            c.execute("""CREATE TABLE IF NOT EXISTS experiment (
                id INTEGER PRIMARY KEY CHECK(id=1), started_at TEXT NOT NULL,
                ends_at TEXT NOT NULL, starting_capital DOUBLE PRECISION NOT NULL,
                risk_pct DOUBLE PRECISION NOT NULL, fee_rate DOUBLE PRECISION NOT NULL, status TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS experiment_trades (
                id BIGSERIAL PRIMARY KEY, signal_key TEXT UNIQUE, opened_at TEXT NOT NULL,
                closed_at TEXT, venue TEXT, symbol TEXT, asset_class TEXT, side TEXT, timeframe TEXT,
                entry DOUBLE PRECISION, stop_loss DOUBLE PRECISION, take_profit DOUBLE PRECISION,
                quantity DOUBLE PRECISION, risk_usd DOUBLE PRECISION, entry_fee DOUBLE PRECISION,
                exit_fee DOUBLE PRECISION DEFAULT 0, exit_price DOUBLE PRECISION, gross_pnl DOUBLE PRECISION,
                net_pnl DOUBLE PRECISION, status TEXT DEFAULT 'OPEN', exit_reason TEXT, score INTEGER,
                opportunity_score DOUBLE PRECISION, trend_4h TEXT, trend_1d TEXT,
                last_price DOUBLE PRECISION, last_price_at TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS experiment_events (
                id BIGSERIAL PRIMARY KEY, created_at TEXT NOT NULL, event_type TEXT NOT NULL, message TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS worker_state (
                state_key TEXT PRIMARY KEY, state_value TEXT NOT NULL
            )""")
        else:
            c.execute("""CREATE TABLE IF NOT EXISTS experiment (
                id INTEGER PRIMARY KEY CHECK(id=1), started_at TEXT NOT NULL,
                ends_at TEXT NOT NULL, starting_capital REAL NOT NULL,
                risk_pct REAL NOT NULL, fee_rate REAL NOT NULL, status TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS experiment_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, signal_key TEXT UNIQUE,
                opened_at TEXT NOT NULL, closed_at TEXT, venue TEXT, symbol TEXT,
                asset_class TEXT, side TEXT, timeframe TEXT, entry REAL,
                stop_loss REAL, take_profit REAL, quantity REAL, risk_usd REAL,
                entry_fee REAL, exit_fee REAL DEFAULT 0, exit_price REAL,
                gross_pnl REAL, net_pnl REAL, status TEXT DEFAULT 'OPEN',
                exit_reason TEXT, score INTEGER, opportunity_score REAL,
                trend_4h TEXT, trend_1d TEXT, last_price REAL, last_price_at TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS experiment_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                event_type TEXT NOT NULL, message TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS worker_state (
                state_key TEXT PRIMARY KEY, state_value TEXT NOT NULL
            )""")
        cols={r["name"] for r in c.execute("PRAGMA table_info(experiment_trades)").fetchall()}
        if "last_price" not in cols: c.execute("ALTER TABLE experiment_trades ADD COLUMN last_price DOUBLE PRECISION")
        if "last_price_at" not in cols: c.execute("ALTER TABLE experiment_trades ADD COLUMN last_price_at TEXT")

def active_experiment():
    init_experiment()
    with _conn() as c:
        r=c.execute("SELECT * FROM experiment WHERE id=1").fetchone()
        return dict(r) if r else None

def start_experiment():
    init_experiment()
    now=datetime.now(timezone.utc)
    end=now+timedelta(days=10)
    with _conn() as c:
        c.execute("DELETE FROM experiment_trades")
        c.execute("DELETE FROM experiment_events")
        c.execute("""INSERT INTO experiment
        (id,started_at,ends_at,starting_capital,risk_pct,fee_rate,status)
        VALUES (1,?,?,?,?,?,?) ON CONFLICT (id) DO UPDATE SET
        started_at=EXCLUDED.started_at, ends_at=EXCLUDED.ends_at,
        starting_capital=EXCLUDED.starting_capital, risk_pct=EXCLUDED.risk_pct,
        fee_rate=EXCLUDED.fee_rate, status=EXCLUDED.status""",
        (now.isoformat(),end.isoformat(),START_CAPITAL,RISK_PCT,FEE_RATE,"ACTIVE"))
    log_event("START",f"Испытание начато: виртуальный капитал $2,000; срок 10 дней; риск 5% на сделку ($100 при старте); комиссия 0.10% на сторону. Завершение {end.isoformat()}")
    return active_experiment()

def log_event(event_type,message):
    init_experiment()
    with _conn() as c:
        c.execute("INSERT INTO experiment_events(created_at,event_type,message) VALUES(?,?,?)",
                  (datetime.now(timezone.utc).isoformat(),event_type,str(message)))

def _equity(c):
    exp=c.execute("SELECT starting_capital FROM experiment WHERE id=1").fetchone()
    start=float(exp["starting_capital"]) if exp else START_CAPITAL
    realized=c.execute("SELECT COALESCE(SUM(net_pnl),0) AS p FROM experiment_trades WHERE status='CLOSED'").fetchone()["p"]
    return start+float(realized)

def open_top_trades(rows, max_open=10):
    init_experiment()
    opened=[]
    with _conn() as c:
        exp=c.execute("SELECT * FROM experiment WHERE id=1").fetchone()
        if not exp or exp["status"]!="ACTIVE": return opened
        now=datetime.now(timezone.utc)
        if now >= datetime.fromisoformat(exp["ends_at"]):
            c.execute("UPDATE experiment SET status='COMPLETED' WHERE id=1")
            return opened
        equity=_equity(c)
        open_count=c.execute("SELECT COUNT(*) n FROM experiment_trades WHERE status='OPEN'").fetchone()["n"]
        slots=max(0,max_open-int(open_count))
        iterable = (row for _, row in rows.iterrows()) if isinstance(rows, pd.DataFrame) else rows
        for r in iterable:
            if slots<=0: break
            d=r.to_dict() if hasattr(r,"to_dict") else dict(r)
            side=d.get("side")
            if side not in ("LONG","SHORT") or not d.get("mtf_confirmed"): continue
            symbol=str(d.get("symbol",""))
            key=f"{d.get('venue')}|{symbol}|{d.get('timeframe')}|{side}|{d.get('bar_time')}"
            if c.execute("SELECT 1 FROM experiment_trades WHERE signal_key=?",(key,)).fetchone(): continue
            entry=float(d["entry"]); sl=float(d["stop_loss"]); tp=float(d["take_profit"])
            dist=abs(entry-sl)
            if entry<=0 or dist<=0: continue
            risk=equity*float(exp["risk_pct"])
            qty=risk/dist
            notional=qty*entry
            fee=notional*float(exp["fee_rate"])
            # Size strictly to the requested cash risk. This can imply simulated
            # leverage/margin when the stop is tight; no real orders are sent.
            if qty<=0: continue
            c.execute("""INSERT INTO experiment_trades
            (signal_key,opened_at,venue,symbol,asset_class,side,timeframe,entry,stop_loss,take_profit,
             quantity,risk_usd,entry_fee,status,score,opportunity_score,trend_4h,trend_1d)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN',?,?,?,?) ON CONFLICT (signal_key) DO NOTHING""",
            (key,now.isoformat(),d.get("venue"),symbol,d.get("asset_class"),side,d.get("timeframe"),
             entry,sl,tp,qty,risk,qty*entry*float(exp["fee_rate"]),int(d.get("score",0)),
             float(d.get("opportunity_score",0)),d.get("trend_4h"),d.get("trend_1d")))
            if c.execute("SELECT changes()").fetchone()[0]:
                opened.append({"symbol":symbol,"side":side,"entry":entry,"sl":sl,"tp":tp,"risk":risk,"qty":qty})
                slots-=1
        return opened

def update_prices(price_rows):
    """Update intraday positions. Never evaluate a newly opened trade on its signal candle.
    Any position still open when the UTC trading date changes is force-closed at the
    first observed close of the new UTC date (no position is intentionally held overnight).
    SL wins if both SL and TP are touched in one candle.
    """
    init_experiment()
    closed=[]
    with _conn() as c:
        exp=c.execute("SELECT fee_rate FROM experiment WHERE id=1").fetchone()
        if not exp: return closed
        fee_rate=float(exp["fee_rate"])
        for p in price_rows:
            sym=str(p["symbol"]); high=float(p["high"]); low=float(p["low"])
            close_px=float(p.get("close", p.get("high", p.get("low"))))
            candle_time=p.get("bar_time")
            if candle_time is None or pd.isna(candle_time):
                candle_dt=datetime.now(timezone.utc)
            elif isinstance(candle_time, (int, float)):
                candle_dt=datetime.fromtimestamp(float(candle_time), timezone.utc)
            else:
                candle_dt=pd.to_datetime(candle_time, utc=True).to_pydatetime()
            venue = p.get("venue")
            if venue:
                trades=c.execute("SELECT * FROM experiment_trades WHERE status='OPEN' AND symbol=? AND venue=?",(sym,str(venue))).fetchall()
            else:
                trades=c.execute("SELECT * FROM experiment_trades WHERE status='OPEN' AND symbol=?",(sym,)).fetchall()
            for t in trades:
                opened_dt=datetime.fromisoformat(t["opened_at"])
                if opened_dt.tzinfo is None: opened_dt=opened_dt.replace(tzinfo=timezone.utc)
                # Do not use the already-completed candle that produced the entry signal.
                if candle_dt <= opened_dt:
                    continue
                exit_px=None; reason=None
                if candle_dt.date() != opened_dt.astimezone(timezone.utc).date():
                    # Use the last observed close from the opening UTC day, not a next-day price.
                    exit_px=float(t["last_price"]) if t["last_price"] is not None else close_px
                    reason="END OF DAY (UTC)"
                elif t["side"]=="LONG":
                    if low<=t["stop_loss"]: exit_px=t["stop_loss"]; reason="STOP LOSS"
                    elif high>=t["take_profit"]: exit_px=t["take_profit"]; reason="TAKE PROFIT"
                else:
                    if high>=t["stop_loss"]: exit_px=t["stop_loss"]; reason="STOP LOSS"
                    elif low<=t["take_profit"]: exit_px=t["take_profit"]; reason="TAKE PROFIT"
                if exit_px is None:
                    c.execute("UPDATE experiment_trades SET last_price=?,last_price_at=? WHERE id=?",
                              (close_px,candle_dt.isoformat(),t["id"]))
                    continue
                gross=(exit_px-t["entry"])*t["quantity"]*(1 if t["side"]=="LONG" else -1)
                exit_fee=abs(exit_px*t["quantity"])*fee_rate
                net=gross-float(t["entry_fee"])-exit_fee
                c.execute("""UPDATE experiment_trades SET status='CLOSED',closed_at=?,exit_price=?,
                gross_pnl=?,exit_fee=?,net_pnl=?,exit_reason=? WHERE id=?""",
                (datetime.now(timezone.utc).isoformat(),exit_px,gross,exit_fee,net,reason,t["id"]))
                closed.append({"symbol":sym,"side":t["side"],"reason":reason,"exit":exit_px,"net_pnl":net})
    return closed

def experiment_report():
    init_experiment()
    with _conn() as c:
        exp=c.execute("SELECT * FROM experiment WHERE id=1").fetchone()
        if not exp: return None, pd.DataFrame(), pd.DataFrame()
        trades=pd.read_sql_query("SELECT * FROM experiment_trades ORDER BY id",c)
        events=pd.read_sql_query("SELECT * FROM experiment_events ORDER BY id",c)
        realized=float(trades.loc[trades.status=="CLOSED","net_pnl"].fillna(0).sum()) if not trades.empty else 0.0
        closed=trades[trades.status=="CLOSED"] if not trades.empty else trades
        wins=int((closed.net_pnl>0).sum()) if not closed.empty else 0
        losses=int((closed.net_pnl<=0).sum()) if not closed.empty else 0
        gross_win=float(closed.loc[closed.net_pnl>0,"net_pnl"].sum()) if not closed.empty else 0
        gross_loss=abs(float(closed.loc[closed.net_pnl<0,"net_pnl"].sum())) if not closed.empty else 0
        eq=float(exp["starting_capital"])+realized
        summary={"started_at":exp["started_at"],"ends_at":exp["ends_at"],"status":exp["status"],
          "starting_capital":float(exp["starting_capital"]),"ending_equity_realized":eq,
          "net_pnl":realized,"roi_pct":realized/float(exp["starting_capital"])*100,
          "closed_trades":len(closed),"open_trades":int((trades.status=="OPEN").sum()) if not trades.empty else 0,
          "wins":wins,"losses":losses,"win_rate":wins/len(closed)*100 if len(closed) else 0,
          "profit_factor":gross_win/gross_loss if gross_loss else (float("inf") if gross_win else 0),
          "signals_opened":len(trades),"fee_total":float(trades.entry_fee.fillna(0).sum()+trades.exit_fee.fillna(0).sum()) if not trades.empty else 0}
        return summary,trades,events

def finish_if_due():
    exp=active_experiment()
    if exp and exp["status"]=="ACTIVE" and datetime.now(timezone.utc)>=datetime.fromisoformat(exp["ends_at"]):
        with _conn() as c: c.execute("UPDATE experiment SET status='COMPLETED' WHERE id=1")
        log_event("FINISH","Десятидневное испытание завершено. Сформирован итоговый отчёт.")
        return True
    return False


def get_worker_state(key):
    init_experiment()
    with _conn() as c:
        row = c.execute("SELECT state_value FROM worker_state WHERE state_key=?", (key,)).fetchone()
        return row["state_value"] if row else None

def set_worker_state(key, value):
    init_experiment()
    with _conn() as c:
        if DATABASE_URL:
            c.execute("""INSERT INTO worker_state(state_key,state_value) VALUES(?,?)
                ON CONFLICT(state_key) DO UPDATE SET state_value=EXCLUDED.state_value""", (key, str(value)))
        else:
            c.execute("INSERT OR REPLACE INTO worker_state(state_key,state_value) VALUES(?,?)", (key, str(value)))


def get_open_positions():
    """Return open positions from the same configured database as the app."""
    init_experiment()
    with _conn() as c:
        rows = c.execute("SELECT DISTINCT venue, symbol, asset_class FROM experiment_trades WHERE status='OPEN'").fetchall()
        return [dict(r) for r in rows]
