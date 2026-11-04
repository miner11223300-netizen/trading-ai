import pandas as pd
import ccxt

COLS = ["timestamp","open","high","low","close","volume"]

def normalize(df):
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    aliases = {"date":"timestamp","datetime":"timestamp","time":"timestamp","vol":"volume"}
    d = d.rename(columns={k:v for k,v in aliases.items() if k in d.columns and v not in d.columns})
    missing = [c for c in COLS if c not in d.columns]
    if missing:
        raise ValueError("CSV/источник не содержит: " + ", ".join(missing))
    d = d[COLS].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    for c in COLS[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if len(d) < 80:
        raise ValueError("Для анализа нужно минимум 80 корректных свечей.")
    return d

def load_csv(file):
    return normalize(pd.read_csv(file))

def fetch_crypto(exchange_id, symbol, timeframe="1h", limit=700):
    ex_cls = getattr(ccxt, exchange_id)
    ex = ex_cls({"enableRateLimit": True})
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return normalize(pd.DataFrame(rows, columns=COLS).assign(timestamp=lambda x: pd.to_datetime(x.timestamp, unit="ms", utc=True)))

def fetch_yahoo(ticker, timeframe="1d", limit=700):
    import yfinance as yf
    interval = "1d" if timeframe == "1d" else "1wk"
    period = "5y" if interval == "1d" else "10y"
    d = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if d.empty:
        raise ValueError(f"Yahoo Finance не вернул данные для {ticker}.")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.reset_index()
    d = d.rename(columns={"Date":"timestamp","Datetime":"timestamp","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    return normalize(d[COLS].tail(limit))

def fetch_moex(secid, timeframe="1h", limit=700):
    import requests
    interval = 60 if timeframe == "1h" else 24
    url = f"https://iss.moex.com/iss/engines/futures/markets/forts/securities/{secid}/candles.json"
    params = {"interval": interval, "start": 0, "iss.meta": "off"}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    candles = payload.get("candles", {}).get("data", [])
    if not candles:
        raise ValueError("MOEX ISS не вернул свечи. Проверьте SECID, рынок и доступный интервал.")
    # MOEX candle columns: open, close, high, low, value, volume, begin, end
    names = payload.get("candles", {}).get("columns", [])
    d = pd.DataFrame(candles, columns=names)
    d = d.rename(columns={"begin":"timestamp"})
    d["volume"] = d["volume"] if "volume" in d else 0
    d = d[["timestamp","open","high","low","close","volume"]].tail(limit)
    return normalize(d)
