import time
import numpy as np
import pandas as pd

from core.market_data import normalize
from core.indicators import enrich
from core.regime import detect_regime

STABLES = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDP", "EUR", "GBP"}

def _signal(df, symbol, venue, asset_class, timeframe):
    d = enrich(normalize(df))
    if len(d) < 100:
        return None
    x, prev = d.iloc[-1], d.iloc[-2]
    close = float(x.close)
    atr = float(x.atr) if pd.notna(x.atr) else close * 0.01
    if not np.isfinite(atr) or atr <= 0:
        atr = close * 0.01

    long_score = short_score = 0
    reasons = []
    if x.ema20 > x.ema50 > x.ema200:
        long_score += 2; reasons.append("EMA20>EMA50>EMA200")
    elif x.ema20 < x.ema50 < x.ema200:
        short_score += 2; reasons.append("EMA20<EMA50<EMA200")
    if x.macd > x.macd_signal:
        long_score += 1; reasons.append("MACD bullish")
    elif x.macd < x.macd_signal:
        short_score += 1; reasons.append("MACD bearish")
    if 50 <= x.rsi <= 68:
        long_score += 1; reasons.append("RSI supports upside")
    elif 32 <= x.rsi <= 50:
        short_score += 1; reasons.append("RSI supports downside")
    if close > float(x.ema20) and close > float(prev.close):
        long_score += 1
    if close < float(x.ema20) and close < float(prev.close):
        short_score += 1

    vol_mean = float(d.volume.tail(20).mean())
    vol_ratio = float(x.volume / max(vol_mean, 1e-12)) if x.volume > 0 else 1.0
    if vol_ratio >= 1.4:
        if close > float(prev.close):
            long_score += 1
        elif close < float(prev.close):
            short_score += 1
        reasons.append(f"volume {vol_ratio:.1f}x")

    regime = detect_regime(d)
    side, score = "WAIT", max(long_score, short_score)
    if long_score >= 4 and long_score - short_score >= 2 and regime["regime"] not in ("DOWNTREND", "HIGH_VOLATILITY"):
        side = "LONG"
    elif short_score >= 4 and short_score - long_score >= 2 and regime["regime"] not in ("UPTREND", "HIGH_VOLATILITY"):
        side = "SHORT"

    sl = tp = None
    if side == "LONG":
        sl, tp = close - 1.5 * atr, close + 3 * atr
    elif side == "SHORT":
        sl, tp = close + 1.5 * atr, close - 3 * atr

    # Ranking score: directional agreement + trend regime + volume confirmation.
    directional = max(long_score, short_score)
    alignment = 1 if ((side == "LONG" and regime["regime"] == "UPTREND") or
                      (side == "SHORT" and regime["regime"] == "DOWNTREND")) else 0
    opportunity_score = min(100, int(directional * 13 + min(vol_ratio, 3) * 8 + alignment * 12))
    return {
        "venue": venue, "asset_class": asset_class, "symbol": symbol, "timeframe": timeframe,
        "bar_time": str(x.timestamp), "price": close, "high": float(x.high), "low": float(x.low), "side": side, "score": score,
        "opportunity_score": opportunity_score, "long_score": long_score, "short_score": short_score,
        "entry": close, "stop_loss": sl, "take_profit": tp, "rsi": float(x.rsi),
        "atr_pct": atr / close * 100, "volume_ratio": vol_ratio,
        "regime": regime["regime"], "reasons": "; ".join(reasons)
    }

def _higher_tf_side(df):
    """Return trend direction only when EMA structure and MACD agree."""
    try:
        d = enrich(normalize(df))
        if len(d) < 100:
            return "UNKNOWN"
        x = d.iloc[-1]
        if x.ema20 > x.ema50 > x.ema200 and x.macd > x.macd_signal:
            return "LONG"
        if x.ema20 < x.ema50 < x.ema200 and x.macd < x.macd_signal:
            return "SHORT"
        return "NEUTRAL"
    except Exception:
        return "UNKNOWN"

def _confirm_higher_timeframes(exchange, symbol, side, limit=220):
    """Require both 4h and 1d trend to agree with the entry timeframe."""
    states = {}
    for tf in ("4h", "1d"):
        try:
            rows = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            if not rows:
                states[tf] = "UNKNOWN"
                continue
            d = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            d["timestamp"] = pd.to_datetime(d.timestamp, unit="ms", utc=True)
            states[tf] = _higher_tf_side(d)
        except Exception:
            states[tf] = "UNKNOWN"
    confirmed = side in ("LONG", "SHORT") and states["4h"] == side and states["1d"] == side
    return states["4h"], states["1d"], bool(confirmed)

def scan_crypto(exchange_id, top_n=30, timeframe="1h", limit=180, candidates=60, pause=0.03, exclude_providers=None):
    """Scan liquid USDT spot markets, then rank by technical opportunity, not volume alone."""
    import ccxt
    # Binance/Bybit may be geo-blocked from some cloud regions. Use exchanges with
    # public spot endpoints as fallbacks; no API keys or private endpoints required.
    fallback_map = {
        "binance": ["binance", "kraken", "kucoin", "coinbase"],
        "bybit": ["bybit", "kraken", "kucoin", "coinbase"],
        "kraken": ["kraken", "kucoin", "coinbase"],
        "kucoin": ["kucoin", "kraken", "coinbase"],
        "coinbase": ["coinbase", "kraken", "kucoin"],
    }
    providers = fallback_map.get(exchange_id.lower(), [exchange_id.lower(), "kraken", "kucoin"])
    excluded = set(exclude_providers or [])
    last_error = None
    ex = None
    actual_exchange_id = None
    tickers = None
    for provider in providers:
        if provider in excluded:
            continue
        try:
            cls = getattr(ccxt, provider)
            candidate_ex = cls({"enableRateLimit": True, "timeout": 25000})
            candidate_ex.load_markets()
            candidate_tickers = candidate_ex.fetch_tickers()
            ex = candidate_ex
            tickers = candidate_tickers
            actual_exchange_id = provider
            break
        except Exception as exc:
            last_error = exc
    if ex is None:
        raise RuntimeError(f"Не удалось подключиться к биржам {providers}: {str(last_error)[:180]}")
    exchange_id = actual_exchange_id
    markets = ex.markets
    ranked = []
    for sym, t in tickers.items():
        m = markets.get(sym, {})
        is_spot = m.get("spot")
        if is_spot is None:
            is_spot = not (m.get("swap") or m.get("future") or m.get("option"))
        if not m.get("active", True) or not is_spot or not sym.endswith("/USDT"):
            continue
        base = sym.split("/")[0]
        if base in STABLES or any(tag in base for tag in ("UP", "DOWN", "BULL", "BEAR")):
            continue
        qv = t.get("quoteVolume")
        if qv is None and t.get("baseVolume") and t.get("last"):
            qv = float(t["baseVolume"]) * float(t["last"])
        if qv is not None and np.isfinite(float(qv)) and float(qv) > 0:
            ranked.append((sym, float(qv), float(t.get("percentage") or 0)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    results, errors = [], []
    for sym, qv, change24 in ranked[:candidates]:
        try:
            rows = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            if rows:
                d = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                d["timestamp"] = pd.to_datetime(d.timestamp, unit="ms", utc=True)
                r = _signal(d, sym, exchange_id.upper(), "CRYPTO", timeframe)
                if r:
                    h4, d1, confirmed = _confirm_higher_timeframes(ex, sym, r["side"])
                    r["trend_4h"] = h4
                    r["trend_1d"] = d1
                    r["mtf_confirmed"] = confirmed
                    r["quote_volume_24h"] = qv
                    r["change_24h_pct"] = change24
                    if confirmed:
                        r["opportunity_score"] = min(100, int(r["opportunity_score"] + 20))
                    results.append(r)
        except Exception as exc:
            errors.append(f"{sym} ({exchange_id}): {str(exc)[:120]}")
        if pause:
            time.sleep(pause)
    frame = pd.DataFrame(results)
    if not frame.empty:
        frame = frame.sort_values(["opportunity_score", "quote_volume_24h"], ascending=False).head(top_n)
    return frame, errors

def scan_gold(timeframe="1h", limit=180):
    """XAU/USD spot proxy from Yahoo Finance (XAUUSD=X), not a futures contract."""
    import yfinance as yf
    interval = "1h" if timeframe in ("5m", "15m", "1h") else "1d"
    period = "60d" if interval == "1h" else "2y"
    tickers = ["XAUUSD=X", "GC=F"]
    errors = []
    attempts = []
    ticker = None
    d = None
    for candidate in tickers:
        try:
            candidate_data = yf.Ticker(candidate).history(period=period, interval=interval, auto_adjust=False)
            if candidate_data is not None and not candidate_data.empty:
                ticker, d = candidate, candidate_data
                break
            attempts.append(f"{candidate}: пустой ответ Yahoo Finance")
        except Exception as exc:
            attempts.append(f"{candidate}: {str(exc)[:120]}")
    if d is None or d.empty:
        errors.append("XAU/USD: не удалось получить ни спотовый XAUUSD=X, ни резервный GC=F. " + "; ".join(attempts))
        return pd.DataFrame(), errors
    try:
        d = d.reset_index()
        d.columns = [str(c).lower().replace(" ", "_") for c in d.columns]
        d = d.rename(columns={"datetime": "timestamp", "date": "timestamp"})
        d = d[["timestamp", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"]).tail(limit)
        d["volume"] = d["volume"].fillna(0)
        r = _signal(d, ticker, "YAHOO", f"GOLD: XAU/USD proxy ({ticker})", timeframe)
        if r:
            r["display_name"] = "Золото XAU/USD"
            try:
                h1 = yf.Ticker(ticker).history(period="60d", interval="1h", auto_adjust=False)
                d1 = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=False)
                h4 = h1.resample("4h").agg({"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"}).dropna() if h1 is not None and not h1.empty else pd.DataFrame()
                r["trend_4h"] = _higher_tf_side(h4.reset_index()) if not h4.empty else "UNKNOWN"
                r["trend_1d"] = _higher_tf_side(d1.reset_index()) if d1 is not None and not d1.empty else "UNKNOWN"
            except Exception:
                r["trend_4h"] = r["trend_1d"] = "UNKNOWN"
            r["mtf_confirmed"] = bool(r["side"] in ("LONG", "SHORT") and r["trend_4h"] == r["side"] and r["trend_1d"] == r["side"])
            if r["mtf_confirmed"]:
                r["opportunity_score"] = min(100, int(r["opportunity_score"] + 20))
            r["quote_volume_24h"] = np.nan
            r["change_24h_pct"] = np.nan
            return pd.DataFrame([r]), errors
        raise ValueError("Недостаточно истории для расчёта индикаторов золота")
    except Exception as exc:
        errors.append(f"XAU/USD: {str(exc)[:160]}")
        return pd.DataFrame(), errors

def scan_universe(crypto_venues=("binance", "bybit"), crypto_n=30, commodity_n=1, timeframe="1h", limit=180):
    """Return the 30 highest-ranked crypto opportunities plus one XAU/USD row."""
    frames, errors = [], []
    used_providers = set()
    for venue in crypto_venues:
        try:
            frame, err = scan_crypto(
                venue, top_n=crypto_n, timeframe=timeframe, limit=limit,
                candidates=30, exclude_providers=used_providers
            )
            if not frame.empty:
                frames.append(frame)
                used_providers.update(frame["venue"].dropna().astype(str).str.lower().unique())
            errors.extend(err)
        except Exception as exc:
            errors.append(f"{venue}: {str(exc)[:180]}")

    gold, gold_errors = scan_gold(timeframe=timeframe, limit=limit)
    errors.extend(gold_errors)

    crypto = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not crypto.empty:
        # Keep one best venue observation per pair, then rank the top 30 by technical setup.
        crypto = crypto.sort_values(["opportunity_score", "quote_volume_24h"], ascending=False)
        crypto = crypto.drop_duplicates("symbol").head(crypto_n).copy()
        crypto.insert(0, "rank", range(1, len(crypto) + 1))
    if not gold.empty:
        gold = gold.copy()
        gold["rank"] = "GOLD"
    combined = pd.concat([crypto, gold], ignore_index=True, sort=False) if not crypto.empty or not gold.empty else pd.DataFrame()
    if not combined.empty:
        combined["priority"] = combined["opportunity_score"].fillna(0)
        combined = combined.sort_values(["asset_class", "priority"], ascending=[True, False]).reset_index(drop=True)
    return combined, errors
