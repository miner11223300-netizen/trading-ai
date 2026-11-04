"""Multi-timeframe trend alignment. Resamples only from sufficiently granular OHLCV data."""
from __future__ import annotations
import pandas as pd
import numpy as np
from .indicators import enrich

RULES = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def infer_interval_minutes(df: pd.DataFrame) -> float:
    ts = pd.to_datetime(df.timestamp, utc=True).sort_values().drop_duplicates()
    diffs = ts.diff().dropna().dt.total_seconds().div(60)
    if diffs.empty:
        return float("nan")
    return float(diffs.median())


def resample_ohlcv(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in RULES:
        raise ValueError(f"Неподдерживаемый таймфрейм: {target}")
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True)
    d = d.set_index("timestamp").sort_index()
    rule = RULES[target]
    out = d.resample(rule, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def build_mtf_analysis(df: pd.DataFrame) -> dict:
    """Build 5m/15m/1h/4h/1d frames only where source granularity permits."""
    base_minutes = infer_interval_minutes(df)
    frames, rows = {}, []
    for tf, mins in MINUTES.items():
        if not np.isfinite(base_minutes) or base_minutes > mins:
            rows.append({"Таймфрейм": tf, "Статус": "Недоступен", "Направление": "—", "Причина": f"Исходные свечи {base_minutes:.1f} мин крупнее целевого интервала" if np.isfinite(base_minutes) else "Не удалось определить интервал"})
            continue
        frame = df.copy() if abs(base_minutes-mins) < 0.5 else resample_ohlcv(df, tf)
        if len(frame) < 60:
            rows.append({"Таймфрейм": tf, "Статус": "Недостаточно данных", "Направление": "—", "Причина": f"Доступно {len(frame)} свечей; нужно не менее 60"})
            continue
        enriched = enrich(frame)
        last = enriched.iloc[-1]
        close, e20, e50, e200 = [float(last.get(k, np.nan)) for k in ("close", "ema20", "ema50", "ema200")]
        if np.isfinite(e200) and close > e20 > e50 > e200:
            direction = "Бычий"
        elif np.isfinite(e200) and close < e20 < e50 < e200:
            direction = "Медвежий"
        elif close > e20 > e50:
            direction = "Умеренно бычий"
        elif close < e20 < e50:
            direction = "Умеренно медвежий"
        else:
            direction = "Нейтральный / смешанный"
        frames[tf] = enriched
        rows.append({"Таймфрейм": tf, "Статус": "Готов", "Направление": direction, "Цена": close, "EMA20": e20, "EMA50": e50, "EMA200": e200, "Свечей": len(enriched), "Причина": ""})
    ready = [r for r in rows if r["Статус"] == "Готов"]
    bullish = sum("бычий" in r["Направление"].lower() for r in ready)
    bearish = sum("медвежий" in r["Направление"].lower() for r in ready)
    # Higher timeframes have greater weight; unavailable frames are not silently fabricated.
    weights = {"5m": 1, "15m": 1.5, "1h": 2, "4h": 3, "1d": 4}
    score = sum(weights[r["Таймфрейм"]] * (1 if "бычий" in r["Направление"].lower() else -1 if "медвежий" in r["Направление"].lower() else 0) for r in ready)
    if len(ready) < 2:
        decision = "Недостаточно таймфреймов"
    elif score >= 3:
        decision = "Преимущество LONG"
    elif score <= -3:
        decision = "Преимущество SHORT"
    else:
        decision = "Смешанная картина — WAIT"
    return {"table": pd.DataFrame(rows), "frames": frames, "decision": decision, "weighted_score": float(score), "ready_count": len(ready), "bullish": bullish, "bearish": bearish, "source_interval_minutes": base_minutes}


def analyze_mtf_frames(source_frames: dict) -> dict:
    """Analyze independently fetched timeframe frames; each is already OHLCV."""
    rows, frames = [], {}
    weights = {"5m": 1, "15m": 1.5, "1h": 2, "4h": 3, "1d": 4}
    for tf in ["5m", "15m", "1h", "4h", "1d"]:
        raw = source_frames.get(tf)
        if raw is None or len(raw) < 60:
            rows.append({"Таймфрейм":tf,"Статус":"Недоступен" if raw is None else "Недостаточно данных","Направление":"—","Причина":"Источник не предоставил этот таймфрейм" if raw is None else f"Получено {len(raw)} свечей; требуется минимум 60"})
            continue
        enriched = enrich(raw)
        last = enriched.iloc[-1]
        close, e20, e50, e200 = [float(last.get(k, np.nan)) for k in ("close","ema20","ema50","ema200")]
        if close > e20 > e50 > e200: direction="Бычий"
        elif close < e20 < e50 < e200: direction="Медвежий"
        elif close > e20 > e50: direction="Умеренно бычий"
        elif close < e20 < e50: direction="Умеренно медвежий"
        else: direction="Нейтральный / смешанный"
        frames[tf]=enriched
        rows.append({"Таймфрейм":tf,"Статус":"Готов","Направление":direction,"Цена":close,"EMA20":e20,"EMA50":e50,"EMA200":e200,"Свечей":len(enriched),"Причина":""})
    ready=[r for r in rows if r["Статус"]=="Готов"]
    score=sum(weights[r["Таймфрейм"]]*(1 if "бычий" in r["Направление"].lower() else -1 if "медвежий" in r["Направление"].lower() else 0) for r in ready)
    decision="Недостаточно таймфреймов" if len(ready)<2 else "Преимущество LONG" if score>=3 else "Преимущество SHORT" if score<=-3 else "Смешанная картина — WAIT"
    return {"table":pd.DataFrame(rows),"frames":frames,"decision":decision,"weighted_score":float(score),"ready_count":len(ready)}
