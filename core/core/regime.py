import numpy as np
import pandas as pd


def detect_regime(df: pd.DataFrame) -> dict:
    """Heuristic market regime label from OHLCV; not a predictive probability."""
    d = df.copy()
    if len(d) < 60:
        return {"regime": "INSUFFICIENT_DATA", "description": "Нужно минимум 60 свечей", "volatility": None, "trend_strength": None, "change": "не определено"}
    close = d.close.astype(float)
    ret = close.pct_change().dropna()
    vol_now = float(ret.tail(20).std() * 100)
    vol_ref = float(ret.tail(100).std() * 100)
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    slope = float((ema20.iloc[-1] / ema20.iloc[-10] - 1) * 100) if ema20.iloc[-10] else 0
    spread = float(abs(ema20.iloc[-1] / ema50.iloc[-1] - 1) * 100)
    high_low = ((d.high - d.low) / close).tail(20).mean() * 100
    if vol_ref and vol_now > vol_ref * 1.45:
        regime = "HIGH_VOLATILITY"
        desc = "Повышенная волатильность: уменьшить размер позиции и избегать погони за ценой."
    elif spread < 0.25 and abs(slope) < 0.12:
        regime = "RANGE"
        desc = "Вероятный боковой рынок: ждать выхода из диапазона или работать от подтверждённых границ."
    elif slope > 0.12 and ema20.iloc[-1] > ema50.iloc[-1]:
        regime = "UPTREND"
        desc = "Восходящий режим: приоритет сценариям LONG после отката и подтверждения."
    elif slope < -0.12 and ema20.iloc[-1] < ema50.iloc[-1]:
        regime = "DOWNTREND"
        desc = "Нисходящий режим: приоритет сценариям SHORT после ретеста и подтверждения."
    else:
        regime = "TRANSITION"
        desc = "Переходный режим: направление недостаточно устойчиво, предпочтительно WAIT."
    prior = ret.tail(100).head(80).std() * 100
    change = "волатильность растёт" if vol_now > prior * 1.15 else "волатильность снижается" if vol_now < prior * 0.85 else "без резкого изменения"
    return {"regime": regime, "description": desc, "volatility": round(vol_now, 2), "trend_strength": round(spread, 3), "slope_pct": round(slope, 3), "change": change, "avg_range_pct": round(float(high_low), 3)}
