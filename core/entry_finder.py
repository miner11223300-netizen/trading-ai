"""Rule-based entry timing: liquidity sweeps, breakout/retest and ATR risk levels."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .smc import analyze_smc


def find_entry_setup(df: pd.DataFrame, rr: float = 2.0, lookback: int = 50) -> dict:
    if df is None or len(df) < max(80, lookback + 10):
        raise ValueError("Для поиска точки входа требуется не менее 80 свечей.")
    d = df.copy().reset_index(drop=True)
    x = d.iloc[-1]
    close, high, low = map(float, (x.close, x.high, x.low))
    atr = float(x.get("atr", np.nan))
    if not np.isfinite(atr) or atr <= 0:
        tr = pd.concat([(d.high-d.low), (d.high-d.close.shift()).abs(), (d.low-d.close.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.tail(14).mean())
    prev = d.iloc[-2]
    window = d.iloc[-lookback-1:-1]
    prior_high, prior_low = float(window.high.max()), float(window.low.min())
    body = abs(close-float(x.open))
    upper_wick = high-max(close,float(x.open))
    lower_wick = min(close,float(x.open))-low
    vol_avg = float(d.volume.tail(21).head(20).mean()) if len(d) >= 21 else 0
    vol_ratio = float(x.volume/vol_avg) if vol_avg > 0 else 1.0
    # Liquidity sweep: wick takes prior extreme, close returns inside the range.
    bull_sweep = low < prior_low and close > prior_low and lower_wick > max(body, atr*0.15)
    bear_sweep = high > prior_high and close < prior_high and upper_wick > max(body, atr*0.15)
    # Breakout and retest: previous close outside range, latest candle retests the broken level.
    bull_breakout = float(prev.close) > prior_high and low <= prior_high + 0.2*atr and close >= prior_high
    bear_breakout = float(prev.close) < prior_low and high >= prior_low - 0.2*atr and close <= prior_low
    smc = analyze_smc(d)
    structure = str(smc.get("structure", ""))
    long_points = int(bull_sweep)*2 + int(bull_breakout)*2 + int("bullish" in structure.lower()) + int(close > float(x.get("ema20", close))) + int(vol_ratio >= 1.2)
    short_points = int(bear_sweep)*2 + int(bear_breakout)*2 + int("bearish" in structure.lower()) + int(close < float(x.get("ema20", close))) + int(vol_ratio >= 1.2)
    if long_points >= 3 and long_points >= short_points + 2:
        side = "LONG"
        reason = "; ".join([s for s, ok in [("снятие ликвидности снизу",bull_sweep),("пробой и ретест сопротивления",bull_breakout),("бычья структура", "bullish" in structure.lower()),("цена выше EMA20",close > float(x.get("ema20",close))),("объём выше среднего",vol_ratio>=1.2)] if ok])
        entry = max(close, prior_high if bull_breakout else close)
        stop = min(low, prior_low) - 0.2*atr if bull_sweep else min(low, entry-1.2*atr)-0.1*atr
        target = entry + (entry-stop)*rr
    elif short_points >= 3 and short_points >= long_points + 2:
        side = "SHORT"
        reason = "; ".join([s for s, ok in [("снятие ликвидности сверху",bear_sweep),("пробой и ретест поддержки",bear_breakout),("медвежья структура", "bearish" in structure.lower()),("цена ниже EMA20",close < float(x.get("ema20",close))),("объём выше среднего",vol_ratio>=1.2)] if ok])
        entry = min(close, prior_low if bear_breakout else close)
        stop = max(high, prior_high) + 0.2*atr if bear_sweep else max(high, entry+1.2*atr)+0.1*atr
        target = entry - (stop-entry)*rr
    else:
        side, entry, stop, target = "WAIT", np.nan, np.nan, np.nan
        reason = "Нет достаточного подтверждения снятия ликвидности, ретеста или импульса. Не входить в середине диапазона."
    return {"side":side,"entry":float(entry) if np.isfinite(entry) else np.nan,"stop_loss":float(stop) if np.isfinite(stop) else np.nan,"take_profit":float(target) if np.isfinite(target) else np.nan,"rr":float(rr),"long_score":long_points,"short_score":short_points,"bullish_sweep":bool(bull_sweep),"bearish_sweep":bool(bear_sweep),"bullish_breakout_retest":bool(bull_breakout),"bearish_breakout_retest":bool(bear_breakout),"prior_high":prior_high,"prior_low":prior_low,"volume_ratio":vol_ratio,"structure":structure,"reason":reason,"confirmation":"Дождаться закрытия следующей свечи в сторону сценария; если цена не удерживает уровень ретеста, сценарий отменяется."}
