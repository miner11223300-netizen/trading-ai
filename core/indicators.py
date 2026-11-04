import pandas as pd
import numpy as np

def enrich(raw):
    d = raw.copy()
    for span in [9,20,50,100,200]:
        d[f"ema{span}"] = d.close.ewm(span=span, adjust=False).mean()
    delta = d.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - 100/(1+rs)
    e12 = d.close.ewm(span=12, adjust=False).mean()
    e26 = d.close.ewm(span=26, adjust=False).mean()
    d["macd"] = e12-e26
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()
    prev = d.close.shift()
    tr = pd.concat([(d.high-d.low),(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["vol_sma20"] = d.volume.rolling(20).mean()
    d["returns"] = d.close.pct_change()
    d["volatility20"] = d.returns.rolling(20).std()
    d["range_pct"] = (d.high-d.low)/d.close.replace(0, np.nan)
    return d
