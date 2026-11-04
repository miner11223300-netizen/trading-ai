import numpy as np
import pandas as pd

def _pivots(df, left=3, right=3):
    highs, lows = [], []
    for i in range(left, len(df)-right):
        h = df.high.iloc[i]
        l = df.low.iloc[i]
        if h >= df.high.iloc[i-left:i+right+1].max():
            highs.append((i,float(h)))
        if l <= df.low.iloc[i-left:i+right+1].min():
            lows.append((i,float(l)))
    return highs, lows

def analyze_smc(df):
    d = df.reset_index(drop=True)
    ph, pl = _pivots(d)
    events = []
    structure = "RANGE / не определено"
    last_high = last_low = None
    bias = None
    for i in range(1,len(d)):
        prior_highs = [x for x in ph if x[0] < i]
        prior_lows = [x for x in pl if x[0] < i]
        if prior_highs: last_high = prior_highs[-1][1]
        if prior_lows: last_low = prior_lows[-1][1]
        close = float(d.close.iloc[i])
        prev_close = float(d.close.iloc[i-1])
        if last_high and prev_close <= last_high and close > last_high:
            event = "BOS bullish" if bias != "bearish" else "CHoCH bullish"
            bias = "bullish"; structure = event
            events.append({"time":str(d.timestamp.iloc[i]),"event":event,"level":last_high})
        elif last_low and prev_close >= last_low and close < last_low:
            event = "BOS bearish" if bias != "bullish" else "CHoCH bearish"
            bias = "bearish"; structure = event
            events.append({"time":str(d.timestamp.iloc[i]),"event":event,"level":last_low})
    fvg=[]
    for i in range(2,len(d)):
        # Three-candle imbalance
        if d.low.iloc[i] > d.high.iloc[i-2]:
            fvg.append({"time":str(d.timestamp.iloc[i]),"type":"bullish","lower":float(d.high.iloc[i-2]),"upper":float(d.low.iloc[i])})
        if d.high.iloc[i] < d.low.iloc[i-2]:
            fvg.append({"time":str(d.timestamp.iloc[i]),"type":"bearish","lower":float(d.high.iloc[i]),"upper":float(d.low.iloc[i-2])})
    order_blocks=[]
    for ev in events[-20:]:
        t = pd.Timestamp(ev["time"])
        idxs = d.index[d.timestamp == t].tolist()
        if not idxs: continue
        i=idxs[0]
        start=max(0,i-8)
        prior=d.iloc[start:i]
        if prior.empty: continue
        if "bullish" in ev["event"]:
            candidates=prior[prior.close < prior.open]
        else:
            candidates=prior[prior.close > prior.open]
        if not candidates.empty:
            c=candidates.iloc[-1]
            order_blocks.append({"time":str(c.timestamp),"type":"bullish" if "bullish" in ev["event"] else "bearish",
                                 "low":float(c.low),"high":float(c.high),"basis":"последняя противоположная свеча перед пробоем"})
    liquidity=[]
    tol = float(d.close.iloc[-1])*0.0015
    for kind, pivots in [("equal highs",ph),("equal lows",pl)]:
        clusters=[]
        for idx,price in pivots[-80:]:
            match=next((cl for cl in clusters if abs(cl["price"]-price)<=tol),None)
            if match:
                match["count"]+=1; match["indices"].append(idx)
            else:
                clusters.append({"price":price,"count":1,"indices":[idx]})
        for cl in clusters:
            if cl["count"]>=2:
                liquidity.append({"type":kind,"level":cl["price"],"touches":cl["count"]})
    return {"structure":structure,"events":events[-50:],"fvg":fvg[-50:],
            "order_blocks":order_blocks[-30:],"liquidity":liquidity[-30:],
            "swing_highs":[{"index":i,"price":p} for i,p in ph[-30:]],
            "swing_lows":[{"index":i,"price":p} for i,p in pl[-30:]]}
