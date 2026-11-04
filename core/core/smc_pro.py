import pandas as pd
import numpy as np


def analyze_smc_pro(df: pd.DataFrame, lookback=120):
    d=df.tail(lookback).reset_index(drop=True)
    if len(d)<20: return {"events":[],"sweeps":[],"fvg":[],"order_blocks":[],"liquidity":[],"bias":"neutral"}
    highs=d.high.astype(float); lows=d.low.astype(float); closes=d.close.astype(float)
    events=[]; sweeps=[]; fvgs=[]; obs=[]; liquidity=[]
    # Confirmed local swing levels; right-side confirmation avoids using unfinished future bars.
    piv_hi=[]; piv_lo=[]
    for i in range(3,len(d)-3):
        if highs.iloc[i] == highs.iloc[i-3:i+4].max(): piv_hi.append((i,float(highs.iloc[i])))
        if lows.iloc[i] == lows.iloc[i-3:i+4].min(): piv_lo.append((i,float(lows.iloc[i])))
    bias='neutral'
    for i in range(1,len(d)):
        prior_hi=[p for p in piv_hi if p[0] < i]
        prior_lo=[p for p in piv_lo if p[0] < i]
        if prior_hi:
            level=prior_hi[-1][1]
            if closes.iloc[i-1] <= level < closes.iloc[i]:
                typ='BOS bullish' if bias!='bearish' else 'CHoCH bullish'; bias='bullish'
                events.append({"time":str(d.timestamp.iloc[i]),"event":typ,"level":level})
                candidates=d.iloc[max(0,i-8):i]
                c=candidates[candidates.close<candidates.open]
                if not c.empty:
                    z=c.iloc[-1]; obs.append({"time":str(z.timestamp),"type":"bullish","low":float(z.low),"high":float(z.high),"basis":"last bearish candle before bullish break"})
            elif highs.iloc[i] > level and closes.iloc[i] < level and closes.iloc[i-1] <= level:
                sweeps.append({"time":str(d.timestamp.iloc[i]),"type":"buy-side liquidity sweep","level":level,"close":float(closes.iloc[i]),"wick":float(highs.iloc[i]-level)})
        if prior_lo:
            level=prior_lo[-1][1]
            if closes.iloc[i-1] >= level > closes.iloc[i]:
                typ='BOS bearish' if bias!='bullish' else 'CHoCH bearish'; bias='bearish'
                events.append({"time":str(d.timestamp.iloc[i]),"event":typ,"level":level})
                candidates=d.iloc[max(0,i-8):i]
                c=candidates[candidates.close>candidates.open]
                if not c.empty:
                    z=c.iloc[-1]; obs.append({"time":str(z.timestamp),"type":"bearish","low":float(z.low),"high":float(z.high),"basis":"last bullish candle before bearish break"})
            elif lows.iloc[i] < level and closes.iloc[i] > level and closes.iloc[i-1] >= level:
                sweeps.append({"time":str(d.timestamp.iloc[i]),"type":"sell-side liquidity sweep","level":level,"close":float(closes.iloc[i]),"wick":float(level-lows.iloc[i])})
    for i in range(2,len(d)):
        if lows.iloc[i] > highs.iloc[i-2]: fvgs.append({"time":str(d.timestamp.iloc[i]),"type":"bullish FVG","low":float(highs.iloc[i-2]),"high":float(lows.iloc[i]),"status":"open" if lows.iloc[-1] > highs.iloc[i-2] else "possibly mitigated"})
        if highs.iloc[i] < lows.iloc[i-2]: fvgs.append({"time":str(d.timestamp.iloc[i]),"type":"bearish FVG","low":float(highs.iloc[i]),"high":float(lows.iloc[i-2]),"status":"open" if highs.iloc[-1] < lows.iloc[i-2] else "possibly mitigated"})
    tol=float(closes.iloc[-1])*0.001
    for typ,points in [('equal highs',piv_hi),('equal lows',piv_lo)]:
        clusters=[]
        for idx,price in points[-60:]:
            hit=next((x for x in clusters if abs(x['level']-price)<=tol),None)
            if hit: hit['touches']+=1
            else: clusters.append({'level':price,'touches':1})
        liquidity.extend([{'type':typ,**x} for x in clusters if x['touches']>=2])
    return {"events":events[-20:],"sweeps":sweeps[-20:],"fvg":fvgs[-20:],"order_blocks":obs[-15:],"liquidity":liquidity[-20:],"bias":bias}
