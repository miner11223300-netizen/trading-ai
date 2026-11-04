import time
import numpy as np
import pandas as pd
from core.market_data import normalize
from core.indicators import enrich
from core.regime import detect_regime

# Liquid, commonly quoted commodity futures proxies on Yahoo Finance.
COMMODITY_UNIVERSE = {
    "Золото": "GC=F", "Серебро": "SI=F", "Платина": "PL=F", "Палладий": "PA=F",
    "Медь": "HG=F", "Алюминий": "ALI=F", "Нефть WTI": "CL=F", "Нефть Brent": "BZ=F",
    "Бензин RBOB": "RB=F", "Топочный мазут": "HO=F", "Природный газ": "NG=F",
    "Пшеница": "ZW=F", "Кукуруза": "ZC=F", "Соя": "ZS=F", "Соевое масло": "ZL=F",
    "Соевый шрот": "ZM=F", "Овёс": "ZO=F", "Рис": "ZR=F", "Хлопок": "CT=F",
    "Кофе": "KC=F", "Какао": "CC=F", "Сахар": "SB=F", "Апельсиновый сок": "OJ=F",
    "Живой скот": "LE=F", "Откормочный скот": "GF=F", "Свинина": "HE=F", "Древесина": "LBS=F",
    "Молоко Class III": "DC=F", "Пшеница Kansas HRW": "KE=F", "Канола": "RS=F"
}
STABLES = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDP", "EUR", "GBP"}

def _signal(df, symbol, venue, asset_class, timeframe):
    d = enrich(normalize(df))
    if len(d) < 100: return None
    x=d.iloc[-1]; prev=d.iloc[-2]
    close=float(x.close); atr=float(x.atr) if pd.notna(x.atr) else close*0.01
    if not np.isfinite(atr) or atr <= 0: atr=close*0.01
    long_score=0; short_score=0; reasons=[]
    if x.ema20 > x.ema50 > x.ema200: long_score+=2; reasons.append("EMA20>50>200")
    elif x.ema20 < x.ema50 < x.ema200: short_score+=2; reasons.append("EMA20<50<200")
    if x.macd > x.macd_signal: long_score+=1; reasons.append("MACD bullish")
    elif x.macd < x.macd_signal: short_score+=1; reasons.append("MACD bearish")
    if 50 <= x.rsi <= 68: long_score+=1; reasons.append("RSI поддерживает рост")
    elif 32 <= x.rsi <= 50: short_score+=1; reasons.append("RSI поддерживает снижение")
    if x.close > x.ema20 and x.close > prev.close: long_score+=1
    if x.close < x.ema20 and x.close < prev.close: short_score+=1
    vol_ratio=float(x.volume / max(d.volume.tail(20).mean(),1e-12)) if x.volume > 0 else 1.0
    if vol_ratio >= 1.4:
        if close > float(prev.close): long_score+=1
        elif close < float(prev.close): short_score+=1
        reasons.append(f"объём {vol_ratio:.1f}x")
    regime=detect_regime(d)
    side="WAIT"; score=max(long_score,short_score)
    if long_score>=4 and long_score-short_score>=2 and regime['regime'] not in ('DOWNTREND','HIGH_VOLATILITY'): side="LONG"
    elif short_score>=4 and short_score-long_score>=2 and regime['regime'] not in ('UPTREND','HIGH_VOLATILITY'): side="SHORT"
    entry=close; sl=tp=None
    if side=="LONG": sl=close-1.5*atr; tp=close+3*atr
    elif side=="SHORT": sl=close+1.5*atr; tp=close-3*atr
    return {"venue":venue,"asset_class":asset_class,"symbol":symbol,"timeframe":timeframe,"bar_time":str(x.timestamp),"price":close,
            "side":side,"score":score,"long_score":long_score,"short_score":short_score,"entry":entry,
            "stop_loss":sl,"take_profit":tp,"rsi":float(x.rsi),"atr_pct":atr/close*100,
            "volume_ratio":vol_ratio,"regime":regime['regime'],"reasons":"; ".join(reasons)}

def scan_crypto(exchange_id, top_n=30, timeframe="1h", limit=180, pause=0.05):
    import ccxt
    cls=getattr(ccxt,exchange_id)
    ex=cls({"enableRateLimit":True,"timeout":20000})
    markets=ex.load_markets()
    tickers=ex.fetch_tickers()
    ranked=[]
    for sym,t in tickers.items():
        m=markets.get(sym,{})
        if not m.get('active',True) or not m.get('spot',False) or not sym.endswith('/USDT'): continue
        base=sym.split('/')[0]
        if base in STABLES or any(tag in base for tag in ('UP','DOWN','BULL','BEAR')): continue
        qv=t.get('quoteVolume')
        if qv is None and t.get('baseVolume') and t.get('last'):
            qv=float(t['baseVolume'])*float(t['last'])
        if qv is not None and np.isfinite(float(qv)): ranked.append((sym,float(qv)))
    ranked=sorted(ranked,key=lambda z:z[1],reverse=True)[:top_n]
    results=[]; errors=[]
    for sym,qv in ranked:
        try:
            rows=ex.fetch_ohlcv(sym,timeframe=timeframe,limit=limit)
            if rows:
                d=pd.DataFrame(rows,columns=['timestamp','open','high','low','close','volume'])
                d['timestamp']=pd.to_datetime(d.timestamp,unit='ms',utc=True)
                r=_signal(d,sym,exchange_id.upper(),'CRYPTO',timeframe)
                if r:
                    r['quote_volume_24h']=qv; results.append(r)
        except Exception as e: errors.append(f"{sym}: {str(e)[:120]}")
        if pause: time.sleep(pause)
    return pd.DataFrame(results), errors

def scan_commodities(top_n=30, timeframe="1h", limit=180):
    import yfinance as yf
    names=list(COMMODITY_UNIVERSE.items())[:top_n]
    tickers=[t for _,t in names]
    interval="1h" if timeframe in ("5m","15m","1h") else "1d"
    period="60d" if interval=="1h" else "2y"
    raw=yf.download(tickers=tickers,period=period,interval=interval,group_by='ticker',auto_adjust=False,progress=False,threads=True)
    results=[]; errors=[]
    for name,ticker in names:
        try:
            if isinstance(raw.columns,pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0): d=raw[ticker].copy()
                elif ticker in raw.columns.get_level_values(1): d=raw.xs(ticker, axis=1, level=1).copy()
                else: raise ValueError("нет данных тикера")
            else: d=raw.copy()
            d=d.reset_index()
            d.columns=[str(c).lower().replace(' ','_') for c in d.columns]
            d=d.rename(columns={'datetime':'timestamp','date':'timestamp'})
            d=d[['timestamp','open','high','low','close','volume']].dropna(subset=['open','high','low','close']).tail(limit)
            d['volume']=d['volume'].fillna(0)
            r=_signal(d,ticker,'YAHOO',f'COMMODITY: {name}',timeframe)
            if r: r['display_name']=name; results.append(r)
        except Exception as e: errors.append(f"{name} ({ticker}): {str(e)[:120]}")
    return pd.DataFrame(results), errors

def scan_universe(crypto_venues=('binance','bybit'), crypto_n=30, commodity_n=30, timeframe='1h', limit=180):
    frames=[]; errors=[]
    for venue in crypto_venues:
        try:
            df,err=scan_crypto(venue,crypto_n,timeframe,limit)
            if not df.empty: frames.append(df)
            errors.extend(err)
        except Exception as e: errors.append(f"{venue}: {str(e)[:180]}")
    try:
        df,err=scan_commodities(commodity_n,timeframe,limit)
        if not df.empty: frames.append(df)
        errors.extend(err)
    except Exception as e: errors.append(f"Yahoo Finance commodities: {str(e)[:180]}")
    combined=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        # Rank a single cross-exchange universe of 30 distinct crypto symbols.
        crypto=combined[combined.asset_class=='CRYPTO'].copy()
        other=combined[combined.asset_class!='CRYPTO'].copy()
        if not crypto.empty:
            crypto=crypto.sort_values('quote_volume_24h',ascending=False).drop_duplicates('symbol').head(crypto_n)
        combined=pd.concat([crypto,other],ignore_index=True)
        combined['priority']=combined.apply(lambda r: (r['score']*10 + min(r.get('volume_ratio',1),3)) if r['side']!='WAIT' else 0,axis=1)
        combined=combined.sort_values(['priority','score'],ascending=False).reset_index(drop=True)
    return combined,errors
