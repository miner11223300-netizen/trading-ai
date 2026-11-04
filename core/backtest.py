import numpy as np
import pandas as pd

def _rsi(close):
    delta=close.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    return 100-100/(1+gain/loss.replace(0,np.nan))

def run_backtest(raw, strategy="EMA + RSI", fast=20, slow=50, use_rsi=True, commission_bps=6, slippage_bps=2):
    d=raw.copy().reset_index(drop=True)
    if len(d)<max(slow,50)+30: raise ValueError("Недостаточно свечей для выбранной стратегии.")
    ef=d.close.ewm(span=fast,adjust=False).mean()
    es=d.close.ewm(span=slow,adjust=False).mean()
    rsi=_rsi(d.close)
    if strategy=="EMA + RSI":
        sig=np.where(ef>es,1,-1)
        if use_rsi:
            sig=np.where((sig==1)&(rsi>50),1,np.where((sig==-1)&(rsi<50),-1,0))
    elif strategy=="MACD crossover":
        macd=d.close.ewm(span=12,adjust=False).mean()-d.close.ewm(span=26,adjust=False).mean()
        ms=macd.ewm(span=9,adjust=False).mean()
        sig=np.where(macd>ms,1,-1)
    elif strategy=="RSI mean reversion":
        sig=np.where(rsi<30,1,np.where(rsi>70,-1,0))
    else:
        raise ValueError("Неизвестная стратегия.")
    pos=pd.Series(sig,index=d.index).shift(1).fillna(0)
    gross=pos*d.close.pct_change().fillna(0)
    turnover=pos.diff().abs().fillna(pos.abs())
    cost=turnover*((commission_bps+slippage_bps)/10000)
    net=gross-cost
    equity=(1+net).cumprod()
    peak=equity.cummax()
    dd=equity/peak-1
    changes=pos.ne(pos.shift()).fillna(True)
    starts=list(d.index[changes & pos.ne(0)])
    trade_rows=[]
    for j,start in enumerate(starts):
        end=(starts[j+1]-1) if j+1<len(starts) else len(d)-1
        segment=net.iloc[start:end+1]
        trade_ret=float((1+segment).prod()-1)
        trade_rows.append({"entry_time":str(d.timestamp.iloc[start]),"exit_time":str(d.timestamp.iloc[end]),
                           "side":"LONG" if pos.iloc[start]>0 else "SHORT","return_pct":trade_ret*100})
    tr=pd.DataFrame(trade_rows)
    wins=int((tr.return_pct>0).sum()) if not tr.empty else 0
    losses=int((tr.return_pct<0).sum()) if not tr.empty else 0
    gp=float(tr.loc[tr.return_pct>0,"return_pct"].sum()) if not tr.empty else 0
    gl=abs(float(tr.loc[tr.return_pct<0,"return_pct"].sum())) if not tr.empty else 0
    metrics={"strategy":strategy,"total_return_pct":float((equity.iloc[-1]-1)*100),
             "max_drawdown_pct":float(dd.min()*100),
             "win_rate_pct":float(wins/max(1,wins+losses)*100),
             "profit_factor":float(gp/gl) if gl>0 else (float("inf") if gp>0 else 0),
             "trades":len(tr)}
    return {"metrics":metrics,"equity":pd.DataFrame({"timestamp":d.timestamp,"equity":equity}),
            "trades":tr}
