"""Expanding-window walk-forward evaluation with explicit out-of-sample metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _signals(df, strategy, fast, slow):
    close=df.close.astype(float)
    if strategy == "EMA crossover":
        a=close.ewm(span=fast,adjust=False).mean(); b=close.ewm(span=slow,adjust=False).mean()
        return np.where(a>b,1,-1)
    if strategy == "MACD crossover":
        m=close.ewm(span=12,adjust=False).mean()-close.ewm(span=26,adjust=False).mean(); s=m.ewm(span=9,adjust=False).mean()
        return np.where(m>s,1,-1)
    if strategy == "RSI mean reversion":
        delta=close.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
        rsi=100-100/(1+gain/loss.replace(0,np.nan))
        return np.where(rsi<30,1,np.where(rsi>70,-1,0))
    raise ValueError("Неизвестная стратегия")


def _metrics(returns, positions, timestamps):
    equity=(1+returns).cumprod(); dd=equity/equity.cummax()-1
    changes=pd.Series(positions).diff().abs().fillna(abs(positions[0]) if len(positions) else 0)
    entries=int(((changes>0)&(np.asarray(positions)!=0)).sum())
    trade_returns=[]
    p=np.asarray(positions); r=np.asarray(returns)
    start=None; side=0; acc=1.0
    for i in range(len(p)):
        if p[i] != side:
            if side != 0 and start is not None: trade_returns.append((acc-1)*100)
            side=p[i]; start=i if side!=0 else None; acc=1.0
        if side != 0: acc *= 1+r[i]
    if side != 0: trade_returns.append((acc-1)*100)
    wins=[x for x in trade_returns if x>0]; losses=[x for x in trade_returns if x<0]
    gp=sum(wins); gl=abs(sum(losses))
    return {"return_pct":float((equity.iloc[-1]-1)*100) if len(equity) else 0.0,"max_drawdown_pct":float(dd.min()*100) if len(dd) else 0.0,"win_rate_pct":float(len(wins)/max(1,len(wins)+len(losses))*100),"profit_factor":float(gp/gl) if gl else (float("inf") if gp else 0.0),"trades":len(trade_returns)}


def walk_forward_test(raw: pd.DataFrame, commission_bps=6, slippage_bps=2, folds=5):
    d=raw.copy().reset_index(drop=True)
    if len(d)<300: raise ValueError("Для walk-forward теста нужно не менее 300 свечей; желательно 1 000+.")
    strategies=["EMA crossover","MACD crossover","RSI mean reversion"]
    # Each fold selects EMA parameters on the past training segment, then evaluates untouched future data.
    min_train=max(150,int(len(d)*0.4)); remaining=len(d)-min_train
    test_size=max(20,remaining//folds)
    rows=[]; all_oos={s:[] for s in strategies}
    all_positions={s:[] for s in strategies}
    for fold in range(folds):
        train_end=min_train+fold*test_size
        test_end=min(len(d),train_end+test_size)
        if test_end-train_end<10: break
        train=d.iloc[:train_end]
        test=d.iloc[train_end:test_end].copy().reset_index(drop=True)
        # Train-only parameter selection for EMA crossover. Selection criterion is net return, not test data.
        best=(20,50,-1e9)
        for fast,slow in [(8,21),(10,30),(20,50),(20,100)]:
            if len(train)<=slow+5: continue
            sig=_signals(train,"EMA crossover",fast,slow)
            pos=pd.Series(sig).shift(1).fillna(0).to_numpy()
            gross=pos[1:]*train.close.pct_change().fillna(0).to_numpy()[1:]
            turn=np.abs(np.diff(pos,prepend=0))
            net=gross-turn[1:]*((commission_bps+slippage_bps)/10000)
            score=float(np.prod(1+net)-1)
            if score>best[2]: best=(fast,slow,score)
        for strategy in strategies:
            fast,slow=(best[0],best[1]) if strategy=="EMA crossover" else (20,50)
            # Use preceding training history to initialize indicators, but score only the OOS test segment.
            combined=pd.concat([train.tail(250),test],ignore_index=True)
            sig=_signals(combined,strategy,fast,slow)[-len(test):]
            pos=pd.Series(sig).shift(1).fillna(0).to_numpy()
            close=test.close.to_numpy(dtype=float)
            price_ret=np.zeros(len(test)); price_ret[1:]=close[1:]/close[:-1]-1
            turnover=np.abs(np.diff(pos,prepend=0))
            net=pos*price_ret-turnover*((commission_bps+slippage_bps)/10000)
            all_oos[strategy].extend(net.tolist())
            all_positions[strategy].extend(pos.tolist())
            rows.append({"fold":fold+1,"strategy":strategy,"train_end":str(train.timestamp.iloc[-1]),"test_start":str(test.timestamp.iloc[0]),"test_end":str(test.timestamp.iloc[-1]),"ema_fast":fast if strategy=="EMA crossover" else None,"ema_slow":slow if strategy=="EMA crossover" else None,"oos_return_pct":float((np.prod(1+net)-1)*100),"oos_trades":int((turnover>0).sum())})
    summary=[]
    for strategy, vals in all_oos.items():
        # Concatenated OOS fold returns, no in-sample observations included.
        rets=pd.Series(vals,dtype=float)
        if rets.empty: continue
        fold_rows=[r for r in rows if r["strategy"]==strategy]
        pos=np.asarray(all_positions[strategy], dtype=float)
        equity=(1+rets).cumprod(); dd=equity/equity.cummax()-1
        trade_returns=[]; active=False; acc=1.0; previous=0.0
        for p, r in zip(pos, rets.to_numpy()):
            if p != previous:
                if active: trade_returns.append((acc-1)*100)
                active = p != 0; acc=1.0
            if active: acc *= 1+r
            previous=p
        if active: trade_returns.append((acc-1)*100)
        wins=[v for v in trade_returns if v>0]; losses=[v for v in trade_returns if v<0]
        gp=sum(wins); gl=abs(sum(losses))
        summary.append({"strategy":strategy,"oos_return_pct":float((equity.iloc[-1]-1)*100),"oos_max_drawdown_pct":float(dd.min()*100),"win_rate_pct":float(len(wins)/max(1,len(wins)+len(losses))*100),"profit_factor":float(gp/gl) if gl else (float("inf") if gp else 0.0),"oos_trades":len(trade_returns),"folds":len(fold_rows),"mean_fold_return_pct":float(np.mean([r["oos_return_pct"] for r in fold_rows]))})
    return {"summary":pd.DataFrame(summary),"folds":pd.DataFrame(rows),"method":"Expanding-window walk-forward; параметры EMA подбираются только на прошлом участке, метрики считаются на следующих неиспользованных свечах. Комиссия и проскальзывание включены."}
