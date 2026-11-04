def make_trade_plan(df, signal, capital, risk_pct, rr=2.0):
    if signal not in ("LONG","SHORT"):
        return {"side":"WAIT"}
    last=df.iloc[-1]
    price=float(last.close)
    atr=float(last.atr) if last.atr==last.atr else price*0.01
    distance=max(atr*1.5,price*0.002)
    side=signal
    stop=price-distance if side=="LONG" else price+distance
    target=price+distance*rr if side=="LONG" else price-distance*rr
    risk_amount=float(capital)*float(risk_pct)/100
    qty=risk_amount/distance if distance>0 else 0
    return {"side":side,"entry":price,"stop_loss":stop,"take_profit":target,
            "quantity":qty,"risk_amount":risk_amount,"rr":rr}
