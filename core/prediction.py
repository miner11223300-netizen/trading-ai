import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

FEATURES=["rsi","macd","macd_signal","atr","volatility20","range_pct","ema9","ema20","ema50","ema200","vol_sma20"]
def ensemble_forecast(enriched, horizon=3):
    d=enriched.copy()
    d["target"]=(d.close.shift(-horizon)>d.close).astype(int)
    d.loc[d.index[-horizon:],"target"]=np.nan
    data=d.dropna(subset=FEATURES+["target"]).copy()
    if len(data)<160:
        raise ValueError("Для ансамбля нужно минимум 160 пригодных обучающих строк; загрузите больше истории.")
    X=data[FEATURES].replace([np.inf,-np.inf],np.nan)
    y=data.target.astype(int)
    split=int(len(data)*0.8)
    if split<100 or len(data)-split<30:
        raise ValueError("Недостаточно строк для хронологической проверки.")
    X_train,X_test=X.iloc[:split],X.iloc[split:]
    y_train,y_test=y.iloc[:split],y.iloc[split:]
    models={
        "Random Forest":RandomForestClassifier(n_estimators=250,max_depth=6,min_samples_leaf=8,class_weight="balanced",random_state=42),
        "Extra Trees":ExtraTreesClassifier(n_estimators=250,max_depth=6,min_samples_leaf=8,class_weight="balanced",random_state=42),
        "Hist Gradient Boosting":HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=2,random_state=42),
        "Logistic Regression":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced"))
    }
    probs=[]; metrics=[]; notes=[]
    for name,model in models.items():
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        acc=float((pred==y_test).mean())
        try:
            latest_X=enriched[FEATURES].replace([np.inf,-np.inf],np.nan).tail(1)
            p=float(model.predict_proba(latest_X)[0][1])
        except Exception:
            p=float(np.mean(pred))
        probs.append(p)
        metrics.append({"model":name,"out_of_sample_accuracy_pct":round(acc*100,2),"latest_probability_up_pct":round(p*100,2)})
    avg=float(np.mean(probs))
    agreement=max(sum(p>=0.5 for p in probs),sum(p<0.5 for p in probs))/len(probs)
    signal="LONG" if avg>=0.55 else "SHORT" if avg<=0.45 else "WAIT"
    notes.append(f"Модели обучены на первых 80% временного ряда и оценены на последних 20%; случайное перемешивание не использовалось.")
    notes.append("Вероятности усреднены между моделями; отдельная калибровка вероятностей и walk-forward тест ещё необходимы.")
    notes.append("Это оценка направления close через заданное число свечей, а не вероятность достижения TP до SL.")
    return {"signal":signal,"prob_up":avg,"prob_down":1-avg,"agreement":agreement,
            "model_metrics":metrics,"notes":notes,"horizon":horizon}
