import streamlit as st
import os
try:
    if st.secrets.get("DATABASE_URL"):
        os.environ["DATABASE_URL"] = str(st.secrets["DATABASE_URL"])
except Exception:
    pass
import pandas as pd
from core.market_data import load_csv, fetch_crypto, fetch_yahoo, fetch_moex
from core.indicators import enrich
from core.vision import inspect_chart
from core.smc import analyze_smc
from core.prediction import ensemble_forecast
from core.backtest import run_backtest
from core.risk import make_trade_plan
from core.paper_broker import init_db, create_paper_trade, close_paper_trade, list_trades, monitor_paper_trades
from core.advisor import analyze_market
from core.multitimeframe import analyze_mtf_frames
from core.entry_finder import find_entry_setup
from core.signal_quality import walk_forward_test
from core.scanner import scan_universe
from core.experiment import (init_experiment, active_experiment, start_experiment,
    open_top_trades, update_prices, experiment_report, finish_if_due, log_event)
from core.smc_pro import analyze_smc_pro
from core.regime import detect_regime
from core.journal_analytics import record_signal, list_signals, performance_summary, close_signal
import hashlib
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ccxt
import yfinance as yf

def send_telegram(text):
    try:
        token=st.secrets["TELEGRAM_BOT_TOKEN"]
        chat=st.secrets["TELEGRAM_CHAT_ID"]
        if not token or not chat: return False, "Telegram Secrets не заполнены"
        response=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                               json={"chat_id":chat,"text":text},timeout=15)
        response.raise_for_status()
        return True, "sent"
    except Exception as exc:
        return False, str(exc)[:180]


def fetch_chart_ohlcv(row, limit=90):
    """Получить свечи для графика конкретного сигнала."""
    venue=str(row.get("venue","")).upper()
    symbol=str(row.get("symbol",""))
    tf=str(row.get("timeframe","1h"))
    if venue == "YAHOO" or "GOLD" in str(row.get("asset_class","")).upper():
        ticker=symbol if symbol in ("XAUUSD=X","GC=F") else "XAUUSD=X"
        interval = "1m" if tf in ("1m","5m","15m") else ("1h" if tf in ("1h","4h") else "1d")
        hist=yf.Ticker(ticker).history(period="7d" if interval in ("1m","1h") else "1y", interval=interval)
        if hist is None or hist.empty: return None
        return hist.tail(limit)
    exchange_id=venue.lower()
    if not hasattr(ccxt, exchange_id): return None
    ex=getattr(ccxt,exchange_id)({"enableRateLimit":True,"timeout":15000})
    rows=ex.fetch_ohlcv(symbol,timeframe=tf,limit=limit)
    if not rows: return None
    return pd.DataFrame(rows,columns=["timestamp","open","high","low","close","volume"]).assign(
        timestamp=lambda d: pd.to_datetime(d.timestamp,unit="ms",utc=True)
    ).set_index("timestamp")

def draw_signal_chart(row, history):
    """Отрисовать свечной график с входом, стопом и целью."""
    fig, ax=plt.subplots(figsize=(10,4.5),dpi=140)
    if history is None or history.empty:
        ax.text(.5,.5,"Нет исторических свечей для графика",ha="center",va="center",transform=ax.transAxes)
        ax.axis("off")
        return fig
    h=history.copy()
    cols={str(c).lower():c for c in h.columns}
    o,hg,l,cl=[cols[k] for k in ("open","high","low","close")]
    xs=list(range(len(h)))
    for i,(_,r) in enumerate(h.iterrows()):
        up=float(r[cl])>=float(r[o])
        candle_color="#159b72" if up else "#d9534f"
        ax.vlines(i,float(r[l]),float(r[hg]),color=candle_color,linewidth=1)
        bottom=min(float(r[o]),float(r[cl]))
        height=max(abs(float(r[cl])-float(r[o])),1e-9)
        ax.add_patch(Rectangle((i-.28,bottom),.56,height,facecolor=candle_color,edgecolor=candle_color))
    entry=float(row.get("entry",row.get("price",0)) or 0)
    sl=float(row.get("stop_loss") or 0)
    tp=float(row.get("take_profit") or 0)
    ax.axhline(entry,color="#2878b5",linestyle="-",linewidth=1.3,label=f"Вход: {entry:g}")
    if sl: ax.axhline(sl,color="#d9534f",linestyle="--",linewidth=1.2,label=f"Стоп: {sl:g}")
    if tp: ax.axhline(tp,color="#159b72",linestyle="--",linewidth=1.2,label=f"Цель: {tp:g}")
    side="ПОКУПКА" if row.get("side")=="LONG" else "ПРОДАЖА"
    ax.set_title(f"{row.get('symbol','')} · {side} · оценка {row.get('opportunity_score',0)}")
    ax.set_ylabel("Цена")
    ax.grid(alpha=.2)
    ax.legend(loc="best",fontsize=8)
    tick_idx=list(range(0,len(h),max(1,len(h)//6)))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(h.index[i])[:16] for i in tick_idx],rotation=25,ha="right",fontsize=7)
    fig.tight_layout()
    return fig

st.set_page_config(page_title="Trading AI", page_icon="📊", layout="wide")
init_db()
init_experiment()
st.title("📊 Trading AI — Research & Paper Trading")
st.caption("Multi-market analytics · Smart Money · ML ensemble · backtesting · approval workflow")
st.warning("Исследовательская версия. Все сделки исполняются только в Paper Trading. Реальные ордера и хранение API-ключей здесь не включены.")

with st.sidebar:
    st.header("Рынок и данные")
    asset_class = st.selectbox("Класс рынка", ["Crypto", "Precious metals", "Industrial metals"])
    provider = st.selectbox("Источник", ["Binance", "Bybit", "Yahoo Finance", "MOEX ISS", "CSV"])
    if provider in ["Binance", "Bybit"]:
        symbol = st.text_input("Символ CCXT", "BTC/USDT")
        timeframe = st.selectbox("Таймфрейм", ["5m","15m","1h","4h","1d"], index=2)
    elif provider == "Yahoo Finance":
        symbols = {
            "Precious metals": ["GC=F","SI=F","PL=F","PA=F"],
            "Industrial metals": ["HG=F","ALI=F"]
        }
        options = symbols.get(asset_class, ["GC=F","SI=F","PL=F","PA=F","HG=F","ALI=F"])
        symbol = st.selectbox("Тикер", options)
        timeframe = st.selectbox("Таймфрейм", ["1d","1wk"], index=0)
    elif provider == "MOEX ISS":
        symbol = st.text_input("SECID фьючерса/инструмента", "Si-12.26")
        timeframe = st.selectbox("Интервал MOEX", ["1h","1d"], index=0)
    else:
        symbol = st.text_input("Название инструмента", "CUSTOM")
        timeframe = st.selectbox("Таймфрейм CSV", ["5m","15m","1h","4h","1d"], index=2)
    limit = st.slider("Свечей", 150, 1500, 700, 50)
    st.divider()
    st.header("Управление риском")
    capital = st.number_input("Капитал (в валюте котировки)", min_value=100.0, value=10000.0, step=500.0)
    risk_pct = st.slider("Риск на сделку, %", 0.1, 2.0, 0.5, 0.1)
    rr = st.slider("Risk / Reward", 1.0, 5.0, 2.0, 0.5)
    commission_bps = st.number_input("Комиссия, bps за сторону", 0.0, 100.0, 6.0)
    slippage_bps = st.number_input("Проскальзывание, bps за сторону", 0.0, 100.0, 2.0)

st.caption("Trading AI v5.1 · Сканер и журнал вынесены в первые вкладки для удобства на телефоне.")
tabs = st.tabs(["🌐 TOP-30 + XAU/USD", "📒 ЖУРНАЛ СИГНАЛОВ", "Обзор рынка", "AI Vision", "Smart Money", "Prediction Engine", "AI Аналитик", "Backtesting", "Multi-Timeframe & Entry", "Качество сигналов", "Paper Trading", "🧪 Испытание $2,000"])

def get_data():
    if provider == "CSV":
        f = st.session_state.get("csv_data")
        if f is None:
            st.error("Сначала загрузите CSV на вкладке «Обзор рынка».")
            return None
        return load_csv(f)
    if provider in ["Binance", "Bybit"]:
        return fetch_crypto(provider.lower(), symbol, timeframe, limit)
    if provider == "Yahoo Finance":
        return fetch_yahoo(symbol, timeframe, limit)
    return fetch_moex(symbol, timeframe, limit)

with tabs[2]:
    st.subheader("Загрузка рыночных данных")
    if provider == "CSV":
        csv_file = st.file_uploader("CSV: timestamp, open, high, low, close, volume", type=["csv"], key="csv_main")
        if csv_file:
            st.session_state["csv_data"] = csv_file
    if st.button("Получить / обновить котировки", type="primary"):
        try:
            df0 = get_data()
            if df0 is not None:
                st.session_state["market_df"] = df0
                st.success(f"Загружено {len(df0)} свечей.")
        except Exception as e:
            st.error(f"Ошибка получения данных: {e}")
    if "market_df" in st.session_state:
        df = enrich(st.session_state["market_df"])
        st.session_state["enriched_df"] = df
        last = df.iloc[-1]
        a,b,c,d = st.columns(4)
        a.metric("Последняя цена", f"{last.close:,.4f}")
        b.metric("RSI 14", f"{last.rsi:.1f}" if pd.notna(last.rsi) else "—")
        c.metric("ATR 14", f"{last.atr:,.4f}" if pd.notna(last.atr) else "—")
        d.metric("Объём", f"{last.volume:,.2f}")
        regime = detect_regime(df)
        st.info(f"**Рыночный режим: {regime['regime']}** — {regime['description']} · Волатильность: {regime['volatility']}% · {regime['change']}")
        st.line_chart(df.set_index("timestamp")[["close","ema20","ema50","ema200"]].tail(300))
        st.dataframe(df.tail(30), use_container_width=True)
        st.download_button("Скачать подготовленные данные CSV", df.to_csv(index=False), "trading_ai_data.csv", "text/csv")
    else:
        st.info("Выберите источник и загрузите данные. Для металлов доступны Yahoo Finance и MOEX ISS; доступность конкретных контрактов зависит от поставщика.")

with tabs[3]:
    st.subheader("AI Vision — анализ скриншота")
    img = st.file_uploader("Загрузите скриншот TradingView / терминала", type=["png","jpg","jpeg","webp"], key="vision")
    if img:
        st.image(img, use_container_width=True)
        if st.button("Исследовать изображение"):
            try:
                report = inspect_chart(img.getvalue())
                st.success("Изображение обработано.")
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("Размер изображения", f'{report["image_width"]} × {report["image_height"]}')
                v2.metric("Зелёные области", f'{report["green_pixel_share_pct"]:.2f}%')
                v3.metric("Красные области", f'{report["red_pixel_share_pct"]:.2f}%')
                v4.metric("Кандидаты на свечи", report["candidate_candle_components"])
                st.write(f'**Визуальный баланс:** {report["visual_balance"]}')
                lines = report.get("candidate_horizontal_lines", [])
                if lines:
                    st.write("**Обнаруженные горизонтальные линии (координаты в пикселях):**")
                    st.dataframe(pd.DataFrame(lines), use_container_width=True, hide_index=True)
                with st.expander("Ограничения визуального анализа"):
                    for note in report.get("limitations", []):
                        st.write("• " + note)
                st.info("Скриншот сам по себе не даёт надёжной цены входа, Stop Loss и Take Profit. Для торговой рекомендации загрузите OHLCV-котировки на вкладке «Обзор рынка», затем откройте «Prediction Engine» и рассчитайте прогноз.")
            except Exception as e:
                st.error(f"Ошибка обработки изображения: {e}")
    else:
        st.info("Загрузите скриншот. Для точных уровней и сигналов дополнительно используйте OHLCV из выбранного источника.")

with tabs[4]:
    st.subheader("Smart Money Concepts")
    if "enriched_df" in st.session_state:
        smc = analyze_smc(st.session_state["enriched_df"])
        st.metric("Последнее структурное событие", smc["structure"])
        x,y,z = st.columns(3)
        x.metric("FVG найдено", len(smc["fvg"]))
        y.metric("Order Blocks", len(smc["order_blocks"]))
        z.metric("Пулы ликвидности", len(smc["liquidity"]))
        st.write("**События структуры**")
        st.dataframe(pd.DataFrame(smc["events"]), use_container_width=True)
        st.write("**Fair Value Gaps**")
        st.dataframe(pd.DataFrame(smc["fvg"]), use_container_width=True)
        st.write("**Order Blocks (эвристические зоны)**")
        st.dataframe(pd.DataFrame(smc["order_blocks"]), use_container_width=True)
        st.write("**Пулы ликвидности / равные экстремумы**")
        st.dataframe(pd.DataFrame(smc["liquidity"]), use_container_width=True)
        with st.expander("Smart Money Pro — sweep, FVG и расширенные зоны"):
            pro = analyze_smc_pro(st.session_state["enriched_df"])
            st.write(f"**Структурный уклон:** {pro['bias']}")
            for label, key in [("Снятие ликвидности", "sweeps"), ("BOS / CHoCH", "events"), ("FVG и статус", "fvg"), ("Order Blocks", "order_blocks"), ("Пулы равных экстремумов", "liquidity")]:
                st.markdown(f"**{label}**")
                st.dataframe(pd.DataFrame(pro[key]), use_container_width=True, hide_index=True)
    else:
        st.info("Сначала загрузите рыночные данные.")

with tabs[5]:
    st.subheader("Prediction Engine — ансамбль моделей")
    if "enriched_df" in st.session_state:
        horizon = st.selectbox("Горизонт прогноза (свечей)", [1,3,5,10], index=1)
        if st.button("Рассчитать прогноз", type="primary"):
            try:
                forecast = ensemble_forecast(st.session_state["enriched_df"], horizon=horizon)
                st.session_state["forecast"] = forecast
                p,q,r = st.columns(3)
                p.metric("Направление", forecast["signal"])
                q.metric("Вероятность роста", f'{forecast["prob_up"]*100:.1f}%')
                r.metric("Вероятность снижения", f'{forecast["prob_down"]*100:.1f}%')
                st.caption(f'Горизонт: {horizon} свечей. Доля совпадения моделей: {forecast["agreement"]*100:.1f}%. Вероятности — ансамблевые оценки направления, а не вероятность прибыльной сделки.')
                plan = make_trade_plan(st.session_state["enriched_df"], forecast["signal"], capital, risk_pct, rr)
                st.divider()
                st.subheader("Торговая рекомендация")
                if plan["side"] == "WAIT":
                    st.warning("РЕШЕНИЕ: ЖДАТЬ. Сейчас ансамбль не показывает достаточно выраженного преимущества ни для LONG, ни для SHORT. Не открывайте позицию только ради входа.")
                    st.write(f'Условие для пересмотра: дождаться нового закрытия свечи и повторно рассчитать прогноз. Текущая оценка роста: {forecast["prob_up"]*100:.1f}%, снижения: {forecast["prob_down"]*100:.1f}%.')
                else:
                    st.success(f'РЕШЕНИЕ: рассмотреть {plan["side"]}. Это учебный сигнал; перед входом дождитесь закрытия текущей свечи и повторной проверки условий.')
                    e1,e2,e3,e4 = st.columns(4)
                    e1.metric("Направление", plan["side"])
                    e2.metric("Ориентир входа", f'{plan["entry"]:,.4f}')
                    e3.metric("STOP LOSS", f'{plan["stop_loss"]:,.4f}')
                    e4.metric("TAKE PROFIT", f'{plan["take_profit"]:,.4f}')
                    e5,e6,e7 = st.columns(3)
                    e5.metric("Размер позиции", f'{plan["quantity"]:.6f}')
                    e6.metric("Риск в деньгах", f'{plan["risk_amount"]:,.2f}')
                    e7.metric("Risk / Reward", f'1 : {plan["rr"]:.1f}')
                    st.write(f'Оценка моделей: рост {forecast["prob_up"]*100:.1f}% · снижение {forecast["prob_down"]*100:.1f}%.')
                    st.warning("Не является гарантией результата. Размер позиции рассчитан по заданному риску и расстоянию до стопа; проверьте ликвидность, спред, комиссию и актуальность котировки. Реальные ордера не отправляются.")
                with st.expander("Качество моделей и подробности"):
                    st.dataframe(pd.DataFrame(forecast["model_metrics"]), use_container_width=True)
                    for note in forecast["notes"]:
                        st.write("• " + note)
            except Exception as e:
                st.error(f"Не удалось обучить модели: {e}")
    else:
        st.info("Сначала загрузите рыночные данные.")

with tabs[6]:
    st.subheader("AI Аналитик — комплексное торговое заключение")
    st.caption("Совместная оценка тренда, импульса, объёма, волатильности, уровней и Smart Money. Анализ использует загруженные OHLCV-данные.")
    if "enriched_df" not in st.session_state:
        st.info("Сначала откройте «Обзор рынка» и загрузите котировки.")
    else:
        if st.button("Сформировать торговое заключение", type="primary", key="run_full_advisor"):
            try:
                st.session_state["advisor_report"] = analyze_market(
                    st.session_state["enriched_df"], capital=capital,
                    risk_pct=risk_pct, rr=rr
                )
            except Exception as exc:
                st.error(f"Не удалось сформировать заключение: {exc}")
        report = st.session_state.get("advisor_report")
        if report:
            side = report["side"]
            if side == "LONG":
                st.success("РЕШЕНИЕ: РАССМОТРЕТЬ LONG — только после подтверждения следующей свечой.")
            elif side == "SHORT":
                st.error("РЕШЕНИЕ: РАССМОТРЕТЬ SHORT — только после подтверждения следующей свечой.")
            else:
                st.warning("РЕШЕНИЕ: WAIT — сейчас лучше ждать, чем открывать сделку.")
                st.write(report["wait_reason"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Цена последней свечи", f'{report["price"]:,.6g}')
            m2.metric("Перевес факторов", f'{report["net_score"]:+d}', f'LONG {report["long_score"]} / SHORT {report["short_score"]}')
            m3.metric("Индекс согласованности", f'{report["strength_index"]}/95')
            m4.metric("ADX", f'{report["adx"]:.1f}')
            if side != "WAIT":
                p1,p2,p3,p4 = st.columns(4)
                p1.metric("Направление", side)
                p2.metric("Вход (ориентир)", f'{report["entry"]:,.6g}')
                p3.metric("STOP LOSS", f'{report["stop_loss"]:,.6g}')
                p4.metric("TAKE PROFIT", f'{report["take_profit"]:,.6g}')
                p5,p6,p7 = st.columns(3)
                p5.metric("Размер позиции", f'{report["quantity"]:,.6g}')
                p6.metric("Риск в деньгах", f'{report["risk_cash"]:,.2f}')
                p7.metric("Risk / Reward", f'1 : {report["rr"]:.1f}')
                st.caption("Размер позиции рассчитан по заданному денежному риску и расстоянию до стопа. Для контрактов с множителем, плечом, шагом лота или иной валютой расчёта его нужно скорректировать.")
            st.markdown("#### Что говорят индикаторы")
            st.dataframe(pd.DataFrame(report["factors"]), use_container_width=True, hide_index=True)
            st.markdown("#### Ключевые уровни")
            l1,l2,l3,l4 = st.columns(4)
            l1.metric("Поддержка", f'{report["support"]:,.6g}')
            l2.metric("Сопротивление", f'{report["resistance"]:,.6g}')
            l3.metric("RSI 14", f'{report["rsi"]:.1f}')
            l4.metric("Объём / средний", f'{report["volume_ratio"]:.2f}x')
            st.write(f'**Структура рынка:** {report["structure"]}. FVG: {report["fvg_count"]}; Order Blocks: {report["order_block_count"]}.')
            st.markdown("#### План действий")
            for item in report["triggers"]:
                st.write("• " + item)
            for item in report["warnings"]:
                st.warning(item)
            st.info(report["explanation"])
            st.caption("Исследовательский инструмент, не финансовая рекомендация. Индекс согласованности не является вероятностью прибыльной сделки. Входные уровни — ориентиры по последней свече; учитывайте спред, комиссию, проскальзывание, funding и спецификацию инструмента.")
            if side != "WAIT":
                st.warning("Paper Trading: подтверждение ниже создаст только виртуальную позицию.")
                if st.button("Подтвердить эту рекомендацию в Paper Trading", type="primary", key="advisor_paper_confirm"):
                    paper_plan = {"side": side, "entry": report["entry"], "stop_loss": report["stop_loss"],
                                  "take_profit": report["take_profit"], "quantity": report["quantity"],
                                  "risk_amount": report["risk_cash"], "rr": report["rr"]}
                    create_paper_trade(symbol, paper_plan)
                    st.success("Виртуальная сделка зарегистрирована в журнале Paper Trading.")

with tabs[7]:
    st.subheader("Backtesting — EMA crossover + RSI filter")
    if "enriched_df" in st.session_state:
        strategy = st.selectbox("Стратегия", ["EMA + RSI", "MACD crossover", "RSI mean reversion"])
        fast = st.slider("Быстрая EMA", 5, 50, 20)
        slow = st.slider("Медленная EMA", 30, 200, 50)
        rsi_filter = st.checkbox("Использовать RSI-фильтр", value=True)
        if st.button("Запустить исторический тест"):
            try:
                bt = run_backtest(st.session_state["market_df"], strategy=strategy, fast=fast, slow=slow,
                                  use_rsi=rsi_filter, commission_bps=commission_bps,
                                  slippage_bps=slippage_bps)
                m = bt["metrics"]
                a,b,c,d,e = st.columns(5)
                a.metric("Доходность", f'{m["total_return_pct"]:.2f}%')
                b.metric("Max Drawdown", f'{m["max_drawdown_pct"]:.2f}%')
                c.metric("Win Rate", f'{m["win_rate_pct"]:.1f}%')
                d.metric("Profit Factor", f'{m["profit_factor"]:.2f}')
                e.metric("Сделок", m["trades"])
                st.line_chart(bt["equity"].set_index("timestamp")[["equity"]])
                st.dataframe(bt["trades"], use_container_width=True)
                st.caption("Тест использует сигнал предыдущей закрытой свечи для следующей свечи; комиссия и проскальзывание вычитаются. Это базовый бэктест без моделирования стакана, funding, ликвидаций и точного исполнения стопов внутри свечи.")
            except Exception as e:
                st.error(f"Ошибка бэктеста: {e}")
    else:
        st.info("Сначала загрузите рыночные данные.")

with tabs[8]:
    st.subheader("Multi-Timeframe Analysis — 5m / 15m / 1h / 4h / 1d")
    st.caption("Старшие таймфреймы имеют больший вес. Отсутствующие интервалы не синтезируются из более крупных свечей.")
    if st.button("Загрузить и сопоставить 5 таймфреймов", type="primary", key="load_mtf"):
        try:
            frames = {}
            tfs = ["5m", "15m", "1h", "4h", "1d"]
            if provider in ["Binance", "Bybit"]:
                for tf in tfs:
                    try:
                        frames[tf] = fetch_crypto(provider.lower(), symbol, tf, min(limit, 1000))
                    except Exception:
                        frames[tf] = None
            elif provider == "Yahoo Finance":
                try: frames["1d"] = fetch_yahoo(symbol, "1d", limit)
                except Exception: frames["1d"] = None
            elif provider == "MOEX ISS":
                for tf in ["1h", "1d"]:
                    try: frames[tf] = fetch_moex(symbol, tf, limit)
                    except Exception: frames[tf] = None
            elif "market_df" in st.session_state:
                # CSV can only be resampled to equal/larger intervals than its native cadence.
                from core.multitimeframe import infer_interval_minutes, resample_ohlcv
                base = st.session_state["market_df"]
                mins = infer_interval_minutes(base)
                mins_map = {"5m":5,"15m":15,"1h":60,"4h":240,"1d":1440}
                for tf, target_mins in mins_map.items():
                    if mins <= target_mins:
                        frames[tf] = base if abs(mins-target_mins)<0.5 else resample_ohlcv(base, tf)
            st.session_state["mtf_report"] = analyze_mtf_frames(frames)
        except Exception as exc:
            st.error(f"Ошибка Multi-Timeframe: {exc}")
    mtf = st.session_state.get("mtf_report")
    if mtf:
        if mtf["decision"] == "Преимущество LONG": st.success(mtf["decision"])
        elif mtf["decision"] == "Преимущество SHORT": st.error(mtf["decision"])
        else: st.warning(mtf["decision"])
        st.metric("Взвешенный баланс таймфреймов", f'{mtf["weighted_score"]:+.1f}')
        st.dataframe(mtf["table"], use_container_width=True, hide_index=True)
        st.caption("Правило: 4h и 1d имеют больший вес, чем 5m. При конфликте старшего и младшего таймфрейма не следует автоматически входить против старшего тренда.")
    st.divider()
    st.subheader("Умный поиск точки входа")
    if "enriched_df" in st.session_state:
        if st.button("Найти сетап входа", type="primary", key="find_entry"):
            try:
                st.session_state["entry_setup"] = find_entry_setup(st.session_state["enriched_df"], rr=rr)
            except Exception as exc:
                st.error(f"Ошибка поиска входа: {exc}")
        setup = st.session_state.get("entry_setup")
        if setup:
            if setup["side"] == "LONG": st.success("Сетап LONG — требуется подтверждение")
            elif setup["side"] == "SHORT": st.error("Сетап SHORT — требуется подтверждение")
            else: st.warning("WAIT — подходящего входа пока нет")
            st.write(setup["reason"])
            a,b,c = st.columns(3)
            a.metric("Снятие ликвидности", "Снизу" if setup["bullish_sweep"] else "Сверху" if setup["bearish_sweep"] else "Не найдено")
            b.metric("Пробой + ретест", "Вверх" if setup["bullish_breakout_retest"] else "Вниз" if setup["bearish_breakout_retest"] else "Не подтверждён")
            c.metric("Объём / средний", f'{setup["volume_ratio"]:.2f}x')
            if setup["side"] != "WAIT":
                x,y,z = st.columns(3)
                x.metric("Вход", f'{setup["entry"]:,.6g}')
                y.metric("STOP LOSS", f'{setup["stop_loss"]:,.6g}')
                z.metric("TAKE PROFIT", f'{setup["take_profit"]:,.6g}')
            st.info(setup["confirmation"])
    else:
        st.info("Сначала загрузите OHLCV-котировки на вкладке «Обзор рынка».")

with tabs[9]:
    st.subheader("Качество сигналов — walk-forward validation")
    st.caption("Проверка вне обучающего периода: каждый тестовый отрезок расположен после обучающего. Комиссия и проскальзывание учитываются.")
    folds = st.slider("Количество walk-forward отрезков", 3, 8, 5, key="wf_folds")
    if st.button("Проверить качество сигналов", type="primary", key="run_walk_forward"):
        try:
            st.session_state["wf_report"] = walk_forward_test(st.session_state["market_df"], commission_bps=commission_bps, slippage_bps=slippage_bps, folds=folds)
        except Exception as exc:
            st.error(f"Не удалось выполнить walk-forward тест: {exc}")
    wf = st.session_state.get("wf_report")
    if wf:
        st.markdown("#### Сводка по типам сигналов")
        st.dataframe(wf["summary"], use_container_width=True, hide_index=True)
        st.markdown("#### Результаты отдельных тестовых окон")
        st.dataframe(wf["folds"], use_container_width=True, hide_index=True)
        st.info(wf["method"])
        st.warning("Исторический walk-forward результат не гарантирует будущую доходность. Для надёжной оценки нужны длинная история, разные рыночные режимы, реальные комиссии, funding, спред, ликвидность и проверка на другом источнике данных.")
    else:
        if "market_df" not in st.session_state:
            st.info("Сначала загрузите исторические OHLCV-данные. Для walk-forward желательно не менее 1 000 свечей.")
        else:
            st.info("Нажмите кнопку, чтобы выполнить последовательную проверку на отложенных временных отрезках.")

with tabs[10]:
    st.subheader("Semi-auto workflow — подтверждение сделки")
    st.caption("В этой сборке реализована безопасная бумажная торговля. Никакие реальные заявки не отправляются.")
    if "enriched_df" in st.session_state:
        df = st.session_state["enriched_df"]
        fc = st.session_state.get("forecast")
        if fc:
            direction = fc["signal"]
            plan = make_trade_plan(df, direction, capital, risk_pct, rr)
            if plan["side"] == "WAIT":
                st.info("Ансамбль не сформировал торговый сигнал.")
            else:
                a,b,c,d = st.columns(4)
                a.metric("Сторона", plan["side"])
                b.metric("Вход (ориентир)", f'{plan["entry"]:.4f}')
                c.metric("Stop Loss", f'{plan["stop_loss"]:.4f}')
                d.metric("Take Profit", f'{plan["take_profit"]:.4f}')
                st.write(f'Размер позиции: **{plan["quantity"]:.6f}** · денежный риск: **{plan["risk_amount"]:.2f}**')
                st.warning("Подтверждение ниже создаст виртуальную сделку в локальном журнале. Цена исполнения в реальном рынке может отличаться.")
                if st.button("Подтвердить PAPER-сделку", type="primary"):
                    create_paper_trade(symbol, plan)
                    st.success("Бумажная сделка зарегистрирована.")
    if "enriched_df" in st.session_state:
        last_bar = st.session_state["enriched_df"].iloc[-1]
        closed_by_levels = monitor_paper_trades(symbol, float(last_bar.high), float(last_bar.low), float(last_bar.close))
        for tid, reason, px in closed_by_levels:
            st.warning(f"Позиция #{tid} закрыта по {reason} на {px:.4f}")
    open_trades = list_trades(open_only=True)
    st.write("**Открытые бумажные позиции**")
    if open_trades.empty:
        st.caption("Открытых позиций нет.")
    else:
        st.dataframe(open_trades, use_container_width=True)
        trade_id = st.selectbox("Выберите позицию для закрытия", open_trades["id"].tolist())
        exit_price = st.number_input("Цена закрытия", min_value=0.000001, value=1.0, format="%.6f")
        col_manual, col_emergency = st.columns(2)
        with col_manual:
            if st.button("Закрыть по указанной цене"):
                close_paper_trade(int(trade_id), float(exit_price))
                st.success("Позиция закрыта.")
        with col_emergency:
            if st.button("Аварийное закрытие по последней цене", type="primary") and "enriched_df" in st.session_state:
                close_paper_trade(int(trade_id), float(st.session_state["enriched_df"].iloc[-1].close))
                st.success("Позиция аварийно закрыта по последней доступной цене.")
    st.write("**История бумажных сделок**")
    st.dataframe(list_trades(), use_container_width=True)


with tabs[0]:
    st.subheader("Онлайн-сканер: 30 перспективных криптопар + XAU/USD")
    st.caption("Сканер анализирует ликвидные USDT-пары Binance/Bybit, ранжирует их по технической перспективности и добавляет только золото XAU/USD. Монитор открытых позиций проверяет минутные свечи; поиск новых сигналов выполняется отдельно примерно каждые 15 минут через GitHub Actions.")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        venues = st.multiselect("Криптобиржи", ["Binance", "Bybit"], default=["Binance", "Bybit"])
    with sc2:
        scan_tf = st.selectbox("Таймфрейм сканера", ["15m", "1h", "4h", "1d"], index=1)
    with sc3:
        scan_limit = st.selectbox("История свечей", [120, 180, 300], index=1)
    st.write("Результат: до 30 уникальных криптопар с лучшими техническими сетапами и отдельная котировка XAU/USD. Сначала отбираются ликвидные пары, затем они ранжируются по тренду, MACD, RSI, объёму и режиму рынка.")
    if st.button("Запустить сканирование рынков", type="primary", key="run_universe_scan"):
        venue_ids=[v.lower() for v in venues]
        if not venue_ids:
            st.error("Выберите хотя бы одну криптобиржу.")
        else:
            try:
                with st.spinner("Сканирую ликвидные криптопары на Binance/Bybit и XAU/USD..."):
                    scan_df, scan_errors = scan_universe(tuple(venue_ids), 30, 30, scan_tf, int(scan_limit))
                st.session_state["universe_scan_df"]=scan_df
                st.session_state["universe_scan_errors"]=scan_errors
                if scan_df.empty:
                    st.error("Не удалось получить данные ни по одному инструменту. Проверьте доступность источников.")
                else:
                    # Only the five highest-ranked setups with strict 4h + 1d trend agreement
                    # are eligible for journal insertion and Telegram notification.
                    strong=scan_df[(scan_df.side!="WAIT") & (scan_df.score>=5) & (scan_df.mtf_confirmed==True)].copy()
                    strong=strong.sort_values("opportunity_score",ascending=False).head(5)
                    # Update experiment positions from the latest candle extremes.
                    exp_now=active_experiment()
                    if exp_now and exp_now["status"]=="ACTIVE":
                        closed_now=update_prices(scan_df[["symbol","high","low","price","bar_time"]].rename(columns={"price":"close"}).dropna().to_dict("records"))
                        for trade in closed_now:
                            note=(f"🧪 ЗАКРЫТИЕ Paper Trading: {trade['symbol']} "
                                  f"{'ПОКУПКА' if trade['side']=='LONG' else 'ПРОДАЖА'} · {trade['reason']} "
                                  f"· цена выхода {trade['exit']:.8g} · результат ${trade['net_pnl']:.2f}")
                            send_telegram(note); log_event("CLOSE",note)
                        opened_now=open_top_trades(strong, max_open=10)
                        for trade in opened_now:
                            note=(f"🧪 ОТКРЫТИЕ Paper Trading: {trade['symbol']} "
                                  f"{'ПОКУПКА' if trade['side']=='LONG' else 'ПРОДАЖА'} "
                                  f"· вход {trade['entry']:.8g} · стоп {trade['sl']:.8g} · цель {trade['tp']:.8g} "
                                  f"· риск ${trade['risk']:.2f}")
                            send_telegram(note); log_event("OPEN",note)
                        send_telegram(f"🧪 Trading AI: проверено инструментов — {len(scan_df)}; "
                                      f"подтверждено 4Ч+1Д — {int(scan_df.mtf_confirmed.fillna(False).sum())}; "
                                      f"лучших сигналов — {len(strong)}; открыто — {len(opened_now)}; закрыто — {len(closed_now)}.")
                        if finish_if_due():
                            summary_end, _, _ = experiment_report()
                            final_msg=(f"🏁 Trading AI: 10-дневное испытание завершено. "
                              f"Сделок закрыто: {summary_end['closed_trades']}; Win Rate: {summary_end['win_rate']:.1f}%; "
                              f"Net P&L: ${summary_end['net_pnl']:.2f}; ROI: {summary_end['roi_pct']:.2f}%. "
                              f"Подробный отчёт доступен в приложении.")
                            send_telegram(final_msg); log_event("FINISH",final_msg)
                    inserted_count=0; notified_count=0
                    for _, row in strong.iterrows():
                        key_src=f"{row.get('venue')}|{row.get('symbol')}|{row.get('timeframe')}|{row.get('side')}|{row.get('price')}"
                        signal_key=hashlib.sha256(key_src.encode()).hexdigest()
                        rec={**row.to_dict(),"signal_key":signal_key,"strategy":"scanner-v1","notes":row.get("reasons","")}
                        if record_signal(rec):
                            inserted_count+=1
                            try:
                                try:
                                    token=st.secrets["TELEGRAM_BOT_TOKEN"]
                                    chat=st.secrets["TELEGRAM_CHAT_ID"]
                                except Exception:
                                    token=chat=""
                                if token and chat:
                                    msg=(f"Trading AI TOP-5: {row['side']}\n{row.get('display_name', row['symbol'])} ({row['venue']}) · {row['timeframe']}\nЦена: {row['price']:.8g}\nSL: {row['stop_loss']:.8g} · TP: {row['take_profit']:.8g}\nРейтинг: {row['opportunity_score']}/100 · Score: {row['score']}\nТренд 4h: {row.get('trend_4h')} · 1d: {row.get('trend_1d')}\nРежим: {row['regime']}\n{row.get('reasons','')}")
                                    resp=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",json={"chat_id":chat,"text":msg},timeout=10)
                                    resp.raise_for_status(); notified_count+=1
                            except Exception as te:
                                st.warning(f"Сигнал записан, но Telegram-уведомление не отправлено: {str(te)[:160]}")
                    st.success(f"Сканирование завершено: {len(scan_df)} инструментов; подтверждено по 4h+1d: {int(scan_df.mtf_confirmed.fillna(False).sum())}; отобрано не более 5 лучших: {len(strong)}; новых сигналов записано: {inserted_count}; Telegram: {notified_count}.")
            except Exception as exc:
                st.error(f"Ошибка сканирования: {exc}")
    scan_df=st.session_state.get("universe_scan_df")
    if scan_df is not None and not scan_df.empty:
        strong=scan_df[(scan_df.side!="WAIT") & (scan_df.score>=5) & (scan_df.mtf_confirmed.fillna(False)==True)].copy()
        strong=strong.sort_values("opportunity_score",ascending=False).head(5)
        a,b,c,d=st.columns(4)
        a.metric("Криптопар",int((scan_df.asset_class=="CRYPTO").sum())); b.metric("LONG",int((scan_df.side=="LONG").sum())); c.metric("SHORT",int((scan_df.side=="SHORT").sum())); d.metric("XAU/USD",int(scan_df.asset_class.astype(str).str.startswith("GOLD").sum()))
        st.markdown("#### Топ перспективных криптопар и золото")
        actionable=scan_df.copy()
        if actionable.empty: st.info("Пока нет доступных котировок. Проверь ошибки источников.")
        else:
            table_cols=[c for c in ["rank","venue","asset_class","display_name","symbol","timeframe","bar_time","side","opportunity_score","score","trend_4h","trend_1d","mtf_confirmed","price","entry","stop_loss","take_profit","regime","rsi","volume_ratio","change_24h_pct","reasons"] if c in actionable.columns]
            ru_names={"rank":"Место","venue":"Источник","asset_class":"Класс актива","display_name":"Инструмент","symbol":"Торговая пара","timeframe":"Таймфрейм","bar_time":"Время свечи","side":"Направление","opportunity_score":"Оценка возможности","score":"Баллы сигнала","trend_4h":"Тренд 4 часа","trend_1d":"Тренд 1 день","mtf_confirmed":"Подтверждение 4Ч+1Д","price":"Текущая цена","entry":"Вход","stop_loss":"Стоп-лосс","take_profit":"Тейк-профит","regime":"Режим рынка","rsi":"Индекс RSI","volume_ratio":"Объём к среднему","change_24h_pct":"Изменение за 24 ч, %","reasons":"Причины сигнала"}
            view=actionable[table_cols].copy()
            view["side"]=view["side"].map({"LONG":"ПОКУПКА","SHORT":"ПРОДАЖА","WAIT":"ОЖИДАНИЕ"}).fillna(view["side"])
            for col in ("trend_4h","trend_1d"):
                if col in view: view[col]=view[col].map({"LONG":"РОСТ","SHORT":"СНИЖЕНИЕ","NEUTRAL":"НЕЙТРАЛЬНО","UNKNOWN":"НЕТ ДАННЫХ"}).fillna(view[col])
            if "regime" in view:
                view["regime"]=view["regime"].map({"UPTREND":"ВОСХОДЯЩИЙ","DOWNTREND":"НИСХОДЯЩИЙ","RANGE":"БОКОВОЙ","HIGH_VOLATILITY":"ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ"}).fillna(view["regime"])
            view=view.rename(columns=ru_names)
            st.dataframe(view,use_container_width=True,hide_index=True)
            st.markdown("#### Пять лучших сигналов — графики")
            for rank,(_,signal) in enumerate(strong.iterrows(),start=1):
                direction="ПОКУПКА" if signal["side"]=="LONG" else "ПРОДАЖА"
                st.markdown(f"**{rank}. {signal['symbol']} — {direction}** · оценка {signal['opportunity_score']}")
                try:
                    history=fetch_chart_ohlcv(signal)
                    fig=draw_signal_chart(signal,history)
                    st.pyplot(fig,use_container_width=True)
                    plt.close(fig)
                    st.caption(f"Вход: {signal['entry']:.8g} · Стоп-лосс: {signal['stop_loss']:.8g} · Тейк-профит: {signal['take_profit']:.8g} · Подтверждение 4Ч: {signal['trend_4h']} · 1Д: {signal['trend_1d']}")
                except Exception as chart_error:
                    st.warning(f"Не удалось построить график {signal['symbol']}: {str(chart_error)[:160]}")
        with st.expander("Полный список инструментов"):
            all_cols=[c for c in ["venue","asset_class","display_name","symbol","side","score","price","regime","rsi","volume_ratio"] if c in scan_df.columns]
            all_view=scan_df[all_cols].copy()
            all_view=all_view.rename(columns={"venue":"Источник","asset_class":"Класс актива","display_name":"Инструмент","symbol":"Торговая пара","side":"Направление","score":"Баллы сигнала","price":"Цена","regime":"Режим рынка","rsi":"Индекс RSI","volume_ratio":"Объём к среднему"})
            if "Направление" in all_view: all_view["Направление"]=all_view["Направление"].map({"LONG":"ПОКУПКА","SHORT":"ПРОДАЖА","WAIT":"ОЖИДАНИЕ"}).fillna(all_view["Направление"])
            st.dataframe(all_view,use_container_width=True,hide_index=True)
        st.download_button("Скачать результаты сканирования CSV",scan_df.to_csv(index=False),"trading_ai_universe_scan.csv","text/csv")
    errors=st.session_state.get("universe_scan_errors",[])
    if errors:
        with st.expander(f"Недоступные инструменты / ошибки источников ({len(errors)})"):
            for err in errors[:100]: st.write("• "+str(err))
    st.info("XAU/USD берётся из Yahoo Finance: сначала XAUUSD=X, затем GC=F как резервный фьючерсный прокси. Котировки могут задерживаться и не заменяют цену исполнения у брокера. Список криптопар ограничивается 30 уникальными символами после ранжирования.")
    st.markdown("#### Настройка Telegram")
    st.caption("Не вводите токен в эту форму и не отправляйте его в чат. Добавьте в Streamlit → Settings → Secrets:")
    st.code('TELEGRAM_BOT_TOKEN = "токен_бота"\nTELEGRAM_CHAT_ID = "chat_id"',language="toml")
    st.caption("Telegram получает не более 5 лучших новых сигналов за один проход: score ≥ 5, направление LONG/SHORT и обязательное совпадение тренда на 4h и 1d. Если подтверждения нет, инструмент остаётся в таблице, но уведомление не отправляется. Повторные сигналы не дублируются.")

with tabs[1]:
    st.subheader("Журнал сигналов и аналитика")
    journal=list_signals(); stats=performance_summary(journal)
    a,b,c,d,e=st.columns(5)
    a.metric("Сигналов",stats["signals"]); b.metric("Закрыто",stats["closed"]); c.metric("Win rate",f"{stats['win_rate']:.1f}%"); d.metric("Средний P&L",f"{stats['avg_pnl']:.3f}%"); e.metric("Суммарный P&L",f"{stats['total_pnl']:.3f}%")
    if journal.empty: st.info("Журнал пока пуст. Запустите онлайн-сканер: сильные сигналы будут автоматически записываться сюда.")
    else:
        st.dataframe(journal,use_container_width=True,hide_index=True)
        open_rows=journal[journal.status=="OPEN"]
        if not open_rows.empty:
            st.markdown("#### Ручная фиксация результата сигнала")
            selected=st.selectbox("Сигнал для закрытия",open_rows.id.tolist(),key="journal_close_id")
            exit_px=st.number_input("Фактическая цена выхода",min_value=0.00000001,value=1.0,format="%.8f",key="journal_exit_px")
            if st.button("Закрыть сигнал и рассчитать P&L",key="journal_close_btn"):
                close_signal(int(selected),float(exit_px)); st.success("Результат сохранён."); st.rerun()
        st.download_button("Скачать журнал CSV",journal.to_csv(index=False),"trading_ai_signal_journal.csv","text/csv")
    st.caption("Win rate и P&L рассчитываются только по сигналам, которые пользователь вручную закрыл с фактической ценой выхода. До накопления выборки эти показатели не являются оценкой будущей доходности.")


with tabs[11]:
    st.subheader("Испытание капитала $2,000 — 10 дней · внутридневная торговля")
    st.caption("Изолированный Paper Trading контур. Реальные ордера не отправляются. Риск на сделку — 5% от текущего реализованного капитала (на старте $100); одновременно не более 10 позиций. Размер позиции соблюдает полный денежный риск и может подразумевать плечо в симуляции; это не реальное исполнение. Сделки внутридневные: если позиция не закрылась по SL/TP, она фиксируется по последней наблюдавшейся цене дня UTC. Контроль выполняется при запуске сканера, поэтому сканирование нужно запускать регулярно.")
    exp=active_experiment()
    if exp is None or exp["status"]!="ACTIVE":
        st.warning("Десятидневное испытание не запущено.")
        if st.button("Начать испытание: $2,000 / 10 дней / риск 5%",type="primary",key="start_10d_experiment"):
            exp=start_experiment()
            ok,msg=send_telegram("🧪 Trading AI: начато 10-дневное Paper Trading испытание. Баланс $2,000; риск 5% на сделку ($100 при старте); максимум 10 открытых позиций; закрытие в пределах UTC-торгового дня; комиссия 0.10% на сторону.")
            st.success("Испытание запущено. Telegram: "+("отправлено" if ok else msg))
            st.rerun()
    else:
        st.info(f"Статус: {exp['status']} · старт: {exp['started_at']} · окончание: {exp['ends_at']}")
        summary,trades,events=experiment_report()
        if summary:
            a,b,c,d=st.columns(4)
            a.metric("Стартовый капитал",f"${summary['starting_capital']:,.2f}")
            b.metric("Реализованный баланс",f"${summary['ending_equity_realized']:,.2f}")
            c.metric("Чистый P&L",f"${summary['net_pnl']:,.2f}",f"{summary['roi_pct']:.2f}%")
            d.metric("Win Rate",f"{summary['win_rate']:.1f}%")
            e,f,g,h=st.columns(4)
            e.metric("Сделок открыто",summary["signals_opened"])
            f.metric("Закрыто",summary["closed_trades"])
            g.metric("Открытых",summary["open_trades"])
            h.metric("Комиссии",f"${summary['fee_total']:.2f}")
            st.write(f"Profit Factor: {summary['profit_factor']:.2f}" if summary['profit_factor']!=float("inf") else "Profit Factor: ∞ (убыточных закрытых сделок пока нет)")
        st.markdown("#### Сделки испытания")
        if trades.empty: st.info("Сделок пока нет. Запусти сканирование рынков.")
        else: st.dataframe(trades,use_container_width=True,hide_index=True)
        st.markdown("#### События")
        if not events.empty: st.dataframe(events,use_container_width=True,hide_index=True)
        if st.button("Проверить окончание срока и сформировать отчёт",key="check_experiment_due"):
            if finish_if_due():
                send_telegram("🏁 Trading AI: 10-дневное испытание завершено. Откройте раздел «Испытание $2,000» для итогового отчёта.")
            st.rerun()
        if not trades.empty:
            st.download_button("Скачать сделки CSV",trades.to_csv(index=False),"paper_experiment_trades.csv","text/csv")
        if not events.empty:
            st.download_button("Скачать журнал событий CSV",events.to_csv(index=False),"paper_experiment_events.csv","text/csv")
