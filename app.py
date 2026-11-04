import streamlit as st
import pandas as pd
from core.market_data import load_csv, fetch_crypto, fetch_yahoo, fetch_moex
from core.indicators import enrich
from core.vision import inspect_chart
from core.smc import analyze_smc
from core.prediction import ensemble_forecast
from core.backtest import run_backtest
from core.risk import make_trade_plan
from core.paper_broker import init_db, create_paper_trade, close_paper_trade, list_trades, monitor_paper_trades

st.set_page_config(page_title="Trading AI", page_icon="📊", layout="wide")
init_db()
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

tabs = st.tabs(["Обзор рынка", "AI Vision", "Smart Money", "Prediction Engine", "Backtesting", "Paper Trading"])

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

with tabs[0]:
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
        st.line_chart(df.set_index("timestamp")[["close","ema20","ema50","ema200"]].tail(300))
        st.dataframe(df.tail(30), use_container_width=True)
        st.download_button("Скачать подготовленные данные CSV", df.to_csv(index=False), "trading_ai_data.csv", "text/csv")
    else:
        st.info("Выберите источник и загрузите данные. Для металлов доступны Yahoo Finance и MOEX ISS; доступность конкретных контрактов зависит от поставщика.")

with tabs[1]:
    st.subheader("AI Vision — анализ скриншота")
    img = st.file_uploader("Загрузите скриншот TradingView / терминала", type=["png","jpg","jpeg","webp"], key="vision")
    if img:
        st.image(img, use_container_width=True)
        if st.button("Исследовать изображение"):
            try:
                report = inspect_chart(img.getvalue())
                st.json(report)
                st.caption("Распознавание свечей и цветовых областей — вспомогательная компьютерная эвристика. Точные цены, тикер и таймфрейм нельзя надёжно восстановить без OCR/легенды и исходных OHLCV.")
            except Exception as e:
                st.error(f"Ошибка обработки изображения: {e}")
    else:
        st.info("Загрузите скриншот. Для точных уровней и сигналов дополнительно используйте OHLCV из выбранного источника.")

with tabs[2]:
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
    else:
        st.info("Сначала загрузите рыночные данные.")

with tabs[3]:
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
                st.caption(f'Горизонт: {horizon} свечей. Доля совпадения моделей: {forecast["agreement"]*100:.1f}%. Вероятности — ансамблевые оценки, не гарантия и не калиброванный прогноз.')
                st.dataframe(pd.DataFrame(forecast["model_metrics"]), use_container_width=True)
                for note in forecast["notes"]:
                    st.write("• " + note)
            except Exception as e:
                st.error(f"Не удалось обучить модели: {e}")
    else:
        st.info("Сначала загрузите рыночные данные.")

with tabs[4]:
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

with tabs[5]:
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
