# Trading AI v2 — research beta

## Included modules
1. AI Vision: image loading, OpenCV color/edge analysis and explicit limits.
2. Smart Money: swing pivots, BOS/CHoCH, three-candle FVG, heuristic order blocks, equal-high/low liquidity clusters.
3. Prediction Engine: Random Forest, Extra Trees, HistGradientBoosting and Logistic Regression; chronological 80/20 split.
4. Backtesting: EMA+RSI, MACD crossover and RSI mean-reversion strategies; one-bar signal delay, fee/slippage model, equity curve, drawdown, win rate and profit factor.
5. Multi-market: Binance/Bybit via CCXT, Yahoo Finance futures proxies for selected metals, MOEX ISS futures candles, CSV.
6. Semi-auto: user-approved paper trades, stop/target reference levels, paper position journal.

## Local start
Python 3.10+ recommended.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Create a GitHub account and a new repository.
2. Upload the contents of this project to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Create app -> select repository -> branch `main` -> main file `app.py` -> Deploy.
5. Do not add exchange API keys. This beta only fetches public market data and uses paper trading.

Official deployment docs: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Data source notes
- Binance/Bybit public OHLCV uses CCXT. Symbols and timeframes must be supported by the selected exchange.
- Yahoo Finance futures tickers are indicative data proxies; availability, contract continuity, delays and licensing vary. For institutional/production use, connect licensed CME/COMEX/LME data.
- MOEX ISS endpoint targets FORTS candles and requires a valid SECID; endpoint/market availability may vary.
- CSV schema: timestamp,open,high,low,close,volume.

## Important limitations / not production-ready
- AI Vision currently performs image-level CV statistics, not reliable OCR or full candle-price reconstruction. It does not infer a trade direction from a screenshot.
- Smart Money concepts are algorithmic approximations. BOS/CHoCH and Order Blocks are definition-sensitive and need user-configurable validation.
- ML probabilities are raw ensemble averages, not calibrated probabilities. The 80/20 holdout is a first check, not robust walk-forward validation. Avoid interpreting model agreement as probability of profit.
- Backtest is a baseline close-to-close strategy. It does not simulate intrabar stop/target ordering, funding, borrow, margin, liquidation, market impact, partial fills or exchange lot constraints.
- Paper trading is local SQLite only. On each data refresh, open paper positions are checked against candle high/low for SL/TP (if both touched in one candle, SL is assumed first); manual and emergency close are available. This is not a live broker.
- No live order execution is implemented. Never place real API secrets in source code or public repositories.
- This software is educational/research software, not investment advice.
