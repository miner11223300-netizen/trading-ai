# Trading AI v4 — Multi-Timeframe & Signal Validation

Research-only Streamlit application. All orders remain Paper Trading only; no exchange API keys or real order execution are included.

## New modules
- `core/multitimeframe.py`: 5m / 15m / 1h / 4h / 1d trend alignment. Crypto exchanges are queried independently per timeframe. Missing intervals are shown as unavailable; the app never invents lower-timeframe candles from higher-timeframe data.
- `core/entry_finder.py`: heuristic liquidity sweep, breakout/retest, volume and structure confluence with entry / stop / target estimates. It returns WAIT when evidence is insufficient.
- `core/signal_quality.py`: expanding-window walk-forward evaluation with out-of-sample return, max drawdown, win rate, profit factor, trade count and fold results. EMA parameters are selected only on past training data.

## Install
`pip install -r requirements.txt`

## Run
`streamlit run app.py`

## Important limitations
These are research heuristics, not financial advice or a validated predictive edge. Walk-forward tests require adequate history and can still overfit. Results do not fully model order-book liquidity, partial fills, funding, borrow costs, contract multipliers, exchange outages, intrabar stop/target sequencing or taxes. Yahoo Finance metal tickers are proxies, not licensed exchange feeds. For metals, only provider-supported intervals are displayed. Verify instrument specifications and costs independently.

For the full 5-timeframe crypto view, use Binance or Bybit and load at least 60 candles on each interval; 200+ per interval is preferable for EMA200 context. For walk-forward tests use at least 1,000 candles where possible.

## Trading AI v5 — Universe Scanner, Smart Money Pro, Journal & Telegram

This release preserves the v4 Multi-Timeframe Analysis, Smart Entry and Walk-Forward Signal Quality modules, and adds:
- `core/scanner.py`: cross-exchange top-30 unique USDT spot crypto universe by 24h quote volume from Binance/Bybit, plus up to 30 commodity continuous-futures tickers from Yahoo Finance.
- `core/smc_pro.py`: heuristic liquidity sweeps, BOS/CHoCH, FVG, Order Blocks and equal highs/lows.
- `core/regime.py`: trend, range, high-volatility and transition labels.
- `core/journal_analytics.py`: SQLite scanner signal journal and realized performance after manual signal close.

Open **🌐 Онлайн-сканер 60 рынков**, choose Binance/Bybit and timeframe, then click **Запустить сканирование рынков**. A scan is manual (not a continuously running background service) to control API limits and runtime. It analyzes up to 30 distinct crypto assets across selected exchanges and up to 30 commodity symbols. Missing provider tickers are reported rather than fabricated.

### Telegram
Add to Streamlit Community Cloud → App → Settings → Secrets (never commit secrets to GitHub):

```toml
TELEGRAM_BOT_TOKEN = "..."
TELEGRAM_CHAT_ID = "..."
```
Only newly journaled scanner signals with score >= 5 are eligible for notification. The app sends no real orders. Journal win rate and P&L are based on user-recorded exit prices and are not meaningful until enough closed examples exist.

### Data and risk limitations
Crypto spot OHLCV and ticker data come from public Binance/Bybit endpoints. Commodity quotes use Yahoo Finance continuous futures where available; the configured 30-symbol universe includes a small number of explicitly named ETF/index proxies (not direct commodity futures). Yahoo data may be delayed or unavailable. Signal scores and regime labels are heuristic filters, not calibrated probabilities or investment advice. Use paper trading and independent out-of-sample validation before considering live execution.
