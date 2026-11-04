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


## Scanner v6 scope
- Up to 30 unique USDT cryptocurrency pairs, selected from liquid Binance/Bybit spot markets and ranked by technical opportunity score.
- One additional gold instrument: XAU/USD spot proxy (`XAUUSD=X`) from Yahoo Finance.
- No broad commodity basket is scanned in this version.
- Refresh is user-triggered; quotes may be delayed and should be checked with the execution broker.


## v7 — Top-5 with higher-timeframe confirmation
- The scanner still displays up to 30 ranked crypto pairs plus XAU/USD.
- Telegram and new-signal journal insertion are limited to at most five setups per scan.
- A setup must be LONG/SHORT, have the existing minimum signal score, and have matching trend direction on both 4h and 1d.
- Higher-timeframe direction requires EMA20/EMA50/EMA200 alignment plus MACD agreement. Missing data is UNKNOWN and fails closed (no alert).
- Ranking is based on technical opportunity score; it is not a calibrated probability of profit.


## Paper Capital Experiment (v8)
- Dedicated 10-day paper experiment, starting equity USD 2,000.
- Risk budget 5% of realized equity per trade (USD 100 at the USD 2,000 start); max 10 simultaneous positions.
- Conservative simulated fees: 0.10% per side; update `FEE_RATE` in `core/experiment.py` to match the intended venue fee tier.
- Only signals with LONG/SHORT and strict 4h + 1d agreement are eligible; up to 5 candidates are considered per scan.
- Trade events, scan summaries, opens and closes are sent to Telegram when the scanner is run and Telegram Secrets are configured.
- OHLC candle monitoring uses conservative stop-first logic if SL and TP are both touched in the same candle.
- IMPORTANT: Streamlit Community Cloud does not guarantee continuous background execution. This build processes a scan and updates simulated positions when the user runs the scanner. It is not an always-on 24/7 worker. For continuous 10-day observation, schedule an external worker/cron or keep triggering scans regularly. A missed scan can miss intrabar SL/TP touches.
- This is a simulation, not a forecast or guarantee of future returns. Results are only as reliable as the market data and scan cadence.


## Updated intraday capital test (v8.1)
- Starting virtual capital: USD 2,000.
- Duration: 10 calendar days from the moment the user starts the experiment.
- Risk budget: 5% of current realized equity per trade; USD 100 at the initial balance.
- Maximum simultaneous open positions: 10.
- Intraday-only handling: positions are not intentionally carried into the next UTC date. If SL/TP has not closed a trade by the end of its opening UTC day, it is closed using the last observed close from that UTC day.
- SL is prioritized if both stop and target are touched within the same candle.
- Fee assumption remains 0.10% per side.
- The experiment is evaluated only when the user runs a scan. It is not a continuously running service; scan frequently enough to observe intraday prices. Streamlit Community Cloud does not guarantee background execution or durable local SQLite storage across redeploys/sleeps.

- Position size is calculated to the full stated stop-loss risk. Depending on stop distance, the simulated notional can exceed USD 2,000 and therefore imply leverage/margin; this is a paper model only. Loss at SL can exceed the nominal 5% risk slightly after fees.


## Автоматическое круглосуточное Paper Trading сканирование

Автоматический worker запускается GitHub Actions каждые 15 минут (UTC), независимо от открытой вкладки Streamlit:
- сканирует Binance + Bybit (до 30 уникальных USDT-пар) и XAU/USD;
- требует совпадения направления на 4h и 1d;
- ведёт до 10 виртуальных позиций, риск 5% капитала на сделку;
- проверяет открытые позиции отдельным запросом OHLCV, даже если инструмент выбыл из TOP-30;
- отправляет Telegram-сводки, открытия, закрытия и ошибки;
- сохраняет SQLite-базу в GitHub-репозитории отдельным commit.

### Однократная настройка GitHub
1. В репозитории откройте Settings → Secrets and variables → Actions → New repository secret.
2. Создайте `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID` (значения берите из уже настроенного Telegram-бота; не помещайте их в код).
3. Убедитесь, что в Settings → Actions → General разрешены workflow permissions: Read and write permissions.
4. Загрузите `auto_worker.py` и `.github/workflows/trading-ai-autoscan.yml` в ветку `main`.
5. Откройте Actions → Trading AI 24x7 Paper Scanner → Run workflow для тестового запуска.
6. После успешного запуска проверьте Telegram и наличие `trading_ai_paper.db` в репозитории.

GitHub Actions scheduled workflow может задерживаться и не является биржевым low-latency сервисом. Сканирование выполняется примерно раз в 15 минут, а не непрерывно каждую секунду. Время запуска зависит от очереди GitHub. Репозиторий будет содержать базу Paper Trading; сделайте его Private. Не добавляйте API-ключи бирж. Worker использует публичные рыночные данные и не отправляет реальные ордера.

SQLite файл в Git будет содержать журнал сделок и событий. Не делайте репозиторий публичным, если не хотите раскрывать историю эксперимента.
