# Crypto Scanner

`crypto_scanner` is a simple Python market scanner that pulls 15-minute and 1-hour OHLCV data for a watchlist of major crypto pairs, calculates trend indicators, and prints multi-timeframe pullback trade setups.

## Monitored Coins

- BTC
- ETH
- SOL
- LINK
- AVAX
- SUI

## Features

- Fetches OHLCV candles from Coinbase with `ccxt`
- Calculates EMA 20, EMA 50, ATR 14, RSI 14, 20-period ATR median, and 20-period average volume
- Uses 1-hour trend confirmation to qualify 15-minute pullback setups
- Scores each signal from 0 to 100 using multi-timeframe alignment, RSI quality, ATR expansion, and EMA separation
- Runs continuously, refreshing every 5 minutes
- Clears previous terminal output and prints a timestamp for each scan
- Prints entry, stop-loss, and 2R target levels in a clean summary table
- Includes a Streamlit web app that reuses the same scanner logic

## Project Structure

```text
crypto_scanner/
    app.py
    main.py
    data.py
    indicators.py
    signals.py
    config.py
    requirements.txt
    README.md
```

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Scanner

From inside the `crypto_scanner` folder, run:

```bash
python main.py
```

The scanner will keep running until you stop it with `Ctrl+C`.

### Streamlit Web App

From inside the `crypto_scanner` folder, run:

```bash
streamlit run app.py
```

The Streamlit app shows the latest scan in a web table, includes a refresh button, and color-codes setup values:

- `PULLBACK_LONG` in green
- `PULLBACK_SHORT` in red
- `NONE` in gray

Example output:

```text
Symbol 1H Trend 15m Trend Setup          Score Entry Price Stop Loss Target (2R)
   BTC LONG     LONG      PULLBACK_LONG     88 ...         ...       ...
   ETH SHORT    SHORT     NONE              52 n/a         n/a       n/a
```

## Strategy Rules

- 1-hour trend is `LONG` when EMA20 is above EMA50.
- 1-hour trend is `SHORT` when EMA20 is below EMA50.
- 15-minute trend is `LONG` when EMA20 is above EMA50.
- 15-minute trend is `SHORT` when EMA20 is below EMA50.
- `PULLBACK_LONG` requires:
  - 1-hour trend is `LONG`
  - 15-minute trend is `LONG`
  - latest candle trades through EMA20
  - RSI 14 between 40 and 55
  - ATR 14 above its recent 20-period median
- `PULLBACK_SHORT` requires:
  - 1-hour trend is `SHORT`
  - 15-minute trend is `SHORT`
  - latest candle trades through EMA20
  - RSI 14 between 45 and 60
  - ATR 14 above its recent 20-period median

## Notes

- The scanner evaluates both 15-minute and 1-hour candles.
- The scanner refreshes every 5 minutes.
- The scanner uses Coinbase and resolves the best available market for each asset by checking `USD`, `USDC`, and `USDT` in that order.
- If a requested asset is not listed on Coinbase with one of those quote currencies, the table will show `UNAVAILABLE` for that symbol instead of crashing.
- Entry price is the current EMA20 value when a setup is detected. Stop loss is anchored beyond the pullback candle, and the target is calculated at `2R`.
- The score ranges from 0 to 100 and rewards higher-timeframe alignment, stronger RSI location, volatility expansion, and wider EMA separation.
- The Streamlit app calls the existing scanner code instead of duplicating the market logic.
- Exchange and symbol settings can be adjusted in `config.py`.
- Public exchange APIs can occasionally rate-limit or reject requests; if that happens, the table will show an error message for the affected symbol.
