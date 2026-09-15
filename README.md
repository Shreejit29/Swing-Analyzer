# AI Swing Stock Analyzer

A simple Streamlit application for:

1. **Stock analysis** — technical indicators, Gradient Boosting probability, regime, BUY/SELL/WAIT, entry, stop and target.
2. **Backtesting** — walk-forward AI strategy testing with equity curve, win rate, profit factor and drawdown.

## Data

Yahoo Finance is the default source. The app also accepts OHLCV CSV files so you can backtest longer histories when the requested Yahoo history is unavailable.

CSV columns required:

```text
Date, Open, High, Low, Close, Volume
```

`Datetime` is also accepted instead of `Date`.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Design

The application intentionally does **not** contain a research/approval/model-registry workflow. The model is a practical Gradient Boosting classifier using technical and price/volume features.

### Important backtesting assumption

A prediction at bar `t` is acted on at the **next bar's open**. Stop/target handling is conservative: when both could be hit in the same OHLC bar, the stop is checked first.

This is a research tool, not financial advice or a guarantee of future returns.
