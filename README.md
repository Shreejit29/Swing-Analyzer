# 📈 AI Swing Stock Analyzer

An AI-assisted swing trading analysis platform built with **Python, Streamlit, technical analysis, price action, volume analysis, multi-timeframe analysis, and Gradient Boosting**.

The application is designed to transform raw OHLCV market data into a structured swing-trading analysis.

---

## 🚀 Features

### 📊 Stock Analyzer

Analyze an individual stock using only two inputs:

- Stock symbol
- Prediction horizon

The application automatically performs:

- Market data acquisition
- Technical indicator calculation
- Price-action analysis
- Volume analysis
- Market-regime detection
- Multi-timeframe analysis
- Machine-learning prediction
- Probability estimation
- Signal generation
- Trade-plan generation

---

## 🤖 Machine Learning

The current prediction engine uses:

**Gradient Boosting Classifier**

The model is trained using engineered market features rather than relying on a single technical indicator.

The feature pipeline includes:

- Moving averages
- EMA relationships
- EMA slopes
- RSI
- MACD
- ATR
- Bollinger Bands
- Rate of Change
- Volume relationships
- OBV
- Candle structure
- Support/resistance
- Breakouts
- Breakdowns
- Market regime
- Weekly trend
- Monthly trend
- Multi-timeframe alignment

---

## 📐 Market Structure

The analyzer provides three major market-structure components:

### Trend

Determined using:

- Price vs EMA20
- Price vs EMA50
- Price vs EMA200
- EMA alignment
- EMA slopes

Possible states:

- `BULLISH`
- `BEARISH`
- `SIDEWAYS`

### Momentum

Uses:

- RSI
- MACD histogram
- ROC10
- ROC20

Possible states include:

- `BULLISH`
- `BEARISH`
- `NEUTRAL`

### Volume

Uses:

- Relative volume
- Volume change
- Volume ROC
- OBV slope
- Price-volume relationship

Possible states include:

- `ACCUMULATION`
- `DISTRIBUTION`
- `NEUTRAL`

---

## 🕐 Multi-Timeframe Analysis

The system evaluates multiple timeframes from daily market data.

### Daily

Used for the primary trading signal.

### Weekly

Used for broader trend confirmation.

### Monthly

Used for long-term directional context.

The higher-timeframe features are shifted to completed periods to reduce look-ahead bias.

---

## 🎯 Trading Signal

The model produces:

- Probability of upward movement
- Probability of downward movement
- Confidence
- BUY / SELL / WAIT signal

The signal is combined with multi-timeframe alignment rather than using machine-learning probability alone.

---

## 💰 Trade Plan

When a directional signal is generated, the application calculates:

- Entry
- Stop Loss
- Target

Risk parameters are handled internally by the application rather than exposed as user inputs.

---

## 🧪 Backtesting

The application includes a dedicated backtesting page.

The backtesting engine supports:

- Historical simulation
- Walk-forward model retraining
- Prediction horizon
- Entry logic
- Stop-loss exits
- Target exits
- Time-based exits
- Transaction costs
- Slippage
- Equity curve
- Maximum drawdown
- Win rate
- Profit factor
- Average trade

---

## 🏗️ Project Architecture

```text
Swing-Analyzer/
│
├── app.py
│
├── pages/
│   ├── 1_Stock_Analyzer.py
│   └── 2_Backtesting.py
│
├── config/
│   └── settings.yaml
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py
│   │   ├── market.py
│   │   └── sector.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── technical.py
│   │   ├── price_action.py
│   │   ├── volume.py
│   │   ├── regime.py
│   │   └── multi_timeframe.py
│   │
│   └── models/
│       ├── __init__.py
│       ├── classifier.py
│       ├── predictor.py
│       └── backtest.py
│
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_backtest.py
│
├── requirements.txt
├── pytest.ini
├── README.md
└── .gitignore
