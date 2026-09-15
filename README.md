# AI Swing Analyser

An AI-powered, multi-timeframe stock analysis and swing-trading research platform.

## Objective

The objective of this project is to develop a statistically rigorous machine-learning system that analyses historical market data and estimates:

- Future stock direction
- Probability of upward/downward movement
- Expected price range
- Potential target zones
- Risk/reward
- Market regime
- Multi-timeframe trend alignment

The system will analyse:

- 4 Hour
- 1 Day
- 1 Week
- 1 Month

It will also consider broader Indian market conditions including:

- NIFTY 50
- SENSEX
- NIFTY Bank
- Sector indices
- India VIX

## Core Principle

Historical performance must be tested before the system is allowed to produce production/live trading signals.

The model must never be evaluated on data that was used to train or tune it.

## Anti-Overfitting Strategy

The project will use:

- Chronological train/validation/test splits
- Walk-forward validation
- Purged time-series cross-validation where required
- Embargo periods for overlapping prediction horizons
- Feature selection
- Regularisation
- Hyperparameter constraints
- Ensemble models
- Out-of-sample testing
- Final untouched holdout dataset
- Market-regime stability testing
- Feature stability testing
- Probability calibration
- Transaction-cost-aware backtesting

## Indicators

The feature engine will include, where statistically useful:

- RSI
- MACD
- EMA
- SMA
- Bollinger Bands
- ATR
- ADX
- Stochastic Oscillator
- ROC
- OBV
- VWAP
- Volume features
- Momentum
- Volatility
- Support/resistance
- Price structure

Indicators will not automatically be assumed to be predictive. Their usefulness will be measured empirically.

## Prediction Horizons

Initial horizons:

- 1 trading day
- 3 trading days
- 5 trading days
- 10 trading days
- 20 trading days

Later versions may include:

- 4H
- 1W
- 1M

## Prediction Architecture

The system will eventually contain:

```text
Historical Market Data
        |
        v
Data Quality Engine
        |
        v
Feature Engineering
        |
        +-------------------+
        |                   |
        v                   v
Stock Features       Market Features
        |                   |
        +---------+---------+
                  |
                  v
          Market Regime
                  |
                  v
        Multi-Timeframe Model
                  |
          +-------+-------+
          |               |
          v               v
 Direction Model    Price Range Model
          |               |
          +-------+-------+
                  |
                  v
             Ensemble
                  |
                  v
          Probability Engine
                  |
                  v
           Risk Engine
                  |
                  v
          Swing Setup Score
