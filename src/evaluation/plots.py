"""
Evaluation visualizations for the AI Swing Analyser.

This module contains reusable Plotly chart builders for research,
validation, backtesting, calibration, and model diagnostics.

Important
---------
Charts are descriptive only.

They must never modify model predictions, evaluation results, or
production approval status.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _validate_dataframe(
    data: pd.DataFrame,
    name: str = "data",
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return data.copy()


def _validate_series(
    series: pd.Series,
    name: str = "series",
) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"{name} must be a pandas Series."
        )

    if series.empty:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return series.copy()


def _apply_layout(
    figure: go.Figure,
    *,
    title: str,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
) -> go.Figure:
    figure.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        margin={
            "l": 60,
            "r": 30,
            "t": 70,
            "b": 50,
        },
    )

    if x_title:
        figure.update_xaxes(
            title=x_title
        )

    if y_title:
        figure.update_yaxes(
            title=y_title
        )

    return figure


# ----------------------------------------------------------------------
# Equity curve
# ----------------------------------------------------------------------


def plot_equity_curve(
    data: pd.DataFrame | pd.Series,
    *,
    timestamp_column: Optional[str] = None,
    equity_column: str = "equity",
    title: str = "Strategy Equity Curve",
) -> go.Figure:
    """
    Plot portfolio equity through time.
    """

    if isinstance(
        data,
        pd.Series,
    ):
        series = data.dropna()

        x = series.index
        y = series.values

    else:
        frame = _validate_dataframe(
            data
        )

        if equity_column not in frame.columns:
            raise ValueError(
                f"Missing equity column: {equity_column}"
            )

        y = pd.to_numeric(
            frame[equity_column],
            errors="coerce",
        )

        if timestamp_column is not None:
            if timestamp_column not in frame.columns:
                raise ValueError(
                    f"Missing timestamp column: "
                    f"{timestamp_column}"
                )

            x = frame[
                timestamp_column
            ]

        else:
            x = frame.index

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Equity",
        )
    )

    return _apply_layout(
        figure,
        title=title,
        x_title="Date",
        y_title="Equity",
    )


# ----------------------------------------------------------------------
# Drawdown
# ----------------------------------------------------------------------


def calculate_drawdown(
    equity: pd.Series,
) -> pd.Series:
    """
    Calculate percentage drawdown from the running equity peak.
    """

    series = _validate_series(
        equity,
        "equity",
    ).astype(float)

    running_peak = (
        series.cummax()
    )

    drawdown = (
        series / running_peak
        - 1.0
    )

    return drawdown.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def plot_drawdown(
    equity: pd.Series,
    *,
    title: str = "Strategy Drawdown",
) -> go.Figure:
    """
    Plot percentage portfolio drawdown.
    """

    drawdown = calculate_drawdown(
        equity
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown * 100.0,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
        )
    )

    return _apply_layout(
        figure,
        title=title,
        x_title="Date",
        y_title="Drawdown (%)",
    )


# ----------------------------------------------------------------------
# Walk-forward performance
# ----------------------------------------------------------------------


def plot_walk_forward_accuracy(
    data: pd.DataFrame,
    *,
    fold_column: str = "fold",
    accuracy_column: str = "accuracy",
    title: str = "Walk-Forward Accuracy",
) -> go.Figure:
    """
    Plot accuracy for each walk-forward fold.
    """

    frame = _validate_dataframe(
        data
    )

    missing = [
        column
        for column in (
            fold_column,
            accuracy_column,
        )
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=frame[fold_column],
            y=frame[accuracy_column] * 100.0,
            name="Accuracy",
        )
    )

    figure.add_hline(
        y=95.0,
        line_dash="dash",
        annotation_text="95% research gate",
    )

    return _apply_layout(
        figure,
        title=title,
        x_title="Walk-Forward Fold",
        y_title="Accuracy (%)",
    )


# ----------------------------------------------------------------------
# Calibration / reliability
# ----------------------------------------------------------------------


def plot_reliability_curve(
    data: pd.DataFrame,
    *,
    confidence_column: str = "confidence",
    actual_column: str = "actual_rate",
    count_column: Optional[str] = None,
    title: str = "Probability Reliability Curve",
) -> go.Figure:
    """
    Plot predicted confidence against observed event frequency.
    """

    frame = _validate_dataframe(
        data
    )

    required = [
        confidence_column,
        actual_column,
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing calibration columns: {missing}"
        )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=frame[confidence_column],
            y=frame[actual_column],
            mode="lines+markers",
            name="Observed",
            marker=(
                dict(
                    size=8
                )
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Perfect calibration",
            line=(
                dict(
                    dash="dash"
                )
            ),
        )
    )

    if count_column is not None:
        if count_column not in frame.columns:
            raise ValueError(
                f"Missing count column: {count_column}"
            )

        figure.update_traces(
            selector={
                "name": "Observed"
            },
            customdata=np.asarray(
                frame[count_column]
            ),
            hovertemplate=(
                "Confidence: %{x:.2f}"
                "<br>Observed: %{y:.2f}"
                "<br>Samples: %{customdata}"
                "<extra></extra>"
            ),
        )

    return _apply_layout(
        figure,
        title=title,
        x_title="Predicted Probability",
        y_title="Observed Frequency",
    )


# ----------------------------------------------------------------------
# Prediction vs actual
# ----------------------------------------------------------------------


def plot_prediction_vs_actual(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    title: str = "Predicted vs Actual Return",
    x_title: str = "Actual",
    y_title: str = "Predicted",
) -> go.Figure:
    """
    Scatter plot of predicted versus actual values.
    """

    actual_series = _validate_series(
        actual,
        "actual",
    )

    predicted_series = _validate_series(
        predicted,
        "predicted",
    )

    aligned = pd.concat(
        [
            actual_series.rename(
                "actual"
            ),
            predicted_series.rename(
                "predicted"
            ),
        ],
        axis=1,
    ).dropna()

    if aligned.empty:
        raise ValueError(
            "No overlapping non-null observations."
        )

    minimum = min(
        aligned["actual"].min(),
        aligned["predicted"].min(),
    )

    maximum = max(
        aligned["actual"].max(),
        aligned["predicted"].max(),
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=aligned["actual"],
            y=aligned["predicted"],
            mode="markers",
            name="Predictions",
            marker=dict(
                size=7
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="Perfect prediction",
            line=dict(
                dash="dash"
            ),
        )
    )

    return _apply_layout(
        figure,
        title=title,
        x_title=x_title,
        y_title=y_title,
    )


# ----------------------------------------------------------------------
# Target-price range
# ----------------------------------------------------------------------


def plot_price_range(
    data: pd.DataFrame,
    *,
    timestamp_column: Optional[str] = None,
    lower_column: str = "lower_price",
    median_column: str = "median_price",
    upper_column: str = "upper_price",
    actual_column: Optional[str] = None,
    title: str = "Predicted Target-Price Range",
) -> go.Figure:
    """
    Plot lower/median/upper predicted price bounds.

    The upper and lower traces form the predicted interval.
    """

    frame = _validate_dataframe(
        data
    )

    required = [
        lower_column,
        median_column,
        upper_column,
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing price-range columns: {missing}"
        )

    x = (
        frame[timestamp_column]
        if timestamp_column is not None
        else frame.index
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame[upper_column],
            mode="lines",
            name="Upper Range",
            line=dict(
                width=1
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame[lower_column],
            mode="lines",
            name="Lower Range",
            fill="tonexty",
            line=dict(
                width=1
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame[median_column],
            mode="lines",
            name="Median Target",
            line=dict(
                width=2
            ),
        )
    )

    if actual_column is not None:
        if actual_column not in frame.columns:
            raise ValueError(
                f"Missing actual-price column: "
                f"{actual_column}"
            )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=frame[actual_column],
                mode="markers",
                name="Actual",
                marker=dict(
                    size=7
                ),
            )
        )

    return _apply_layout(
        figure,
        title=title,
        x_title="Date",
        y_title="Price",
    )


# ----------------------------------------------------------------------
# Feature importance
# ----------------------------------------------------------------------


def plot_feature_importance(
    data: pd.DataFrame,
    *,
    feature_column: str = "feature",
    importance_column: str = "importance",
    top_n: int = 25,
    title: str = "Feature Importance",
) -> go.Figure:
    """
    Plot the most important model features.

    Feature importance should come from a training/development process
    and should not be interpreted as proof of causal importance.
    """

    frame = _validate_dataframe(
        data
    )

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    missing = [
        column
        for column in (
            feature_column,
            importance_column,
        )
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature-importance columns: {missing}"
        )

    frame = frame.copy()

    frame[
        importance_column
    ] = pd.to_numeric(
        frame[importance_column],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            importance_column
        ]
    )

    frame = frame.sort_values(
        importance_column,
        ascending=False,
    ).head(
        top_n
    )

    frame = frame.sort_values(
        importance_column,
        ascending=True,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=frame[
                importance_column
            ],
            y=frame[
                feature_column
            ],
            orientation="h",
            name="Importance",
        )
    )

    return _apply_layout(
        figure,
        title=title,
        x_title="Importance",
        y_title="Feature",
    )


# ----------------------------------------------------------------------
# Regime performance
# ----------------------------------------------------------------------


def plot_regime_performance(
    data: pd.DataFrame,
    *,
    regime_column: str = "regime",
    metric_column: str = "accuracy",
    title: str = "Performance by Market Regime",
) -> go.Figure:
    """
    Plot a performance metric by market regime.
    """

    frame = _validate_dataframe(
        data
    )

    missing = [
        column
        for column in (
            regime_column,
            metric_column,
        )
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing regime columns: {missing}"
        )

    frame = frame.copy()

    frame[
        metric_column
    ] = pd.to_numeric(
        frame[metric_column],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            metric_column
        ]
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=frame[
                regime_column
            ],
            y=frame[
                metric_column
            ] * 100.0,
            name=metric_column,
        )
    )

    return _apply_layout(
        figure,
        title=title,
        x_title="Market Regime",
        y_title=f"{metric_column} (%)",
    )


# ----------------------------------------------------------------------
# Return distribution
# ----------------------------------------------------------------------


def plot_return_distribution(
    returns: pd.Series,
    *,
    bins: int = 50,
    title: str = "Return Distribution",
) -> go.Figure:
    """
    Plot a histogram of strategy/trade returns.
    """

    series = _validate_series(
        returns,
        "returns",
    )

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        raise ValueError(
            "No valid return observations."
        )

    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            name="Returns",
        )
    )

    figure.add_vline(
        x=0.0,
        line_dash="dash",
        annotation_text="Break-even",
    )

    return _apply_layout(
        figure,
        title=title,
        x_title="Return",
        y_title="Frequency",
    )


# ----------------------------------------------------------------------
# Confidence distribution
# ----------------------------------------------------------------------


def plot_confidence_distribution(
    confidence: pd.Series,
    *,
    bins: int = 30,
    title: str = "Prediction Confidence Distribution",
) -> go.Figure:
    """
    Plot the distribution of model probabilities/confidence values.
    """

    series = _validate_series(
        confidence,
        "confidence",
    )

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    values = values[
        (values >= 0.0)
        & (values <= 1.0)
    ]

    if values.empty:
        raise ValueError(
            "No valid confidence values between 0 and 1."
        )

    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            name="Confidence",
        )
    )

    figure.add_vline(
        x=0.60,
        line_dash="dash",
        annotation_text="Decision threshold",
    )

    figure.add_vline(
        x=0.70,
        line_dash="dot",
        annotation_text="High confidence",
    )

    return _apply_layout(
        figure,
        title=title,
        x_title="Probability / Confidence",
        y_title="Frequency",
    )


# ----------------------------------------------------------------------
# Price chart with signals
# ----------------------------------------------------------------------


def plot_price_with_signals(
    data: pd.DataFrame,
    *,
    open_column: str = "Open",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    signal_column: Optional[str] = None,
    title: str = "Price and Model Signals",
) -> go.Figure:
    """
    Plot OHLC candles with optional model signals.

    Expected signal values:
        BUY / LONG
        SELL / SHORT
        WAIT
    """

    frame = _validate_dataframe(
        data
    )

    required = [
        open_column,
        high_column,
        low_column,
        close_column,
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing OHLC columns: {missing}"
        )

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame[
                open_column
            ],
            high=frame[
                high_column
            ],
            low=frame[
                low_column
            ],
            close=frame[
                close_column
            ],
            name="Price",
        )
    )

    if signal_column is not None:
        if signal_column not in frame.columns:
            raise ValueError(
                f"Missing signal column: {signal_column}"
            )

        signals = frame[
            signal_column
        ].astype(str).str.upper()

        buy_mask = signals.isin(
            [
                "BUY",
                "LONG",
            ]
        )

        sell_mask = signals.isin(
            [
                "SELL",
                "SHORT",
            ]
        )

        if buy_mask.any():
            figure.add_trace(
                go.Scatter(
                    x=frame.index[
                        buy_mask
                    ],
                    y=frame[
                        low_column
                    ][
                        buy_mask
                    ],
                    mode="markers",
                    name="BUY",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                    ),
                )
            )

        if sell_mask.any():
            figure.add_trace(
                go.Scatter(
                    x=frame.index[
                        sell_mask
                    ],
                    y=frame[
                        high_column
                    ][
                        sell_mask
                    ],
                    mode="markers",
                    name="SELL",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                    ),
                )
            )

    figure.update_xaxes(
        rangeslider_visible=False
    )

    return _apply_layout(
        figure,
        title=title,
        x_title="Date",
        y_title="Price",
    )


# ----------------------------------------------------------------------
# Walk-forward metric stability
# ----------------------------------------------------------------------


def plot_metric_stability(
    data: pd.DataFrame,
    *,
    fold_column: str = "fold",
    metric_columns: Optional[
        Iterable[str]
    ] = None,
    title: str = "Walk-Forward Metric Stability",
) -> go.Figure:
    """
    Plot one or more metrics across walk-forward folds.
    """

    frame = _validate_dataframe(
        data
    )

    if fold_column not in frame.columns:
        raise ValueError(
            f"Missing fold column: {fold_column}"
        )

    if metric_columns is None:
        metric_columns = [
            column
            for column in frame.columns
            if column != fold_column
            and pd.api.types.is_numeric_dtype(
                frame[column]
            )
        ]

    metric_columns = list(
        metric_columns
    )

    if not metric_columns:
        raise ValueError(
            "No numeric metric columns supplied."
        )

    missing = [
        column
        for column in metric_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing metric columns: {missing}"
        )

    figure = go.Figure()

    for column in metric_columns:
        figure.add_trace(
            go.Scatter(
                x=frame[
                    fold_column
                ],
                y=frame[
                    column
                ],
                mode="lines+markers",
                name=column,
            )
        )

    return _apply_layout(
        figure,
        title=title,
        x_title="Fold",
        y_title="Metric",
    )


# ----------------------------------------------------------------------
# Range width
# ----------------------------------------------------------------------


def plot_range_width(
    data: pd.DataFrame,
    *,
    lower_column: str = "lower_price",
    upper_column: str = "upper_price",
    timestamp_column: Optional[str] = None,
    title: str = "Predicted Price-Range Width",
) -> go.Figure:
    """
    Plot target-price interval width.
    """

    frame = _validate_dataframe(
        data
    )

    if (
        lower_column not in frame.columns
        or upper_column not in frame.columns
    ):
        raise ValueError(
            "Price range columns are missing."
        )

    width = (
        pd.to_numeric(
            frame[upper_column],
            errors="coerce",
        )
        - pd.to_numeric(
            frame[lower_column],
            errors="coerce",
        )
    )

    x = (
        frame[timestamp_column]
        if timestamp_column is not None
        else frame.index
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x,
            y=width,
            mode="lines",
            name="Range Width",
        )
    )

    return _apply_layout(
        figure,
        title=title,
        x_title="Date",
        y_title="Price Range Width",
    )


# ----------------------------------------------------------------------
# Multi-timeframe alignment
# ----------------------------------------------------------------------


def plot_mtf_alignment(
    data: pd.DataFrame,
    *,
    timestamp_column: Optional[str] = None,
    alignment_columns: Optional[
        Iterable[str]
    ] = None,
    title: str = "Multi-Timeframe Alignment",
) -> go.Figure:
    """
    Plot multi-timeframe alignment scores/features.
    """

    frame = _validate_dataframe(
        data
    )

    if alignment_columns is None:
        alignment_columns = [
            column
            for column in frame.columns
            if "alignment" in column.lower()
            or "trend" in column.lower()
        ]

    alignment_columns = list(
        alignment_columns
    )

    if not alignment_columns:
        raise ValueError(
            "No alignment columns were found."
        )

    missing = [
        column
        for column in alignment_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing alignment columns: {missing}"
        )

    x = (
        frame[timestamp_column]
        if timestamp_column is not None
        else frame.index
    )

    figure = go.Figure()

    for column in alignment_columns:
        figure.add_trace(
            go.Scatter(
                x=x,
                y=frame[column],
                mode="lines",
                name=column,
            )
        )

    return _apply_layout(
        figure,
        title=title,
        x_title="Date",
        y_title="Alignment / Trend Score",
    )


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------


__all__ = [
    "calculate_drawdown",
    "plot_equity_curve",
    "plot_drawdown",
    "plot_walk_forward_accuracy",
    "plot_reliability_curve",
    "plot_prediction_vs_actual",
    "plot_price_range",
    "plot_feature_importance",
    "plot_regime_performance",
    "plot_return_distribution",
    "plot_confidence_distribution",
    "plot_price_with_signals",
    "plot_metric_stability",
    "plot_range_width",
    "plot_mtf_alignment",
]
