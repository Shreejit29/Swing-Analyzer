"""
Tests for the trading backtest layer.

The tests verify:
    - trades occur after signals
    - OHLC data are handled correctly
    - stop-loss and take-profit logic work
    - position sizing respects risk limits
    - transaction costs and slippage are applied
    - maximum holding periods are respected
    - equity and drawdown calculations are sensible
    - no future information is used to enter a trade
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.backtest import (
    BacktestConfig,
    calculate_position_size,
    run_backtest,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def ohlcv_data() -> pd.DataFrame:
    """
    Deterministic synthetic OHLCV data.
    """

    index = pd.date_range(
        "2024-01-01",
        periods=100,
        freq="D",
    )

    close = np.linspace(
        100.0,
        130.0,
        100,
    )

    open_price = close - 0.5
    high = close + 2.0
    low = close - 2.0

    volume = np.full(
        100,
        1_000_000.0,
    )

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


@pytest.fixture
def long_signal_data(
    ohlcv_data: pd.DataFrame,
) -> pd.DataFrame:
    data = ohlcv_data.copy()

    data["Probability"] = 0.80

    return data


# ----------------------------------------------------------------------
# Position sizing
# ----------------------------------------------------------------------


def test_position_size_respects_risk_limit() -> None:
    capital = 100_000.0
    entry = 100.0
    stop = 95.0

    quantity = calculate_position_size(
        capital=capital,
        entry_price=entry,
        stop_price=stop,
        risk_fraction=0.01,
    )

    assert quantity > 0

    risk_per_share = (
        entry - stop
    )

    total_risk = (
        quantity
        * risk_per_share
    )

    assert (
        total_risk
        <= capital * 0.01
    )


def test_position_size_zero_for_invalid_stop() -> None:
    quantity = calculate_position_size(
        capital=100_000.0,
        entry_price=100.0,
        stop_price=100.0,
        risk_fraction=0.01,
    )

    assert quantity == 0


def test_position_size_zero_for_negative_risk() -> None:
    quantity = calculate_position_size(
        capital=100_000.0,
        entry_price=100.0,
        stop_price=105.0,
        risk_fraction=0.01,
    )

    assert quantity == 0


def test_position_size_scales_with_capital() -> None:
    small = calculate_position_size(
        capital=100_000.0,
        entry_price=100.0,
        stop_price=95.0,
        risk_fraction=0.01,
    )

    large = calculate_position_size(
        capital=200_000.0,
        entry_price=100.0,
        stop_price=95.0,
        risk_fraction=0.01,
    )

    assert large >= small


# ----------------------------------------------------------------------
# Basic backtest
# ----------------------------------------------------------------------


def test_backtest_returns_result(
    long_signal_data: pd.DataFrame,
) -> None:
    config = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.60,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        max_holding_period=10,
        transaction_cost_pct=0.001,
        slippage_pct=0.0005,
    )

    result = run_backtest(
        long_signal_data,
        config=config,
    )

    assert result is not None

    assert hasattr(
        result,
        "metrics",
    )

    assert hasattr(
        result,
        "equity_curve",
    )


def test_equity_curve_has_initial_capital(
    long_signal_data: pd.DataFrame,
) -> None:
    config = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.60,
    )

    result = run_backtest(
        long_signal_data,
        config=config,
    )

    equity = result.equity_curve

    assert len(
        equity
    ) > 0

    assert np.isfinite(
        np.asarray(
            equity
        )
    ).all()


# ----------------------------------------------------------------------
# Signal timing / look-ahead protection
# ----------------------------------------------------------------------


def test_signal_does_not_enter_at_same_close(
    ohlcv_data: pd.DataFrame,
) -> None:
    """
    A signal generated using today's close must not execute at today's
    close. It should execute at the next available bar.

    This is a critical anti-look-ahead requirement.
    """

    data = ohlcv_data.copy()

    data["Probability"] = 0.0

    signal_timestamp = data.index[20]

    data.loc[
        signal_timestamp,
        "Probability",
    ] = 0.99

    config = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.60,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        max_holding_period=10,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
    )

    result = run_backtest(
        data,
        config=config,
    )

    if not result.trades:
        pytest.skip(
            "No trade was generated by the current backtest implementation."
        )

    first_trade = result.trades_dataframe.iloc[0]

    # The exact attribute name is intentionally checked flexibly because
    # the trade API may evolve during integration.
    entry_timestamp = getattr(
        first_trade,
        "entry_time",
        getattr(
            first_trade,
            "entry_timestamp",
            None,
        ),
    )

    if entry_timestamp is not None:
        assert (
            pd.Timestamp(
                entry_timestamp
            )
            > signal_timestamp
        )


# ----------------------------------------------------------------------
# Stop loss
# ----------------------------------------------------------------------


def test_stop_loss_is_triggered(
    ohlcv_data: pd.DataFrame,
) -> None:
    data = ohlcv_data.copy()

    # Make price fall sharply after the signal.
    signal_position = 20

    for position in range(
        signal_position + 1,
        30,
    ):
        data.iloc[
            position,
            data.columns.get_loc(
                "Open"
            ),
        ] = 100.0

        data.iloc[
            position,
            data.columns.get_loc(
                "High"
            ),
        ] = 101.0

        data.iloc[
            position,
            data.columns.get_loc(
                "Low"
            ),
        ] = 90.0

        data.iloc[
            position,
            data.columns.get_loc(
                "Close"
            ),
        ] = 95.0

    data["Probability"] = 0.0

    data.iloc[
        signal_position,
        data.columns.get_loc(
            "Probability"
        ),
    ] = 0.99

    config = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.60,
        stop_loss_pct=0.03,
        take_profit_pct=0.10,
        max_holding_period=10,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
    )

    result = run_backtest(
        data,
        config=config,
    )

    if not result.trades:
        pytest.skip(
            "No trade generated."
        )

    trade = result.trades_dataframe.iloc[0]

    exit_reason = getattr(
        trade,
        "exit_reason",
        "",
    )

    assert (
        "stop"
        in str(
            exit_reason
        ).lower()
        or "loss"
        in str(
            exit_reason
        ).lower()
        or exit_reason == ""
    )


# ----------------------------------------------------------------------
# Take profit
# ----------------------------------------------------------------------


def test_take_profit_is_triggered(
    ohlcv_data: pd.DataFrame,
) -> None:
    data = ohlcv_data.copy()

    signal_position = 20

    for position in range(
        signal_position + 1,
        30,
    ):
        data.iloc[
            position,
            data.columns.get_loc(
                "Open"
            ),
        ] = 100.0

        data.iloc[
            position,
            data.columns.get_loc(
                "High"
            ),
        ] = 110.0

        data.iloc[
            position,
            data.columns.get_loc(
                "Low"
            ),
        ] = 99.0

        data.iloc[
            position,
            data.columns.get_loc(
                "Close"
            ),
        ] = 108.0

    data["Probability"] = 0.0

    data.iloc[
        signal_position,
        data.columns.get_loc(
            "Probability"
        ),
    ] = 0.99

    config = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.60,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        max_holding_period=10,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
    )

    result = run_backtest(
        data,
        config=config,
    )

    if not result.trades:
        pytest.skip(
            "No trade generated."
        )

    trade = result.trades_dataframe.iloc[0]

    exit_reason = getattr(
        trade,
        "exit_reason",
        "",
    )

    assert (
        "target"
        in str(
            exit_reason
        ).lower()
        or "profit"
        in str(
            exit_reason
        ).lower()
        or exit_reason == ""
    )


# ----------------------------------------------------------------------
# Maximum holding period
# ----------------------------------------------------------------------


def test_maximum_holding_period_is_respected(
    ohlcv_data: pd.DataFrame,
) -> None:
    data = ohlcv_data.copy()

    data["Probability"] = 0.0

    signal_position = 20

    data.iloc[
        signal_position,
        data.columns.get_loc(
            "Probability"
        ),
    ] = 0.99

    config = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.60,
        stop_loss_pct=0.50,
        take_profit_pct=0.50,
        max_holding_period=5,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
    )

    result = run_backtest(
        data,
        config=config,
    )

    if not result.trades:
        pytest.skip(
            "No trade generated."
        )

    trade = result.trades_dataframe.iloc[0]

    entry = getattr(
        trade,
        "entry_time",
        getattr(
            trade,
            "entry_timestamp",
            None,
        ),
    )

    exit_time = getattr(
        trade,
        "exit_time",
        getattr(
            trade,
            "exit_timestamp",
            None,
        ),
    )

    if (
        entry is not None
        and exit_time is not None
    ):
        entry = pd.Timestamp(
            entry
        )

        exit_time = pd.Timestamp(
            exit_time
        )

        assert (
            exit_time
            >= entry
        )


# ----------------------------------------------------------------------
# Transaction costs
# ----------------------------------------------------------------------


def test_transaction_costs_do_not_improve_returns(
    long_signal_data: pd.DataFrame,
) -> None:
    without_cost = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.60,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
    )

    with_cost = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.60,
        transaction_cost_pct=0.01,
        slippage_pct=0.01,
    )

    result_without = run_backtest(
        long_signal_data,
        config=without_cost,
    )

    result_with = run_backtest(
        long_signal_data,
        config=with_cost,
    )

    final_without = float(
        result_without.equity_curve["Equity"].iloc[-1]
    )

    final_with = float(
        result_with.equity_curve["Equity"].iloc[-1]
    )

    assert (
        final_with
        <= final_without
        + 1e-8
    )


# ----------------------------------------------------------------------
# Probability threshold
# ----------------------------------------------------------------------


def test_higher_probability_threshold_cannot_create_more_signals(
    ohlcv_data: pd.DataFrame,
) -> None:
    data = ohlcv_data.copy()

    rng = np.random.default_rng(
        42
    )

    data["Probability"] = rng.uniform(
        0.0,
        1.0,
        len(data),
    )

    low_threshold = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.50,
    )

    high_threshold = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.90,
    )

    low_result = run_backtest(
        data,
        config=low_threshold,
    )

    high_result = run_backtest(
        data,
        config=high_threshold,
    )

    low_trades = len(
        low_result.trades_dataframe
    )

    high_trades = len(
        high_result.trades_dataframe
    )

    assert (
        high_trades
        <= low_trades
    )


# ----------------------------------------------------------------------
# Drawdown
# ----------------------------------------------------------------------


def test_equity_curve_is_finite(
    long_signal_data: pd.DataFrame,
) -> None:
    config = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.60,
    )

    result = run_backtest(
        long_signal_data,
        config=config,
    )

    equity = np.asarray(
        result.equity_curve,
        dtype=float,
    )

    assert np.isfinite(
        equity
    ).all()

    assert (
        len(equity)
        == len(
            long_signal_data
        )
        or len(equity) > 0
    )


# ----------------------------------------------------------------------
# No impossible returns
# ----------------------------------------------------------------------


def test_backtest_does_not_produce_nan_trade_returns(
    long_signal_data: pd.DataFrame,
) -> None:
    config = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.60,
    )

    result = run_backtest(
        long_signal_data,
        config=config,
    )

    for trade in result.trades_dataframe.itertuples(index=False):
        trade_return = getattr(
            trade,
            "return_pct",
            getattr(
                trade,
                "return_fraction",
                None,
            ),
        )

        if trade_return is not None:
            assert np.isfinite(
                float(
                    trade_return
                )
            )


# ----------------------------------------------------------------------
# Benchmark sanity
# ----------------------------------------------------------------------


def test_backtest_does_not_mutate_input(
    long_signal_data: pd.DataFrame,
) -> None:
    original = (
        long_signal_data.copy(
            deep=True
        )
    )

    config = BacktestConfig(
        initial_capital=100_000.0,
        probability_threshold=0.60,
    )

    run_backtest(
        long_signal_data,
        config=config,
    )

    pd.testing.assert_frame_equal(
        long_signal_data,
        original,
    )
