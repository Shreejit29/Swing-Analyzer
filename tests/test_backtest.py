from tests.test_features import sample
from src.models.backtest import BacktestConfig, run_backtest


def test_backtest_runs():
    result = run_backtest(sample(320), BacktestConfig(min_train_rows=150, retrain_every=30))
    assert "equity_curve" in result
    assert "trades_df" in result
