import pandas as pd
import numpy as np
from src.features.engine import build_features


def sample(n=260):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(100, 150, n) + np.sin(np.arange(n)), index=idx)
    return pd.DataFrame({"open": close * .995, "high": close * 1.01, "low": close * .99, "close": close, "volume": 100000}, index=idx)


def test_features_exist():
    out = build_features(sample())
    for col in ["ema20", "ema50", "ema200", "rsi14", "atr14", "adx14", "macd", "bb_upper", "relative_volume"]:
        assert col in out.columns
