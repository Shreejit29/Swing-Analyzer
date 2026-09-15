"""
Tests for the data layer.

These tests deliberately avoid live network calls.

External data providers are tested through mocked/fake provider
implementations so that the test suite remains deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cache import (
    CacheConfig,
    DataCache,
    build_cache_key,
)
from src.data.downloader import (
    normalize_downloaded_data,
)
from src.data.providers import (
    DataRequest,
    MarketDataProvider,
    ProviderRegistry,
)
from src.data.quality import (
    assert_quality,
    clean_ohlcv,
    validate_ohlcv,
)
from src.data.resample import (
    resample_4h,
    resample_daily,
)
from src.data.session import (
    DEFAULT_SESSION,
    add_4h_session_bucket,
    assign_4h_session_bucket,
    filter_regular_session,
    is_in_regular_session,
    is_trading_day,
    session_progress,
    validate_session_index,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def daily_ohlcv() -> pd.DataFrame:
    index = pd.date_range(
        "2025-01-01",
        periods=10,
        freq="D",
    )

    close = np.linspace(
        100.0,
        110.0,
        len(index),
    )

    return pd.DataFrame(
        {
            "Open": close - 1.0,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": np.arange(
                1000,
                1000 + len(index),
            ),
        },
        index=index,
    )


@pytest.fixture
def intraday_ohlcv() -> pd.DataFrame:
    """
    Synthetic hourly observations inside and outside NSE session.
    """

    index = pd.DatetimeIndex(
        [
            "2025-01-02 08:30",
            "2025-01-02 09:15",
            "2025-01-02 10:15",
            "2025-01-02 11:15",
            "2025-01-02 12:15",
            "2025-01-02 13:15",
            "2025-01-02 14:15",
            "2025-01-02 15:15",
            "2025-01-02 15:30",
            "2025-01-02 16:00",
        ]
    )

    close = np.arange(
        100.0,
        110.0,
    )

    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(
                len(index),
                1000,
            ),
        },
        index=index,
    )


# ----------------------------------------------------------------------
# OHLCV quality tests
# ----------------------------------------------------------------------


def test_validate_ohlcv_passes_valid_data(
    daily_ohlcv: pd.DataFrame,
) -> None:
    report = validate_ohlcv(
        daily_ohlcv
    )

    assert report.passed is True
    assert report.rows == 10
    assert report.errors == []


def test_quality_rejects_invalid_ohlc(
    daily_ohlcv: pd.DataFrame,
) -> None:
    broken = daily_ohlcv.copy()

    broken.loc[
        broken.index[0],
        "High",
    ] = 50.0

    report = validate_ohlcv(
        broken
    )

    assert report.passed is False
    assert len(
        report.errors
    ) > 0


def test_clean_ohlcv_removes_duplicates(
    daily_ohlcv: pd.DataFrame,
) -> None:
    duplicate = pd.concat(
        [
            daily_ohlcv,
            daily_ohlcv.iloc[
                [0]
            ],
        ]
    )

    cleaned = clean_ohlcv(
        duplicate
    )

    assert not cleaned.index.has_duplicates
    assert len(cleaned) == len(
        daily_ohlcv
    )


def test_assert_quality_raises_for_bad_data(
    daily_ohlcv: pd.DataFrame,
) -> None:
    broken = daily_ohlcv.copy()

    broken.loc[
        broken.index[0],
        "Low",
    ] = 1000.0

    with pytest.raises(
        ValueError
    ):
        assert_quality(
            broken
        )


# ----------------------------------------------------------------------
# Downloader normalization
# ----------------------------------------------------------------------


def test_normalize_downloaded_data(
    daily_ohlcv: pd.DataFrame,
) -> None:
    normalized = (
        normalize_downloaded_data(
            daily_ohlcv
        )
    )

    assert isinstance(
        normalized.index,
        pd.DatetimeIndex,
    )

    assert list(
        normalized.columns
    ) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    assert normalized.index.is_monotonic_increasing


# ----------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------


def test_resample_daily(
    intraday_ohlcv: pd.DataFrame,
) -> None:
    result = resample_daily(
        intraday_ohlcv
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }.issubset(
        result.columns
    )


def test_resample_4h_returns_ohlcv(
    intraday_ohlcv: pd.DataFrame,
) -> None:
    result = resample_4h(
        intraday_ohlcv
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }.issubset(
        result.columns
    )


# ----------------------------------------------------------------------
# NSE session tests
# ----------------------------------------------------------------------


def test_nse_trading_day_weekday() -> None:
    monday = pd.Timestamp(
        "2025-01-06"
    )

    assert is_trading_day(
        monday
    )


def test_nse_weekend_not_trading_day() -> None:
    saturday = pd.Timestamp(
        "2025-01-04"
    )

    assert not is_trading_day(
        saturday
    )


def test_session_boundaries() -> None:
    assert is_in_regular_session(
        "2025-01-06 09:15"
    )

    assert is_in_regular_session(
        "2025-01-06 15:30"
    )

    assert not is_in_regular_session(
        "2025-01-06 09:14"
    )

    assert not is_in_regular_session(
        "2025-01-06 15:31"
    )


def test_filter_regular_session(
    intraday_ohlcv: pd.DataFrame,
) -> None:
    filtered = filter_regular_session(
        intraday_ohlcv
    )

    assert (
        filtered.index.min()
        == pd.Timestamp(
            "2025-01-02 09:15"
        )
    )

    assert (
        filtered.index.max()
        == pd.Timestamp(
            "2025-01-02 15:30"
        )
    )

    assert all(
        is_in_regular_session(
            timestamp
        )
        for timestamp
        in filtered.index
    )


def test_validate_session_index(
    intraday_ohlcv: pd.DataFrame,
) -> None:
    result = validate_session_index(
        intraday_ohlcv
    )

    assert result[
        "rows"
    ] == 10

    assert result[
        "outside_regular_session"
    ] > 0

    assert result[
        "weekend_rows"
    ] == 0


# ----------------------------------------------------------------------
# Session progress
# ----------------------------------------------------------------------


def test_session_progress_open() -> None:
    progress = session_progress(
        "2025-01-06 09:15"
    )

    assert progress == pytest.approx(
        0.0
    )


def test_session_progress_close() -> None:
    progress = session_progress(
        "2025-01-06 15:30"
    )

    assert progress == pytest.approx(
        1.0
    )


# ----------------------------------------------------------------------
# 4H session buckets
# ----------------------------------------------------------------------


def test_assign_4h_bucket_first_session() -> None:
    bucket = assign_4h_session_bucket(
        "2025-01-06 10:00"
    )

    assert bucket == pd.Timestamp(
        "2025-01-06 09:15"
    )


def test_assign_4h_bucket_second_session() -> None:
    bucket = assign_4h_session_bucket(
        "2025-01-06 14:00"
    )

    assert bucket == pd.Timestamp(
        "2025-01-06 13:15"
    )


def test_4h_bucket_rejects_outside_session() -> None:
    with pytest.raises(
        ValueError
    ):
        assign_4h_session_bucket(
            "2025-01-06 08:00"
        )


def test_add_4h_session_bucket(
    intraday_ohlcv: pd.DataFrame,
) -> None:
    result = add_4h_session_bucket(
        intraday_ohlcv
    )

    assert "_4h_bucket" in (
        result.columns
    )

    assert result[
        "_4h_bucket"
    ].notna().all()


# ----------------------------------------------------------------------
# Cache tests
# ----------------------------------------------------------------------


def test_cache_key_is_deterministic() -> None:
    first = build_cache_key(
        symbol="RELIANCE.NS",
        timeframe="1D",
        interval="1d",
        start="2020-01-01",
        end="2025-01-01",
        provider="test",
    )

    second = build_cache_key(
        symbol="RELIANCE.NS",
        timeframe="1D",
        interval="1d",
        start="2020-01-01",
        end="2025-01-01",
        provider="test",
    )

    assert first == second


def test_cache_key_changes_when_request_changes() -> None:
    first = build_cache_key(
        symbol="RELIANCE.NS",
        timeframe="1D",
        interval="1d",
    )

    second = build_cache_key(
        symbol="TCS.NS",
        timeframe="1D",
        interval="1d",
    )

    assert first != second


def test_data_cache_round_trip(
    daily_ohlcv: pd.DataFrame,
    tmp_path: Path,
) -> None:
    cache = DataCache(
        CacheConfig(
            directory=str(
                tmp_path
            )
        )
    )

    key = build_cache_key(
        symbol="TEST.NS",
        timeframe="1D",
        interval="1d",
    )

    metadata = cache.save(
        daily_ohlcv,
        key=key,
        symbol="TEST.NS",
        timeframe="1D",
        interval="1d",
    )

    assert metadata.rows == len(
        daily_ohlcv
    )

    loaded = cache.load(
        key
    )

    assert loaded is not None

    pd.testing.assert_frame_equal(
        loaded,
        daily_ohlcv,
    )


def test_data_cache_detects_corruption(
    daily_ohlcv: pd.DataFrame,
    tmp_path: Path,
) -> None:
    cache = DataCache(
        CacheConfig(
            directory=str(
                tmp_path
            )
        )
    )

    key = build_cache_key(
        symbol="TEST.NS",
        timeframe="1D",
        interval="1d",
    )

    cache.save(
        daily_ohlcv,
        key=key,
        symbol="TEST.NS",
        timeframe="1D",
        interval="1d",
    )

    path = cache.data_path(
        key
    )

    corrupted = daily_ohlcv.copy()

    corrupted.loc[
        corrupted.index[0],
        "Close",
    ] = 999999.0

    corrupted.to_parquet(
        path
    )

    with pytest.raises(
        RuntimeError,
        match="checksum mismatch",
    ):
        cache.load(
            key
        )


# ----------------------------------------------------------------------
# Provider tests
# ----------------------------------------------------------------------


class FakeProvider(
    MarketDataProvider
):
    name = "fake"

    def fetch(
        self,
        request: DataRequest,
    ) -> pd.DataFrame:
        index = pd.date_range(
            "2025-01-01",
            periods=3,
            freq="D",
        )

        return pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000, 1100, 1200],
            },
            index=index,
        )


def test_provider_registry() -> None:
    registry = ProviderRegistry(
        providers=[
            FakeProvider()
        ]
    )

    request = DataRequest(
        symbol="TEST.NS",
        timeframe="1D",
    )

    result = registry.fetch(
        request,
        provider="fake",
    )

    assert len(
        result
    ) == 3


def test_provider_registry_rejects_duplicate() -> None:
    registry = ProviderRegistry(
        providers=[
            FakeProvider()
        ]
    )

    with pytest.raises(
        ValueError
    ):
        registry.register(
            FakeProvider()
        )


def test_data_request_rejects_invalid_timeframe() -> None:
    with pytest.raises(
        ValueError
    ):
        DataRequest(
            symbol="TEST.NS",
            timeframe="5M",
        )


# ----------------------------------------------------------------------
# Cache get-or-fetch
# ----------------------------------------------------------------------


def test_cache_get_or_fetch_uses_cache(
    daily_ohlcv: pd.DataFrame,
    tmp_path: Path,
) -> None:
    cache = DataCache(
        CacheConfig(
            directory=str(
                tmp_path
            )
        )
    )

    key = build_cache_key(
        symbol="TEST.NS",
        timeframe="1D",
        interval="1d",
    )

    calls = {
        "count": 0
    }

    def fetcher() -> pd.DataFrame:
        calls["count"] += 1
        return daily_ohlcv

    first = cache.get_or_fetch(
        key=key,
        fetcher=fetcher,
        symbol="TEST.NS",
        timeframe="1D",
        interval="1d",
    )

    second = cache.get_or_fetch(
        key=key,
        fetcher=fetcher,
        symbol="TEST.NS",
        timeframe="1D",
        interval="1d",
    )

    assert calls[
        "count"
    ] == 1

    pd.testing.assert_frame_equal(
        first,
        second,
    )
