"""
Market-data provider abstraction.

The research pipeline should not depend directly on one data vendor.

This module defines a common interface for:
    - daily data
    - weekly data
    - monthly data
    - intraday data

The initial implementation uses the existing Yahoo Finance downloader.

Future providers can implement the same interface without requiring
changes to the feature-engineering or modelling layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import (
    DEFAULT_DATA_SOURCE,
)
from .downloader import (
    YahooFinanceDownloader,
)


# ----------------------------------------------------------------------
# Provider request
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DataRequest:
    """
    Immutable description of a market-data request.
    """

    symbol: str

    timeframe: str

    start: Optional[str] = None

    end: Optional[str] = None

    interval: Optional[str] = None

    period: Optional[str] = None

    auto_adjust: bool = False

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError(
                "symbol cannot be empty."
            )

        valid_timeframes = {
            "4H",
            "1D",
            "1W",
            "1M",
        }

        if self.timeframe.upper() not in valid_timeframes:
            raise ValueError(
                f"Unsupported timeframe: {self.timeframe}. "
                f"Expected one of {sorted(valid_timeframes)}."
            )

        if (
            self.start is not None
            and self.end is not None
            and str(self.start) >= str(self.end)
        ):
            raise ValueError(
                "start must be earlier than end."
            )


# ----------------------------------------------------------------------
# Provider interface
# ----------------------------------------------------------------------


class MarketDataProvider(ABC):
    """
    Abstract market-data provider.

    Implementations must return a DataFrame with a DatetimeIndex and
    standard OHLCV columns where available.
    """

    name: str = "abstract"

    @abstractmethod
    def fetch(
        self,
        request: DataRequest,
    ) -> pd.DataFrame:
        """
        Fetch market data for a request.
        """
        raise NotImplementedError

    def supports(
        self,
        request: DataRequest,
    ) -> bool:
        """
        Whether the provider supports the requested data.
        """

        return True


# ----------------------------------------------------------------------
# Yahoo Finance provider
# ----------------------------------------------------------------------


class YahooFinanceProvider(
    MarketDataProvider
):
    """
    Adapter around YahooFinanceDownloader.

    This class deliberately contains vendor-specific behaviour in one
    place.

    IMPORTANT
    ---------
    Yahoo Finance's historical intraday availability is limited.
    Therefore this provider should NOT be interpreted as our final
    source for multi-year 4H research.
    """

    name = "yahoo_finance"

    def __init__(
        self,
        downloader: Optional[
            YahooFinanceDownloader
        ] = None,
    ) -> None:
        self.downloader = (
            downloader
            or YahooFinanceDownloader(
                DEFAULT_DATA_SOURCE
            )
        )

    def supports(
        self,
        request: DataRequest,
    ) -> bool:
        timeframe = (
            request.timeframe.upper()
        )

        if timeframe in {
            "1D",
            "1W",
            "1M",
        }:
            return True

        if timeframe == "4H":
            # Current Yahoo implementation obtains intraday data and
            # resamples it. It is supported technically, but not as a
            # multi-year historical source.
            return True

        return False

    def fetch(
        self,
        request: DataRequest,
    ) -> pd.DataFrame:
        if not self.supports(request):
            raise ValueError(
                f"{self.name} does not support "
                f"timeframe {request.timeframe}."
            )

        timeframe = (
            request.timeframe.upper()
        )

        symbol = request.symbol

        if timeframe == "1D":
            return self.downloader.download_daily(
                symbol=symbol,
                start=request.start,
                end=request.end,
            )

        if timeframe == "1W":
            return self.downloader.download_weekly(
                symbol=symbol,
                start=request.start,
                end=request.end,
            )

        if timeframe == "1M":
            return self.downloader.download_monthly(
                symbol=symbol,
                start=request.start,
                end=request.end,
            )

        if timeframe == "4H":
            return self.downloader.download_intraday(
                symbol=symbol,
                start=request.start,
                end=request.end,
            )

        raise ValueError(
            f"Unsupported timeframe: {request.timeframe}"
        )


# ----------------------------------------------------------------------
# Provider registry
# ----------------------------------------------------------------------


class ProviderRegistry:
    """
    Registry of available market-data providers.

    The registry allows the rest of the application to request data by
    provider name rather than constructing vendor-specific classes.
    """

    def __init__(
        self,
        providers: Optional[
            list[MarketDataProvider]
        ] = None,
    ) -> None:
        self._providers: dict[
            str,
            MarketDataProvider,
        ] = {}

        for provider in (
            providers or []
        ):
            self.register(provider)

    def register(
        self,
        provider: MarketDataProvider,
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(
            provider,
            MarketDataProvider,
        ):
            raise TypeError(
                "provider must implement MarketDataProvider."
            )

        name = provider.name.strip().lower()

        if not name:
            raise ValueError(
                "Provider name cannot be empty."
            )

        if (
            name in self._providers
            and not replace
        ):
            raise ValueError(
                f"Provider '{name}' is already registered."
            )

        self._providers[name] = provider

    def get(
        self,
        name: str,
    ) -> MarketDataProvider:
        key = name.strip().lower()

        try:
            return self._providers[key]
        except KeyError as exc:
            available = sorted(
                self._providers
            )

            raise KeyError(
                f"Unknown market-data provider '{name}'. "
                f"Available providers: {available}"
            ) from exc

    def names(
        self,
    ) -> list[str]:
        return sorted(
            self._providers
        )

    def fetch(
        self,
        request: DataRequest,
        *,
        provider: str = "yahoo_finance",
    ) -> pd.DataFrame:
        selected = self.get(
            provider
        )

        if not selected.supports(
            request
        ):
            raise ValueError(
                f"Provider '{provider}' does not support "
                f"{request.timeframe} data."
            )

        data = selected.fetch(
            request
        )

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                f"Provider '{provider}' returned "
                "a non-DataFrame result."
            )

        return data


# ----------------------------------------------------------------------
# Default registry
# ----------------------------------------------------------------------


def create_default_provider_registry() -> ProviderRegistry:
    """
    Create the standard provider registry.

    Additional providers can be registered later without changing
    application-level data code.
    """

    return ProviderRegistry(
        providers=[
            YahooFinanceProvider()
        ]
    )


DEFAULT_PROVIDER_REGISTRY = (
    create_default_provider_registry()
)


# ----------------------------------------------------------------------
# Convenience API
# ----------------------------------------------------------------------


def fetch_market_data(
    symbol: str,
    timeframe: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "yahoo_finance",
    registry: Optional[
        ProviderRegistry
    ] = None,
) -> pd.DataFrame:
    """
    Fetch market data through the provider abstraction.
    """

    request = DataRequest(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )

    selected_registry = (
        registry
        or DEFAULT_PROVIDER_REGISTRY
    )

    return selected_registry.fetch(
        request,
        provider=provider,
    )


def provider_status(
    registry: Optional[
        ProviderRegistry
    ] = None,
) -> pd.DataFrame:
    """
    Return a compact provider status table.
    """

    selected_registry = (
        registry
        or DEFAULT_PROVIDER_REGISTRY
    )

    rows = []

    for name in selected_registry.names():
        provider = selected_registry.get(
            name
        )

        rows.append(
            {
                "provider": name,
                "class": provider.__class__.__name__,
                "4H": provider.supports(
                    DataRequest(
                        symbol="TEST",
                        timeframe="4H",
                    )
                ),
                "1D": provider.supports(
                    DataRequest(
                        symbol="TEST",
                        timeframe="1D",
                    )
                ),
                "1W": provider.supports(
                    DataRequest(
                        symbol="TEST",
                        timeframe="1W",
                    )
                ),
                "1M": provider.supports(
                    DataRequest(
                        symbol="TEST",
                        timeframe="1M",
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "DataRequest",
    "MarketDataProvider",
    "YahooFinanceProvider",
    "ProviderRegistry",
    "DEFAULT_PROVIDER_REGISTRY",
    "create_default_provider_registry",
    "fetch_market_data",
    "provider_status",
]
