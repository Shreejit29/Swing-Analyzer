"""
Historical market-data caching layer.

Purpose
-------
Cache downloaded OHLCV data locally so that repeated research runs do
not unnecessarily request the same data from an external provider.

Design goals
------------
    - deterministic cache keys
    - atomic writes
    - corruption detection
    - expiration support
    - timezone normalization
    - no silent use of invalid cache files
    - easy cache invalidation

IMPORTANT
---------
Caching does not change the research data itself.

A cache hit must return exactly the cached observations. The caller is
still responsible for OHLCV quality validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class CacheConfig:
    """
    Configuration for the local data cache.
    """

    directory: str = "data/cache"

    enabled: bool = True

    default_expiry_hours: Optional[float] = None

    compress: bool = True

    verify_checksum: bool = True

    create_directory: bool = True

    def __post_init__(self) -> None:
        if not self.directory:
            raise ValueError(
                "Cache directory cannot be empty."
            )

        if (
            self.default_expiry_hours is not None
            and self.default_expiry_hours < 0
        ):
            raise ValueError(
                "default_expiry_hours cannot be negative."
            )


# ----------------------------------------------------------------------
# Cache metadata
# ----------------------------------------------------------------------


@dataclass
class CacheMetadata:
    """
    Metadata stored alongside each cached dataset.
    """

    key: str

    symbol: str

    timeframe: str

    interval: str

    start: Optional[str]

    end: Optional[str]

    created_at: str

    rows: int

    columns: list[str]

    checksum: str

    version: str = "1.0"

    extra: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "interval": self.interval,
            "start": self.start,
            "end": self.end,
            "created_at": self.created_at,
            "rows": self.rows,
            "columns": self.columns,
            "checksum": self.checksum,
            "version": self.version,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "CacheMetadata":
        required = [
            "key",
            "symbol",
            "timeframe",
            "interval",
            "created_at",
            "rows",
            "columns",
            "checksum",
        ]

        missing = [
            key
            for key in required
            if key not in data
        ]

        if missing:
            raise ValueError(
                "Cache metadata is missing fields: "
                f"{missing}"
            )

        return cls(
            key=str(
                data["key"]
            ),
            symbol=str(
                data["symbol"]
            ),
            timeframe=str(
                data["timeframe"]
            ),
            interval=str(
                data["interval"]
            ),
            start=data.get(
                "start"
            ),
            end=data.get(
                "end"
            ),
            created_at=str(
                data["created_at"]
            ),
            rows=int(
                data["rows"]
            ),
            columns=list(
                data["columns"]
            ),
            checksum=str(
                data["checksum"]
            ),
            version=str(
                data.get(
                    "version",
                    "1.0",
                )
            ),
            extra=dict(
                data.get(
                    "extra",
                    {},
                )
            ),
        )


# ----------------------------------------------------------------------
# Cache key
# ----------------------------------------------------------------------


def build_cache_key(
    *,
    symbol: str,
    timeframe: str,
    interval: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "unknown",
    extra: Optional[
        Dict[str, Any]
    ] = None,
) -> str:
    """
    Build a deterministic cache key.

    All parameters that can materially affect the downloaded dataset
    should be included.
    """

    payload = {
        "symbol": str(
            symbol
        ).upper(),
        "timeframe": str(
            timeframe
        ).upper(),
        "interval": str(
            interval
        ).lower(),
        "start": start,
        "end": end,
        "provider": str(
            provider
        ).lower(),
        "extra": extra or {},
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


# ----------------------------------------------------------------------
# Cache manager
# ----------------------------------------------------------------------


class DataCache:
    """
    Local persistent cache for pandas DataFrames.

    Data is stored as Parquet when possible.

    A JSON metadata file is stored beside the data file and contains a
    SHA-256 checksum for corruption detection.
    """

    def __init__(
        self,
        config: Optional[
            CacheConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or CacheConfig()
        )

        self.directory = Path(
            self.config.directory
        )

        if (
            self.config.create_directory
            and self.config.enabled
        ):
            self.directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def data_path(
        self,
        key: str,
    ) -> Path:
        return self.directory / (
            f"{key}.parquet"
        )

    def metadata_path(
        self,
        key: str,
    ) -> Path:
        return self.directory / (
            f"{key}.json"
        )

    # ------------------------------------------------------------------
    # Existence
    # ------------------------------------------------------------------

    def exists(
        self,
        key: str,
    ) -> bool:
        if not self.config.enabled:
            return False

        return (
            self.data_path(key).exists()
            and self.metadata_path(key).exists()
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        data: pd.DataFrame,
        *,
        key: str,
        symbol: str,
        timeframe: str,
        interval: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        extra: Optional[
            Dict[str, Any]
        ] = None,
    ) -> CacheMetadata:
        """
        Save a DataFrame to the cache.

        Writes are atomic: temporary files are created first and then
        moved into place.
        """

        if not self.config.enabled:
            raise RuntimeError(
                "Data cache is disabled."
            )

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        frame = self._normalize_dataframe(
            data
        )

        checksum = self._checksum(
            frame
        )

        created_at = (
            pd.Timestamp.utcnow()
            .isoformat()
        )

        metadata = CacheMetadata(
            key=key,
            symbol=symbol,
            timeframe=timeframe,
            interval=interval,
            start=start,
            end=end,
            created_at=created_at,
            rows=len(frame),
            columns=list(
                frame.columns
            ),
            checksum=checksum,
            extra=extra or {},
        )

        data_path = self.data_path(
            key
        )

        metadata_path = self.metadata_path(
            key
        )

        self._atomic_write_parquet(
            frame,
            data_path,
        )

        self._atomic_write_json(
            metadata.to_dict(),
            metadata_path,
        )

        return metadata

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        key: str,
        *,
        max_age_hours: Optional[
            float
        ] = None,
    ) -> Optional[
        pd.DataFrame
    ]:
        """
        Load cached data.

        Returns None when:
            - cache is disabled
            - cache does not exist
            - cache is expired

        Raises an error for corrupted cache data rather than silently
        continuing with potentially invalid historical data.
        """

        if not self.config.enabled:
            return None

        if not self.exists(key):
            return None

        metadata = self._load_metadata(
            key
        )

        expiry = (
            max_age_hours
            if max_age_hours is not None
            else self.config
            .default_expiry_hours
        )

        if expiry is not None:
            if self._is_expired(
                metadata,
                expiry,
            ):
                return None

        data_path = self.data_path(
            key
        )

        try:
            frame = pd.read_parquet(
                data_path
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to read cache file '{data_path}': {exc}"
            ) from exc

        frame = self._normalize_dataframe(
            frame
        )

        if metadata.rows != len(frame):
            raise RuntimeError(
                "Cache integrity failure: row count does not match metadata."
            )

        if metadata.columns != list(
            frame.columns
        ):
            raise RuntimeError(
                "Cache integrity failure: columns do not match metadata."
            )

        if self.config.verify_checksum:
            checksum = self._checksum(
                frame
            )

            if checksum != metadata.checksum:
                raise RuntimeError(
                    "Cache integrity failure: checksum mismatch."
                )

        return frame

    # ------------------------------------------------------------------
    # Get-or-fetch
    # ------------------------------------------------------------------

    def get_or_fetch(
        self,
        *,
        key: str,
        fetcher,
        symbol: str,
        timeframe: str,
        interval: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_age_hours: Optional[
            float
        ] = None,
        extra: Optional[
            Dict[str, Any]
        ] = None,
    ) -> pd.DataFrame:
        """
        Return cached data when available; otherwise call ``fetcher``.

        ``fetcher`` must be a callable returning a DataFrame.
        """

        if self.config.enabled:
            cached = self.load(
                key,
                max_age_hours=max_age_hours,
            )

            if cached is not None:
                return cached

        data = fetcher()

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "Fetcher must return a pandas DataFrame."
            )

        if self.config.enabled:
            self.save(
                data,
                key=key,
                symbol=symbol,
                timeframe=timeframe,
                interval=interval,
                start=start,
                end=end,
                extra=extra,
            )

        return data

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metadata(
        self,
        key: str,
    ) -> Optional[
        CacheMetadata
    ]:
        if not self.exists(key):
            return None

        return self._load_metadata(
            key
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(
        self,
        key: str,
    ) -> bool:
        """
        Delete one cache entry.

        Returns True when something was deleted.
        """

        deleted = False

        for path in (
            self.data_path(key),
            self.metadata_path(key),
        ):
            if path.exists():
                path.unlink()
                deleted = True

        return deleted

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(
        self,
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> int:
        """
        Clear cache entries.

        When filters are supplied, only matching metadata entries are
        deleted.
        """

        if not self.directory.exists():
            return 0

        deleted = 0

        metadata_files = self.directory.glob(
            "*.json"
        )

        for metadata_path in metadata_files:
            try:
                payload = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )

                metadata = (
                    CacheMetadata.from_dict(
                        payload
                    )
                )

            except Exception:
                # Never delete an unknown/corrupt file automatically.
                continue

            if (
                symbol is not None
                and metadata.symbol.upper()
                != symbol.upper()
            ):
                continue

            if (
                timeframe is not None
                and metadata.timeframe.upper()
                != timeframe.upper()
            ):
                continue

            if self.delete(
                metadata.key
            ):
                deleted += 1

        return deleted

    # ------------------------------------------------------------------
    # List entries
    # ------------------------------------------------------------------

    def list_entries(
        self,
    ) -> list[CacheMetadata]:
        """
        Return readable cache metadata entries.
        """

        if not self.directory.exists():
            return []

        entries = []

        for metadata_path in sorted(
            self.directory.glob(
                "*.json"
            )
        ):
            try:
                payload = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )

                entries.append(
                    CacheMetadata.from_dict(
                        payload
                    )
                )

            except Exception:
                continue

        return entries

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_dataframe(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = data.copy()

        if isinstance(
            frame.columns,
            pd.MultiIndex,
        ):
            frame.columns = [
                "_".join(
                    str(part)
                    for part in column
                    if str(part) != ""
                ).strip("_")
                for column in frame.columns
            ]

        if not isinstance(
            frame.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "Cached market data must use a DatetimeIndex."
            )

        frame = frame.sort_index()

        if frame.index.has_duplicates:
            frame = frame[
                ~frame.index.duplicated(
                    keep="last"
                )
            ]

        # Normalize timezone representation.
        if frame.index.tz is not None:
            frame.index = (
                frame.index.tz_convert(
                    "Asia/Kolkata"
                ).tz_localize(None)
            )

        for column in frame.columns:
            if pd.api.types.is_object_dtype(
                frame[column]
            ):
                converted = pd.to_numeric(
                    frame[column],
                    errors="ignore",
                )

                frame[column] = converted

        return frame

    # ------------------------------------------------------------------
    # Checksum
    # ------------------------------------------------------------------

    @staticmethod
    def _checksum(
        data: pd.DataFrame,
    ) -> str:
        """
        Generate deterministic SHA-256 checksum.
        """

        frame = data.copy()

        # Include index in the checksum.
        index_values = np.asarray(
            frame.index.view(
                "int64"
            )
        )

        values = frame.to_numpy(
            dtype=object
        )

        payload = {
            "columns": list(
                frame.columns
            ),
            "index": index_values.tolist(),
            "values": values.tolist(),
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            default=str,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )

        return hashlib.sha256(
            encoded
        ).hexdigest()

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    def _load_metadata(
        self,
        key: str,
    ) -> CacheMetadata:
        path = self.metadata_path(
            key
        )

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to read cache metadata '{path}': {exc}"
            ) from exc

        metadata = CacheMetadata.from_dict(
            payload
        )

        if metadata.key != key:
            raise RuntimeError(
                "Cache metadata key does not match requested key."
            )

        return metadata

    # ------------------------------------------------------------------
    # Expiry
    # ------------------------------------------------------------------

    @staticmethod
    def _is_expired(
        metadata: CacheMetadata,
        max_age_hours: float,
    ) -> bool:
        if max_age_hours < 0:
            raise ValueError(
                "max_age_hours cannot be negative."
            )

        created = pd.Timestamp(
            metadata.created_at
        )

        now = pd.Timestamp.utcnow()

        age_hours = (
            now - created
        ).total_seconds() / 3600.0

        return (
            age_hours
            > max_age_hours
        )

    # ------------------------------------------------------------------
    # Atomic Parquet write
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write_parquet(
        data: pd.DataFrame,
        destination: Path,
    ) -> None:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        suffix = (
            ".parquet"
        )

        fd, temporary = (
            tempfile.mkstemp(
                suffix=suffix,
                dir=str(
                    destination.parent
                ),
            )
        )

        os.close(fd)

        temporary_path = Path(
            temporary
        )

        try:
            data.to_parquet(
                temporary_path,
                compression=(
                    "snappy"
                ),
                index=True,
            )

            os.replace(
                temporary_path,
                destination,
            )

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    # ------------------------------------------------------------------
    # Atomic JSON write
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write_json(
        data: Dict[str, Any],
        destination: Path,
    ) -> None:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fd, temporary = (
            tempfile.mkstemp(
                suffix=".json",
                dir=str(
                    destination.parent
                ),
            )
        )

        os.close(fd)

        temporary_path = Path(
            temporary
        )

        try:
            temporary_path.write_text(
                json.dumps(
                    data,
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
                encoding="utf-8",
            )

            os.replace(
                temporary_path,
                destination,
            )

        finally:
            if temporary_path.exists():
                temporary_path.unlink()


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------


def cached_download(
    fetcher,
    *,
    symbol: str,
    timeframe: str,
    interval: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "unknown",
    config: Optional[
        CacheConfig
    ] = None,
    max_age_hours: Optional[
        float
    ] = None,
    extra: Optional[
        Dict[str, Any]
    ] = None,
) -> pd.DataFrame:
    """
    Download through the local cache.

    Example:

        data = cached_download(
            lambda: provider.download(...),
            symbol="RELIANCE.NS",
            timeframe="1D",
            interval="1d",
        )
    """

    cache = DataCache(
        config=config
    )

    key = build_cache_key(
        symbol=symbol,
        timeframe=timeframe,
        interval=interval,
        start=start,
        end=end,
        provider=provider,
        extra=extra,
    )

    return cache.get_or_fetch(
        key=key,
        fetcher=fetcher,
        symbol=symbol,
        timeframe=timeframe,
        interval=interval,
        start=start,
        end=end,
        max_age_hours=max_age_hours,
        extra=extra,
    )


def cache_summary(
    cache: DataCache,
) -> Dict[str, Any]:
    """
    Return a compact cache summary.
    """

    entries = cache.list_entries()

    return {
        "enabled": cache.config.enabled,
        "directory": str(
            cache.directory
        ),
        "entries": len(
            entries
        ),
        "symbols": sorted(
            {
                entry.symbol
                for entry in entries
            }
        ),
        "timeframes": sorted(
            {
                entry.timeframe
                for entry in entries
            }
        ),
    }


__all__ = [
    "CacheConfig",
    "CacheMetadata",
    "DataCache",
    "build_cache_key",
    "cached_download",
    "cache_summary",
]
