from io import BytesIO

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalise_column_name(name):
    """Convert a column name to a standard lowercase form."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _normalise_yahoo_columns(df):
    """
    Normalize both normal and MultiIndex yfinance output.

    yfinance may return columns such as:
        Open, High, Low, Close, Volume

    or, for a single ticker:
        (Open, RELIANCE.NS), (High, RELIANCE.NS), ...

    Some versions can also return the ticker as the first level.
    """
    if df is None or df.empty:
        raise ValueError("Yahoo Finance returned no data.")

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # Find the MultiIndex level containing OHLCV names.
        price_level = None

        for level in range(out.columns.nlevels):
            values = {
                _normalise_column_name(value)
                for value in out.columns.get_level_values(level)
            }

            if len(values.intersection(REQUIRED_COLUMNS)) >= 2:
                price_level = level
                break

        if price_level is None:
            raise ValueError(
                "Unable to identify OHLCV columns returned by Yahoo Finance."
            )

        out.columns = [
            _normalise_column_name(value)
            for value in out.columns.get_level_values(price_level)
        ]

    else:
        out.columns = [
            _normalise_column_name(column)
            for column in out.columns
        ]

    # Use adjusted close only if a normal close column is unavailable.
    if "close" not in out.columns and "adj_close" in out.columns:
        out = out.rename(columns={"adj_close": "close"})

    return out


def _get_single_series(df, column_name):
    """
    Safely return one 1-D Series.

    Duplicate column names can make df[column_name] return a DataFrame,
    which is the source of errors such as:
    'arg must be a list, tuple, 1-d array, or Series'.
    """
    matches = [
        column
        for column in df.columns
        if _normalise_column_name(column) == column_name
    ]

    if not matches:
        raise ValueError(
            f"Yahoo Finance data is missing required column: {column_name}"
        )

    candidates = []

    for column in matches:
        value = df[column]

        if isinstance(value, pd.Series):
            candidates.append(value)

        elif isinstance(value, pd.DataFrame):
            for sub_column in value.columns:
                series = value[sub_column]
                if isinstance(series, pd.Series):
                    candidates.append(series)

    if not candidates:
        raise ValueError(
            f"Could not convert Yahoo Finance '{column_name}' to a 1-D Series."
        )

    return candidates[0]


def prepare_ohlcv(df):
    """Convert downloaded/uploaded data into clean OHLCV format."""
    normalized = _normalise_yahoo_columns(df)

    data = {}

    for column in REQUIRED_COLUMNS:
        data[column] = _get_single_series(normalized, column)

    out = pd.DataFrame(data, index=normalized.index)

    for column in REQUIRED_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(
        subset=["open", "high", "low", "close"]
    )

    # Remove duplicate dates and keep chronological order.
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()

    if out.empty:
        raise ValueError("Yahoo Finance returned no usable OHLCV data.")

    return out


def download_data(symbol, period="1y", interval="1d"):
    """Download OHLCV data from Yahoo Finance."""
    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Please enter a stock symbol.")

    try:
        data = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )

        return prepare_ohlcv(data)

    except Exception as exc:
        raise RuntimeError(
            f"Yahoo Finance download failed: {exc}"
        ) from exc


def load_csv_data(uploaded_file):
    """Load OHLCV data from a CSV file."""
    if uploaded_file is None:
        raise ValueError("Please upload a CSV file.")

    if isinstance(uploaded_file, (str, bytes)):
        data = pd.read_csv(uploaded_file)
    else:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception:
            if hasattr(uploaded_file, "getvalue"):
                data = pd.read_csv(
                    BytesIO(uploaded_file.getvalue())
                )
            else:
                raise

    return prepare_ohlcv(data)
