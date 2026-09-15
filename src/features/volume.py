
Today 6:30 PM

AI-Swing-Analyser-UPDATED(1).zip
Zip Archive
go through it


Swing-Analyzer-main.zip
Zip Archive


Swing-Analyzer-Simple(1).zip
Zip Archive


Analyzed
Next file: src/features/price_action.py

This is a substantial upgrade. It adds:

Candle anatomy and wick features

Bullish/bearish/doji detection

Gap and multi-day returns

Support/resistance

Leakage-safe breakout/breakdown features

Higher-high / higher-low market structure

ATR-normalized price movement

Bullish/bearish streaks

Structure scores

Replace your existing file with this:

Download price_action.py

Test the app after replacing it. If everything works, say “next” and we'll move to volume.py. 


price_action.py
Code


Missing required columns: ['Close', 'High', 'Low', 'Open']  


Analyzed
Yes — this is a column-format compatibility issue, not a problem with the price-action calculations.

The new version was expecting exact Open/High/Low/Close, while your pipeline is likely passing lowercase or Yahoo-style flattened columns.

Replace your current src/features/price_action.py with this fixed version:

Download price_action_FIXED.py

What I fixed
It now automatically recognizes:

Open / High / Low / Close

open / high / low / close

Adj Close

Yahoo-style flattened names such as Close_RELIANCE.NS

Simple MultiIndex columns

It also gives you the available column names in the error message if something unexpected reaches the function.

Replace only this file and run the app again. If it works, say “next”. 


price_action_FIXED.py
Code


'volume'


Analyzed
That 'volume' error is coming from volume.py.

Replace only src/features/volume.py with:

Download volume_FIXED.py

Fixed
Handles Volume, volume, Vol

Handles Yahoo-style column names

Handles missing volume without crashing

Adds relative volume

Volume spikes

Volume trend/slope

OBV

Price-volume confirmation

Bullish/bearish volume signals

Run the app again.

If it works, say “next”. 


volume_FIXED.py
Code


Library
/
volume_FIXED.py


1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
"""
Robust volume feature engineering for the Swing Analyzer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    """Find a column case-insensitively, including simple Yahoo-style names."""
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    for col in df.columns:
        text = str(col).strip().lower()
        if any(text.startswith(name.lower() + "_") or text.endswith("_" + name.lower())
               for name in names):
            return col

    return None


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume, relative-volume, volume-trend and price-volume features.

    The function gracefully handles data without a Volume column. In that
    case, volume-derived features are filled with NaN instead of crashing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    out = df.copy()

    volume_col = _find_column(out, ["Volume", "volume", "Vol"])
    close_col = _find_column(out, ["Close", "close", "Adj Close", "adj close"])

    # No volume data: keep the feature schema but don't stop the analyzer.
    if volume_col is None:
        volume = pd.Series(np.nan, index=out.index, dtype=float)
    else:
        volume = pd.to_numeric(out[volume_col], errors="coerce")

    if close_col is None:
        close = pd.Series(np.nan, index=out.index, dtype=float)
    else:
        close = pd.to_numeric(out[close_col], errors="coerce")

    out["Volume_Clean"] = volume

    # Basic volume statistics
    out["Volume_SMA5"] = volume.rolling(5, min_periods=1).mean()
    out["Volume_SMA10"] = volume.rolling(10, min_periods=1).mean()
    out["Volume_SMA20"] = volume.rolling(20, min_periods=1).mean()
