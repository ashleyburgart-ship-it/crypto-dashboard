"""Indicator calculations for the crypto scanner."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the input data with scanner indicators added."""
    result = df.copy()

    result["ema20"] = result["close"].ewm(span=20, adjust=False).mean()
    result["ema50"] = result["close"].ewm(span=50, adjust=False).mean()

    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr14"] = true_range.rolling(window=14).mean()
    result["atr14_20_avg"] = result["atr14"].rolling(window=20).mean()
    result["atr_median_20"] = result["atr14"].rolling(window=20).median()

    delta = result["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    result["rsi14"] = 100 - (100 / (1 + relative_strength))
    result.loc[average_loss == 0, "rsi14"] = 100

    result["avg_volume_20"] = result["volume"].rolling(window=20).mean()
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
