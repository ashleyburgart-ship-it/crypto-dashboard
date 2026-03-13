"""Trend and pullback signal logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

SCORE_MAX = {
    "score_htf_trend": 20,
    "score_ltf_trend": 15,
    "score_pullback": 15,
    "score_price_holding": 5,
    "score_volume": 10,
    "score_volatility": 10,
    "score_trend_strength": 5,
    "score_rsi": 10,
    "score_bonus_confluence": 10,
}
MIN_LONG_TREND_STRENGTH_RATIO = 0.004
MIN_DASHBOARD_SCORE = 75


def evaluate_trend(df: pd.DataFrame) -> str:
    """Return LONG, SHORT, or NEUTRAL based on the latest EMA relationship."""
    latest = df.iloc[-1]

    if pd.isna(latest["ema20"]) or pd.isna(latest["ema50"]):
        return "NEUTRAL"
    if latest["ema20"] > latest["ema50"]:
        return "LONG"
    if latest["ema20"] < latest["ema50"]:
        return "SHORT"
    return "NEUTRAL"


def _candle_touches_ema20(latest: pd.Series) -> bool:
    """Return True when the latest candle traded through EMA20."""
    return latest["low"] <= latest["ema20"] <= latest["high"]


def _rsi_quality_ratio(rsi_value: float, trend: str) -> float:
    """Return an RSI quality ratio from 0.0 to 1.0."""
    if pd.isna(rsi_value):
        return 0.0

    if trend == "LONG":
        lower_bound, upper_bound, ideal_value = 40, 55, 47.5
    elif trend == "SHORT":
        lower_bound, upper_bound, ideal_value = 45, 60, 52.5
    else:
        return 0.0

    if rsi_value < lower_bound or rsi_value > upper_bound:
        return 0.0

    half_range = (upper_bound - lower_bound) / 2
    distance_from_ideal = abs(rsi_value - ideal_value)
    return float(max(0, 1 - (distance_from_ideal / half_range)))


def _trend_strength_score(latest: pd.Series) -> int:
    """Score EMA separation as a percent of price from 0 to 5."""
    separation_pct = _trend_strength_ratio(latest)
    if separation_pct == 0:
        return 0
    normalized = min(separation_pct / 0.02, 1)
    return int(round(normalized * SCORE_MAX["score_trend_strength"]))


def _trend_strength_ratio(latest: pd.Series) -> float:
    """Return EMA separation as a fraction of price."""
    close_price = latest["close"]
    if pd.isna(close_price) or close_price == 0 or pd.isna(latest["ema20"]) or pd.isna(latest["ema50"]):
        return 0.0
    return float(abs(latest["ema20"] - latest["ema50"]) / close_price)


def _long_quality_failures(latest_15m: pd.Series) -> list[str]:
    """Return unmet long-only quality gates."""
    failures = []
    if _volume_confirmation(latest_15m) != "ABOVE_AVG":
        failures.append("weak volume")
    if not _volatility_ok(latest_15m):
        failures.append("unstable volatility")
    if _trend_strength_ratio(latest_15m) < MIN_LONG_TREND_STRENGTH_RATIO:
        failures.append("weak trend strength")
    return failures


def _volume_confirmation(latest_15m: pd.Series) -> str:
    """Return a label describing the current volume versus its rolling average."""
    if pd.isna(latest_15m["volume"]) or pd.isna(latest_15m["avg_volume_20"]):
        return "n/a"
    return "ABOVE_AVG" if latest_15m["volume"] > latest_15m["avg_volume_20"] else "BELOW_AVG"


def _volatility_ok(latest_15m: pd.Series) -> bool:
    """Return True when ATR14 is not more than 20% above its recent average."""
    if pd.isna(latest_15m["atr14"]) or pd.isna(latest_15m["atr14_20_avg"]) or latest_15m["atr14_20_avg"] <= 0:
        return False
    return bool(latest_15m["atr14"] <= latest_15m["atr14_20_avg"] * 1.2)


def _pullback_quality_score(latest_15m: pd.Series) -> int:
    """Score pullback confirmation based on EMA interaction and proximity."""
    if not _candle_touches_ema20(latest_15m) or pd.isna(latest_15m["atr14"]) or latest_15m["atr14"] <= 0:
        return 0

    proximity = abs(latest_15m["close"] - latest_15m["ema20"]) / latest_15m["atr14"]
    normalized = max(0.0, 1 - min(proximity / 0.75, 1))
    return int(round(normalized * SCORE_MAX["score_pullback"]))


def _price_holding_ema_score(latest_15m: pd.Series, trend_15m: str) -> int:
    """Score whether price is holding on the expected side of EMA20."""
    close_price = latest_15m["close"]
    ema20 = latest_15m["ema20"]
    if pd.isna(close_price) or pd.isna(ema20):
        return 0

    if trend_15m == "LONG" and close_price >= ema20:
        return SCORE_MAX["score_price_holding"]
    if trend_15m == "SHORT" and close_price <= ema20:
        return SCORE_MAX["score_price_holding"]
    return 0


def _score_components(latest_15m: pd.Series, trend_15m: str, trend_1h: str) -> dict[str, int]:
    """Return the weighted confluence component scores."""
    score_htf_trend = SCORE_MAX["score_htf_trend"] if trend_1h in {"LONG", "SHORT"} and trend_1h == trend_15m else 0
    score_ltf_trend = SCORE_MAX["score_ltf_trend"] if trend_15m in {"LONG", "SHORT"} else 0
    score_pullback = _pullback_quality_score(latest_15m)
    score_price_holding = _price_holding_ema_score(latest_15m, trend_15m)
    score_volume = SCORE_MAX["score_volume"] if _volume_confirmation(latest_15m) == "ABOVE_AVG" else 0
    score_volatility = SCORE_MAX["score_volatility"] if _volatility_ok(latest_15m) else 0
    score_trend_strength = _trend_strength_score(latest_15m)
    score_rsi = int(round(_rsi_quality_ratio(latest_15m["rsi14"], trend_15m) * SCORE_MAX["score_rsi"]))
    score_bonus_confluence = (
        SCORE_MAX["score_bonus_confluence"]
        if all(
            [
                score_htf_trend > 0,
                score_ltf_trend > 0,
                score_pullback > 0,
                score_rsi > 0,
                score_price_holding > 0,
                score_volume > 0,
                score_volatility > 0,
            ]
        )
        else 0
    )

    score_total = int(
        np.clip(
            score_htf_trend
            + score_ltf_trend
            + score_pullback
            + score_price_holding
            + score_volume
            + score_volatility
            + score_trend_strength
            + score_rsi
            + score_bonus_confluence,
            0,
            100,
        )
    )
    return {
        "score_total": score_total,
        "score_htf_trend": score_htf_trend,
        "score_ltf_trend": score_ltf_trend,
        "score_pullback": score_pullback,
        "score_price_holding": score_price_holding,
        "score_volume": score_volume,
        "score_volatility": score_volatility,
        "score_trend_strength": score_trend_strength,
        "score_rsi": score_rsi,
        "score_bonus_confluence": score_bonus_confluence,
    }


def evaluate_setup(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> dict[str, object]:
    """Return multi-timeframe pullback setup details for the latest candle."""
    latest_15m = df_15m.iloc[-1]
    trend_15m = evaluate_trend(df_15m)
    trend_1h = evaluate_trend(df_1h)
    result = {
        "trend_1h": trend_1h,
        "trend_15m": trend_15m,
        "setup": "NONE",
        "score": 0,
        "score_total": 0,
        "score_htf_trend": 0,
        "score_ltf_trend": 0,
        "score_pullback": 0,
        "score_price_holding": 0,
        "score_volume": 0,
        "score_volatility": 0,
        "score_trend_strength": 0,
        "score_rsi": 0,
        "score_bonus_confluence": 0,
        "atr14": pd.NA,
        "atr14_20_avg": pd.NA,
        "volatility_ok": False,
        "volume_confirmation": "n/a",
        "qualification_failures": [],
        "score_breakdown": {},
        "entry_price": pd.NA,
        "stop_loss": pd.NA,
        "target": pd.NA,
    }

    required_fields = ["ema20", "ema50", "atr14", "atr14_20_avg", "atr_median_20", "rsi14", "low", "high"]
    if latest_15m[required_fields].isna().any():
        return result

    result["atr14"] = latest_15m["atr14"]
    result["atr14_20_avg"] = latest_15m["atr14_20_avg"]
    result["volatility_ok"] = _volatility_ok(latest_15m)
    result["volume_confirmation"] = _volume_confirmation(latest_15m)

    score_components = _score_components(latest_15m, trend_15m, trend_1h)
    long_candidate = (
        trend_15m == "LONG"
        and trend_1h == "LONG"
        and _candle_touches_ema20(latest_15m)
        and latest_15m["atr14"] > latest_15m["atr_median_20"]
        and 40 <= latest_15m["rsi14"] <= 55
    )
    long_quality_failures = _long_quality_failures(latest_15m) if long_candidate else []
    if long_quality_failures:
        score_components["score_total"] = min(score_components["score_total"], MIN_DASHBOARD_SCORE - 1)

    result.update(score_components)
    result["score"] = score_components["score_total"]
    result["qualification_failures"] = long_quality_failures
    result["score_breakdown"] = {
        "market_structure": {
            "1h_trend_alignment": score_components["score_htf_trend"],
            "15m_trend_alignment": score_components["score_ltf_trend"],
            "trend_strength": score_components["score_trend_strength"],
        },
        "entry_quality": {
            "pullback_confirmation": score_components["score_pullback"],
            "rsi_momentum_confirmation": score_components["score_rsi"],
            "price_holding_ema": score_components["score_price_holding"],
        },
        "market_conditions": {
            "volume_confirmation": score_components["score_volume"],
            "volatility_healthy": score_components["score_volatility"],
        },
        "bonus_confluence": {
            "all_major_signals_aligned": score_components["score_bonus_confluence"],
        },
        "total": score_components["score_total"],
    }

    if trend_15m == "NEUTRAL" or trend_1h == "NEUTRAL":
        return result

    if trend_15m != trend_1h:
        return result

    if not _candle_touches_ema20(latest_15m):
        return result

    if latest_15m["atr14"] <= latest_15m["atr_median_20"]:
        return result

    entry_price = latest_15m["ema20"]

    if trend_15m == "LONG":
        if not (40 <= latest_15m["rsi14"] <= 55):
            return result
        if long_quality_failures:
            return result

        stop_loss = min(latest_15m["low"], entry_price - (latest_15m["atr14"] * 0.5))
        risk = entry_price - stop_loss
        if risk <= 0:
            return result

        result.update(
            {
                "setup": "PULLBACK_LONG",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": entry_price + (2 * risk),
            }
        )
        return result

    if trend_15m == "SHORT":
        if not (45 <= latest_15m["rsi14"] <= 60):
            return result

        stop_loss = max(latest_15m["high"], entry_price + (latest_15m["atr14"] * 0.5))
        risk = stop_loss - entry_price
        if risk <= 0:
            return result

        result.update(
            {
                "setup": "PULLBACK_SHORT",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": entry_price - (2 * risk),
            }
        )

    return result
