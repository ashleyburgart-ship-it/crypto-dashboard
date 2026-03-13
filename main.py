"""Entry point for the crypto market scanner."""

from __future__ import annotations

import os
import time
from csv import DictReader, DictWriter
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    BASE_SYMBOLS,
    EXCHANGE_ID,
    HIGHER_TIMEFRAME,
    OHLCV_LIMIT,
    PREFERRED_QUOTES,
    SCAN_INTERVAL_SECONDS,
    TIMEFRAME,
)
from data import SymbolUnavailableError, create_exchange, fetch_ohlcv, resolve_market_symbol
from indicators import add_indicators
from signals import evaluate_setup

RESULT_COLUMNS = [
    "Symbol",
    "1H Trend",
    "15m Trend",
    "Setup",
    "Score",
    "score_total",
    "score_htf_trend",
    "score_ltf_trend",
    "score_pullback",
    "score_price_holding",
    "score_volume",
    "score_volatility",
    "score_trend_strength",
    "score_rsi",
    "score_bonus_confluence",
    "ATR14",
    "ATR14_20_avg",
    "Volatility OK",
    "Entry Price",
    "Stop Loss",
    "Target (2R)",
]

DASHBOARD_EXTRA_COLUMNS = ["Trend Strength", "Volume Confirmation"]
SIGNAL_LOG_PATH = Path(__file__).resolve().parent / "signal_log.csv"
SIGNAL_LOG_HEADERS = [
    "timestamp",
    "symbol",
    "setup",
    "score",
    "entry_price",
    "stop_loss",
    "target_price",
    "trend_strength",
    "volume_confirmation",
    "atr14",
    "atr14_20_avg",
    "volatility_ok",
    "score_total",
    "score_htf_trend",
    "score_ltf_trend",
    "score_pullback",
    "score_price_holding",
    "score_volume",
    "score_volatility",
    "score_trend_strength",
    "score_rsi",
    "score_bonus_confluence",
    "result",
    "R_multiple",
    "max_price_after_entry",
    "min_price_after_entry",
    "time_to_resolution_minutes",
]


def format_number(value: float) -> str:
    """Format numeric output for the results table."""
    if pd.isna(value):
        return "n/a"
    return f"{value:,.4f}"


def clear_console() -> None:
    """Clear the terminal so the latest scan stays readable."""
    os.system("cls" if os.name == "nt" else "clear")


def current_timestamp() -> str:
    """Return the local timestamp for the current scan."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _coerce_float(value: object) -> float | None:
    """Convert a value to float when possible."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_signal_log_frame() -> pd.DataFrame:
    """Load the signal log as a DataFrame with normalized columns."""
    log_path = initialize_signal_log()
    signal_log = pd.read_csv(log_path)
    if signal_log.empty:
        return pd.DataFrame(columns=SIGNAL_LOG_HEADERS)

    for column in SIGNAL_LOG_HEADERS:
        if column not in signal_log.columns:
            signal_log[column] = pd.NA

    signal_log["timestamp"] = pd.to_datetime(signal_log["timestamp"], utc=True, errors="coerce")
    signal_log["result"] = signal_log["result"].fillna("").replace("", "OPEN")

    numeric_columns = [
        "entry_price",
        "stop_loss",
        "target_price",
        "score",
        "trend_strength",
        "atr14",
        "atr14_20_avg",
        "score_total",
        "score_htf_trend",
        "score_ltf_trend",
        "score_pullback",
        "score_price_holding",
        "score_volume",
        "score_volatility",
        "score_trend_strength",
        "score_rsi",
        "score_bonus_confluence",
        "R_multiple",
        "max_price_after_entry",
        "min_price_after_entry",
        "time_to_resolution_minutes",
    ]
    for column in numeric_columns:
        signal_log[column] = pd.to_numeric(signal_log[column], errors="coerce")

    return signal_log[SIGNAL_LOG_HEADERS].copy()


def _write_signal_log_frame(signal_log: pd.DataFrame) -> None:
    """Persist the normalized signal log DataFrame to disk."""
    log_path = initialize_signal_log()
    signal_log = signal_log.copy()
    signal_log["timestamp"] = signal_log["timestamp"].apply(
        lambda value: value.isoformat() if pd.notna(value) else ""
    )
    signal_log.to_csv(log_path, index=False, columns=SIGNAL_LOG_HEADERS)


def _unavailable_row(base_symbol: str) -> dict[str, object]:
    """Return a standard row for unavailable symbols."""
    return {
        "Symbol": base_symbol,
        "1H Trend": "UNAVAILABLE",
        "15m Trend": "UNAVAILABLE",
        "Setup": "NONE",
        "Score": 0,
        "Entry Price": pd.NA,
        "Stop Loss": pd.NA,
        "Target (2R)": pd.NA,
    }


def _error_row(base_symbol: str, error: str) -> dict[str, object]:
    """Return a standard row for symbol-level errors."""
    return {
        "Symbol": base_symbol,
        "1H Trend": f"ERROR: {error}",
        "15m Trend": f"ERROR: {error}",
        "Setup": "NONE",
        "Score": 0,
        "Entry Price": pd.NA,
        "Stop Loss": pd.NA,
        "Target (2R)": pd.NA,
    }


def build_exchange():
    """Create and initialize the configured exchange client."""
    exchange = create_exchange(EXCHANGE_ID)
    exchange.load_markets()
    return exchange


def close_exchange(exchange) -> None:
    """Close the exchange client when supported."""
    close_method = getattr(exchange, "close", None)
    if callable(close_method):
        close_method()


def initialize_signal_log() -> Path:
    """Create the signal log CSV with headers if it does not exist."""
    if not SIGNAL_LOG_PATH.exists():
        with SIGNAL_LOG_PATH.open("w", newline="", encoding="utf-8") as log_file:
            writer = DictWriter(log_file, fieldnames=SIGNAL_LOG_HEADERS)
            writer.writeheader()
        return SIGNAL_LOG_PATH

    with SIGNAL_LOG_PATH.open("r", newline="", encoding="utf-8") as log_file:
        reader = DictReader(log_file)
        existing_headers = reader.fieldnames or []
        existing_rows = list(reader)

    if existing_headers != SIGNAL_LOG_HEADERS:
        with SIGNAL_LOG_PATH.open("w", newline="", encoding="utf-8") as log_file:
            writer = DictWriter(log_file, fieldnames=SIGNAL_LOG_HEADERS)
            writer.writeheader()
            for row in existing_rows:
                migrated_row = {header: row.get(header, "") for header in SIGNAL_LOG_HEADERS}
                if not migrated_row.get("result"):
                    migrated_row["result"] = "OPEN"
                writer.writerow(migrated_row)
    return SIGNAL_LOG_PATH


def _load_logged_signal_keys(log_path: Path) -> set[tuple[str, str]]:
    """Load existing timestamp-symbol pairs from the signal log."""
    with log_path.open("r", newline="", encoding="utf-8") as log_file:
        reader = DictReader(log_file)
        return {
            (row["timestamp"], row["symbol"])
            for row in reader
            if row.get("timestamp") and row.get("symbol")
        }


def log_signal(
    signal_row: dict[str, object],
    signal_timestamp: str,
    trend_strength: object,
    volume_confirmation: object,
    atr14: object,
    atr14_20_avg: object,
    volatility_ok: object,
    score_total: object,
    score_htf_trend: object,
    score_ltf_trend: object,
    score_pullback: object,
    score_price_holding: object,
    score_volume: object,
    score_volatility: object,
    score_trend_strength: object,
    score_rsi: object,
    score_bonus_confluence: object,
    existing_keys: set[tuple[str, str]] | None = None,
) -> bool:
    """Append a new signal to the CSV log when it has not been recorded yet."""
    log_path = initialize_signal_log()
    if existing_keys is None:
        existing_keys = _load_logged_signal_keys(log_path)

    signal_key = (signal_timestamp, str(signal_row["Symbol"]))
    if signal_key in existing_keys:
        return False

    log_record = {
        "timestamp": signal_timestamp,
        "symbol": signal_row["Symbol"],
        "setup": signal_row["Setup"],
        "score": signal_row["Score"],
        "entry_price": signal_row["Entry Price"],
        "stop_loss": signal_row["Stop Loss"],
        "target_price": signal_row["Target (2R)"],
        "trend_strength": trend_strength,
        "volume_confirmation": volume_confirmation,
        "atr14": atr14,
        "atr14_20_avg": atr14_20_avg,
        "volatility_ok": volatility_ok,
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
        "result": "OPEN",
        "R_multiple": "",
        "max_price_after_entry": "",
        "min_price_after_entry": "",
        "time_to_resolution_minutes": "",
    }
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = DictWriter(log_file, fieldnames=SIGNAL_LOG_HEADERS)
        writer.writerow(log_record)

    existing_keys.add(signal_key)
    return True


def get_symbol_snapshot(base_symbol: str, exchange=None) -> dict[str, object]:
    """Fetch multi-timeframe data and setup details for one symbol."""
    managed_exchange = exchange is None
    if managed_exchange:
        exchange = build_exchange()

    try:
        market_symbol = resolve_market_symbol(exchange, base_symbol, PREFERRED_QUOTES)
        candles_15m = fetch_ohlcv(
            exchange=exchange,
            symbol=market_symbol,
            timeframe=TIMEFRAME,
            limit=OHLCV_LIMIT,
        )
        candles_1h = fetch_ohlcv(
            exchange=exchange,
            symbol=market_symbol,
            timeframe=HIGHER_TIMEFRAME,
            limit=OHLCV_LIMIT,
        )
        enriched_15m = add_indicators(candles_15m)
        enriched_1h = add_indicators(candles_1h)
        setup = evaluate_setup(enriched_15m, enriched_1h)
        row = {
            "Symbol": base_symbol,
            "1H Trend": setup["trend_1h"],
            "15m Trend": setup["trend_15m"],
            "Setup": setup["setup"],
            "Score": setup["score"],
            "score_total": setup["score_total"],
            "score_htf_trend": setup["score_htf_trend"],
            "score_ltf_trend": setup["score_ltf_trend"],
            "score_pullback": setup["score_pullback"],
            "score_price_holding": setup["score_price_holding"],
            "score_volume": setup["score_volume"],
            "score_volatility": setup["score_volatility"],
            "score_trend_strength": setup["score_trend_strength"],
            "score_rsi": setup["score_rsi"],
            "score_bonus_confluence": setup["score_bonus_confluence"],
            "ATR14": setup["atr14"],
            "ATR14_20_avg": setup["atr14_20_avg"],
            "Volatility OK": setup["volatility_ok"],
            "Entry Price": setup["entry_price"],
            "Stop Loss": setup["stop_loss"],
            "Target (2R)": setup["target"],
        }
        return {
            "row": row,
            "market_symbol": market_symbol,
            "data_15m": enriched_15m,
            "data_1h": enriched_1h,
            "setup": setup,
            "signal_timestamp": enriched_15m.iloc[-1]["timestamp"].isoformat(),
        }
    except SymbolUnavailableError as exc:
        return {
            "row": _unavailable_row(base_symbol),
            "market_symbol": None,
            "data_15m": pd.DataFrame(),
            "data_1h": pd.DataFrame(),
            "setup": None,
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover - network/API behavior
        return {
            "row": _error_row(base_symbol, str(exc)),
            "market_symbol": None,
            "data_15m": pd.DataFrame(),
            "data_1h": pd.DataFrame(),
            "setup": None,
            "error": str(exc),
        }
    finally:
        if managed_exchange:
            close_exchange(exchange)


def format_results(results: pd.DataFrame) -> pd.DataFrame:
    """Format numeric output columns for display."""
    formatted = results.copy()
    for column in ["ATR14", "ATR14_20_avg", "Entry Price", "Stop Loss", "Target (2R)"]:
        formatted[column] = formatted[column].map(format_number)
    return formatted


def get_snapshot_metrics(snapshot: dict[str, object]) -> dict[str, object]:
    """Extract dashboard-friendly metrics from a symbol snapshot."""
    price_data = snapshot["data_15m"]
    if price_data.empty:
        return {
            "Trend Strength": pd.NA,
            "Volume Confirmation": "n/a",
            "ATR14": pd.NA,
            "ATR14_20_avg": pd.NA,
            "Volatility OK": False,
        }

    latest = price_data.iloc[-1]
    close_price = latest["close"]
    if pd.isna(close_price) or close_price == 0 or pd.isna(latest["ema20"]) or pd.isna(latest["ema50"]):
        trend_strength = pd.NA
    else:
        trend_strength = abs(latest["ema20"] - latest["ema50"]) / close_price * 100

    setup_details = snapshot.get("setup") or {}
    volume_confirmation = setup_details.get("volume_confirmation", "n/a")

    if pd.isna(latest["atr14"]) or pd.isna(latest["atr14_20_avg"]) or latest["atr14_20_avg"] <= 0:
        volatility_ok = False
    else:
        volatility_ok = bool(latest["atr14"] <= latest["atr14_20_avg"] * 1.2)

    return {
        "Trend Strength": trend_strength,
        "Volume Confirmation": volume_confirmation,
        "ATR14": latest["atr14"],
        "ATR14_20_avg": latest["atr14_20_avg"],
        "Volatility OK": volatility_ok,
    }


def _fetch_post_signal_candles(exchange, market_symbol: str, signal_timestamp: pd.Timestamp) -> pd.DataFrame:
    """Fetch candles after the signal candle for outcome evaluation."""
    since_timestamp = signal_timestamp + pd.Timedelta(TIMEFRAME)
    since_ms = int(since_timestamp.timestamp() * 1000)
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    batch_limit = max(OHLCV_LIMIT, 300)
    batches: list[pd.DataFrame] = []
    latest_seen: pd.Timestamp | None = None

    for _ in range(8):
        if since_ms >= now_ms:
            break

        batch = fetch_ohlcv(
            exchange=exchange,
            symbol=market_symbol,
            timeframe=TIMEFRAME,
            limit=batch_limit,
            since=since_ms,
        )
        if batch.empty:
            break

        batch = batch[batch["timestamp"] > signal_timestamp].copy()
        if latest_seen is not None:
            batch = batch[batch["timestamp"] > latest_seen].copy()
        if batch.empty:
            break

        latest_seen = batch.iloc[-1]["timestamp"]
        since_ms = int(latest_seen.timestamp() * 1000) + 1
        batches.append(batch)

        if len(batch) < batch_limit:
            break

    if not batches:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    return pd.concat(batches, ignore_index=True)


def _evaluate_outcome_for_row(row: pd.Series, future_candles: pd.DataFrame) -> dict[str, object]:
    """Determine whether a logged signal resolved as a win, loss, or remains open."""
    if future_candles.empty:
        return {
            "result": "OPEN",
            "R_multiple": pd.NA,
            "max_price_after_entry": pd.NA,
            "min_price_after_entry": pd.NA,
            "time_to_resolution_minutes": pd.NA,
        }

    entry_price = _coerce_float(row.get("entry_price"))
    stop_loss = _coerce_float(row.get("stop_loss"))
    target_price = _coerce_float(row.get("target_price"))
    signal_timestamp = row.get("timestamp")
    setup = str(row.get("setup", ""))
    if entry_price is None or stop_loss is None or target_price is None or pd.isna(signal_timestamp):
        return {
            "result": "OPEN",
            "R_multiple": pd.NA,
            "max_price_after_entry": pd.NA,
            "min_price_after_entry": pd.NA,
            "time_to_resolution_minutes": pd.NA,
        }

    max_price = future_candles["high"].max()
    min_price = future_candles["low"].min()
    result = "OPEN"
    r_multiple: float | object = pd.NA
    resolution_minutes: float | object = pd.NA
    resolution_index: int | None = None
    is_short = setup == "PULLBACK_SHORT"
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    win_r = reward / risk if risk > 0 else pd.NA

    for candle_index, candle in future_candles.iterrows():
        high_price = _coerce_float(candle["high"])
        low_price = _coerce_float(candle["low"])
        if high_price is None or low_price is None:
            continue

        if is_short:
            target_hit = low_price <= target_price
            stop_hit = high_price >= stop_loss
        else:
            target_hit = high_price >= target_price
            stop_hit = low_price <= stop_loss

        if target_hit and stop_hit:
            result = "LOSS"
            r_multiple = -1.0
            resolution_index = candle_index
            break
        if target_hit:
            result = "WIN"
            r_multiple = win_r
            resolution_index = candle_index
            break
        if stop_hit:
            result = "LOSS"
            r_multiple = -1.0
            resolution_index = candle_index
            break

    if resolution_index is not None:
        resolved_candles = future_candles.iloc[: resolution_index + 1]
        max_price = resolved_candles["high"].max()
        min_price = resolved_candles["low"].min()
        resolved_at = resolved_candles.iloc[-1]["timestamp"]
        resolution_minutes = (resolved_at - signal_timestamp).total_seconds() / 60

    return {
        "result": result,
        "R_multiple": r_multiple,
        "max_price_after_entry": max_price,
        "min_price_after_entry": min_price,
        "time_to_resolution_minutes": resolution_minutes,
    }


def evaluate_signal_outcomes() -> pd.DataFrame:
    """Resolve logged OPEN signals against post-entry candles and persist outcomes."""
    signal_log = _load_signal_log_frame()
    if signal_log.empty:
        return signal_log

    open_mask = signal_log["result"].fillna("OPEN").eq("OPEN")
    if not open_mask.any():
        return signal_log

    resolved_symbols: dict[str, str | None] = {}
    try:
        exchange = build_exchange()
        for row_index, row in signal_log[open_mask].iterrows():
            symbol = str(row.get("symbol", "")).strip()
            signal_timestamp = row.get("timestamp")
            if not symbol or pd.isna(signal_timestamp):
                continue

            if symbol not in resolved_symbols:
                try:
                    resolved_symbols[symbol] = resolve_market_symbol(exchange, symbol, PREFERRED_QUOTES)
                except SymbolUnavailableError:
                    resolved_symbols[symbol] = None

            market_symbol = resolved_symbols[symbol]
            if not market_symbol:
                continue

            future_candles = _fetch_post_signal_candles(exchange, market_symbol, signal_timestamp)
            outcome = _evaluate_outcome_for_row(row, future_candles)
            for field, value in outcome.items():
                signal_log.at[row_index, field] = value
    except Exception:  # pragma: no cover - network/API behavior
        return signal_log
    finally:
        if "exchange" in locals():
            close_exchange(exchange)

    _write_signal_log_frame(signal_log)
    return signal_log


def calculate_expectancy() -> dict[str, float]:
    """Calculate expectancy metrics from resolved logged signals."""
    signal_log = _load_signal_log_frame()
    resolved = signal_log[signal_log["result"].isin(["WIN", "LOSS"])].copy()
    total_trades = int(len(resolved))
    if total_trades == 0:
        return {
            "expectancy_R": 0.0,
            "win_rate": 0.0,
            "avg_win_R": 0.0,
            "avg_loss_R": 0.0,
            "total_trades": 0,
        }

    wins = resolved[resolved["result"] == "WIN"].copy()
    losses = resolved[resolved["result"] == "LOSS"].copy()
    win_rate = len(wins) / total_trades
    loss_rate = len(losses) / total_trades
    avg_win_r = float(wins["R_multiple"].dropna().mean()) if not wins.empty else 0.0
    avg_loss_r = abs(float(losses["R_multiple"].dropna().mean())) if not losses.empty else 0.0
    expectancy_r = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

    return {
        "expectancy_R": expectancy_r,
        "win_rate": win_rate,
        "avg_win_R": avg_win_r,
        "avg_loss_R": avg_loss_r,
        "total_trades": total_trades,
    }


def scan_market(
    base_symbols: list[str] | None = None,
    format_display: bool = True,
    include_dashboard_fields: bool = False,
) -> pd.DataFrame:
    """Fetch data, calculate indicators, and return a setup summary table."""
    symbols_to_scan = BASE_SYMBOLS if base_symbols is None else base_symbols
    exchange = build_exchange()
    logged_keys = _load_logged_signal_keys(initialize_signal_log())
    rows = []

    try:
        for base_symbol in symbols_to_scan:
            snapshot = get_symbol_snapshot(base_symbol, exchange=exchange)
            row = snapshot["row"].copy()
            snapshot_metrics = get_snapshot_metrics(snapshot)
            if row["Setup"] != "NONE" and snapshot.get("signal_timestamp"):
                log_signal(
                    signal_row=row,
                    signal_timestamp=snapshot["signal_timestamp"],
                    trend_strength=snapshot_metrics["Trend Strength"],
                    volume_confirmation=snapshot_metrics["Volume Confirmation"],
                    atr14=snapshot_metrics["ATR14"],
                    atr14_20_avg=snapshot_metrics["ATR14_20_avg"],
                    volatility_ok=snapshot_metrics["Volatility OK"],
                    score_total=row["score_total"],
                    score_htf_trend=row["score_htf_trend"],
                    score_ltf_trend=row["score_ltf_trend"],
                    score_pullback=row["score_pullback"],
                    score_price_holding=row["score_price_holding"],
                    score_volume=row["score_volume"],
                    score_volatility=row["score_volatility"],
                    score_trend_strength=row["score_trend_strength"],
                    score_rsi=row["score_rsi"],
                    score_bonus_confluence=row["score_bonus_confluence"],
                    existing_keys=logged_keys,
                )
            if include_dashboard_fields:
                row.update(snapshot_metrics)
            rows.append(row)
    finally:
        close_exchange(exchange)

    columns = RESULT_COLUMNS + DASHBOARD_EXTRA_COLUMNS if include_dashboard_fields else RESULT_COLUMNS
    results = pd.DataFrame(rows, columns=columns)
    if format_display:
        return format_results(results)
    return results


def main() -> None:
    """Run the scanner continuously until the user stops it."""
    try:
        while True:
   	    results = scan_market()
            results.to_csv("latest_scan.csv", index=False)
            evaluate_signal_outcomes()
            clear_console()
            print(
                f"Crypto Scanner | Exchange: {EXCHANGE_ID} | Timeframes: {TIMEFRAME} / {HIGHER_TIMEFRAME}"
            )
            print(f"Scan Time: {current_timestamp()}")
            print()
            print(results.to_string(index=False))
            print()
            print("Next scan in 5 minutes. Press Ctrl+C to stop.")
            time.sleep(SCAN_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nScanner stopped by user.")


if __name__ == "__main__":
    main()
