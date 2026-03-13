"""Streamlit web app for the crypto scanner."""

from __future__ import annotations

from csv import DictWriter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from config import LIQUID_UNIVERSE
from main import (
    SIGNAL_LOG_PATH,
    calculate_expectancy,
    current_timestamp,
    evaluate_signal_outcomes,
    get_symbol_snapshot,
    scan_market,
)

AUTO_REFRESH_MS = 300_000
ACTIVE_TRADES_PATH = Path(__file__).resolve().parent / "active_trades.csv"
ACTIVE_TRADES_HEADERS = [
    "timestamp",
    "symbol",
    "signal_entry",
    "actual_entry",
    "stop_loss",
    "target_price",
    "score",
    "setup",
]
WATCH_SCORE_BUFFER = 10
RANKED_TABLE_COLUMNS = [
    "Symbol",
    "Score Total",
    "Status",
    "Qualification",
    "Setup",
    "1H Trend",
    "15m Trend",
    "Trend Strength",
    "Volume Confirmation",
    "Volatility OK",
    "Entry Price",
    "Stop Loss",
    "Target Price",
]
DEFAULT_UI_STATE = {
    "tracked_coins": LIQUID_UNIVERSE,
    "min_score": 75,
    "active_only": False,
    "account_size": 10000.0,
    "risk_percent": 0.5,
    "max_open_trades": 2,
    "chart_coin": LIQUID_UNIVERSE[0],
}


def _parse_bool(value: object, default: bool) -> bool:
    """Parse a boolean-like query param value."""
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: object, default: float) -> float:
    """Parse a float-like query param value."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: object, default: int) -> int:
    """Parse an int-like query param value."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _query_param_value(params: object, key: str) -> object:
    """Read a single query param value from Streamlit query params."""
    value = params.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def style_signal_cell(value: object) -> str:
    """Return a style for trend and setup cells."""
    colors = {
        "LONG": "background-color: #e7f5ed; color: #166534; font-weight: 600;",
        "SHORT": "background-color: #fdf1ef; color: #9f1239; font-weight: 600;",
        "PULLBACK_LONG": "background-color: #dff1e7; color: #166534; font-weight: 700;",
        "PULLBACK_SHORT": "background-color: #f9e4df; color: #9f1239; font-weight: 700;",
        "NONE": "background-color: #f3f4f6; color: #6b7280;",
    }
    return colors.get(str(value), "")


def style_status_cell(value: object) -> str:
    """Return a style for ranked board status labels."""
    colors = {
        "Actionable": "background-color: #dcfce7; color: #166534; font-weight: 800;",
        "Watch": "background-color: #fef3c7; color: #9a6700; font-weight: 700;",
        "Below Threshold": "background-color: #f3f4f6; color: #6b7280; font-weight: 600;",
    }
    return colors.get(str(value), "")


def style_score_cell(value: object) -> str:
    """Highlight strong scores in the results table."""
    if pd.isna(value):
        return ""
    if float(value) >= 90:
        return "background-color: #dff1e7; color: #166534; font-weight: 800;"
    if float(value) >= 80:
        return "background-color: #f6ead2; color: #9a6700; font-weight: 700;"
    return "color: #4b5563;"


def style_result_row(row: pd.Series) -> list[str]:
    """Apply row-level styling to emphasize actionable setups."""
    if row.get("Status") == "Actionable":
        return ["background-color: #f7fcf8;"] * len(row)
    if row.get("Status") == "Watch":
        return ["background-color: #fffaf0;"] * len(row)
    return ["color: #6b7280;"] * len(row)


def initialize_ui_state() -> None:
    """Seed session state for persistent dashboard selections."""
    query_params = st.query_params
    tracked_from_query = query_params.get_all("tracked_coins")
    tracked_coins = [coin for coin in tracked_from_query if coin in LIQUID_UNIVERSE]

    initial_values = {
        "tracked_coins": tracked_coins or DEFAULT_UI_STATE["tracked_coins"].copy(),
        "min_score": _parse_int(_query_param_value(query_params, "min_score"), DEFAULT_UI_STATE["min_score"]),
        "active_only": _parse_bool(
            _query_param_value(query_params, "active_only"),
            DEFAULT_UI_STATE["active_only"],
        ),
        "account_size": _parse_float(
            _query_param_value(query_params, "account_size"),
            DEFAULT_UI_STATE["account_size"],
        ),
        "risk_percent": _parse_float(
            _query_param_value(query_params, "risk_percent"),
            DEFAULT_UI_STATE["risk_percent"],
        ),
        "max_open_trades": _parse_int(
            _query_param_value(query_params, "max_open_trades"),
            DEFAULT_UI_STATE["max_open_trades"],
        ),
        "chart_coin": _query_param_value(query_params, "chart_coin") or DEFAULT_UI_STATE["chart_coin"],
    }

    for key, value in initial_values.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def persist_ui_state() -> None:
    """Persist dashboard selections into query params so refreshes keep them."""
    st.query_params.clear()
    st.query_params["tracked_coins"] = st.session_state.get("tracked_coins", [])
    st.query_params["min_score"] = str(st.session_state.get("min_score", DEFAULT_UI_STATE["min_score"]))
    st.query_params["active_only"] = str(st.session_state.get("active_only", DEFAULT_UI_STATE["active_only"]))
    st.query_params["account_size"] = str(st.session_state.get("account_size", DEFAULT_UI_STATE["account_size"]))
    st.query_params["risk_percent"] = str(st.session_state.get("risk_percent", DEFAULT_UI_STATE["risk_percent"]))
    st.query_params["max_open_trades"] = str(
        st.session_state.get("max_open_trades", DEFAULT_UI_STATE["max_open_trades"])
    )
    chart_coin = st.session_state.get("chart_coin")
    if chart_coin:
        st.query_params["chart_coin"] = str(chart_coin)


def inject_dashboard_styles() -> None:
    """Inject dashboard-specific CSS for hierarchy and readability."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
        }
        .dashboard-title {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .dashboard-subtitle {
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 1.35rem;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .section-subtitle {
            font-size: 0.88rem;
            color: #64748b;
            margin-bottom: 0.8rem;
        }
        .kpi-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            min-height: 96px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .kpi-label {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
        }
        .kpi-note {
            font-size: 0.82rem;
            color: #94a3b8;
            margin-top: 0.3rem;
        }
        div[data-testid="stDataFrame"] div[role="table"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #e5e7eb;
        }
        div[data-testid="stDataFrame"] thead tr th {
            background: #f8fafc !important;
            color: #334155 !important;
            font-weight: 700 !important;
            border-bottom: 1px solid #e2e8f0 !important;
            padding-top: 0.75rem !important;
            padding-bottom: 0.75rem !important;
        }
        div[data-testid="stDataFrame"] tbody tr td {
            padding-top: 0.7rem !important;
            padding-bottom: 0.7rem !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str | None = None) -> None:
    """Render a consistent section heading."""
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def format_display_value(value: object, decimals: int = 2, suffix: str = "") -> str:
    """Format a dashboard display value."""
    if pd.isna(value):
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}{suffix}"
    return str(value)


def add_dashboard_metrics(results: pd.DataFrame, account_size: float, risk_percent: float) -> pd.DataFrame:
    """Add portfolio and decision-support metrics to the scan results."""
    enriched = results.copy()
    risk_dollars = account_size * (risk_percent / 100)
    enriched["Risk Per Trade ($)"] = risk_dollars

    entry = pd.to_numeric(enriched["Entry Price"], errors="coerce")
    stop = pd.to_numeric(enriched["Stop Loss"], errors="coerce")
    target = pd.to_numeric(enriched["Target (2R)"], errors="coerce")
    unit_risk = (entry - stop).abs()
    reward = (target - entry).abs()

    valid_trade = entry.notna() & stop.notna() & target.notna() & (unit_risk > 0)
    enriched["Position Size"] = pd.NA
    enriched["R Multiple"] = pd.NA
    enriched.loc[valid_trade, "Position Size"] = risk_dollars / unit_risk[valid_trade]
    enriched.loc[valid_trade, "R Multiple"] = reward[valid_trade] / unit_risk[valid_trade]
    return enriched


def build_ranked_results(results: pd.DataFrame, timestamp: str, min_score: int) -> pd.DataFrame:
    """Build the full ranked market board for all tracked coins."""
    if results.empty:
        return pd.DataFrame(columns=RANKED_TABLE_COLUMNS)

    ranked = results.copy()
    qualification_notes = []
    for _, row in ranked.iterrows():
        snapshot = get_chart_snapshot(str(row["Symbol"]), timestamp, force_refresh=False)
        failures = (snapshot.get("setup") or {}).get("qualification_failures", [])
        qualification_notes.append(", ".join(failures) if failures else "n/a")

    ranked["Qualification"] = qualification_notes
    ranked["Score Total"] = pd.to_numeric(ranked["Score"], errors="coerce")
    ranked["Target Price"] = ranked["Target (2R)"]

    actionable_mask = (ranked["Setup"] != "NONE") & (ranked["Score Total"] >= min_score)
    watch_floor = max(min_score - WATCH_SCORE_BUFFER, 0)
    watch_mask = (~actionable_mask) & (ranked["Score Total"] >= watch_floor)
    ranked["Status"] = "Below Threshold"
    ranked.loc[watch_mask, "Status"] = "Watch"
    ranked.loc[actionable_mask, "Status"] = "Actionable"

    ranked = ranked.sort_values(["Score Total", "Symbol"], ascending=[False, True])
    return ranked


def get_results(
    tracked_coins: list[str],
    account_size: float,
    risk_percent: float,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Read results from session state or fetch a fresh scan for the tracked universe."""
    tracked_key = tuple(tracked_coins)
    cache_miss = (
        force_refresh
        or "scan_results_raw" not in st.session_state
        or "scan_timestamp" not in st.session_state
        or st.session_state.get("tracked_key") != tracked_key
    )
    if cache_miss:
        if tracked_coins:
            raw_results = scan_market(
                base_symbols=tracked_coins,
                format_display=False,
                include_dashboard_fields=True,
            ).copy()
        else:
            raw_results = pd.DataFrame(
                columns=[
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
                    "Trend Strength",
                    "Volume Confirmation",
                ]
            )
        timestamp = current_timestamp()
        st.session_state["scan_results_raw"] = raw_results
        st.session_state["scan_timestamp"] = timestamp
        st.session_state["tracked_key"] = tracked_key

    raw_results = st.session_state["scan_results_raw"].copy()
    timestamp = st.session_state["scan_timestamp"]
    results = add_dashboard_metrics(raw_results, account_size, risk_percent)
    results["Last Updated"] = timestamp
    return results, timestamp


def get_chart_snapshot(base_symbol: str, refresh_key: str, force_refresh: bool = False) -> dict[str, object]:
    """Read chart data from session state or fetch a fresh symbol snapshot."""
    if "chart_snapshots" not in st.session_state:
        st.session_state["chart_snapshots"] = {}

    cache_key = f"{refresh_key}:{base_symbol}"
    if force_refresh or cache_key not in st.session_state["chart_snapshots"]:
        st.session_state["chart_snapshots"][cache_key] = get_symbol_snapshot(base_symbol)

    return st.session_state["chart_snapshots"][cache_key]


def filter_results(
    results: pd.DataFrame,
    active_only: bool,
    min_score: int,
) -> pd.DataFrame:
    """Apply sidebar filters to the scan results."""
    filtered = results[results["Score"] >= min_score].copy()
    if active_only:
        filtered = filtered[filtered["Setup"] != "NONE"]
    filtered["_active_rank"] = (filtered["Setup"] != "NONE").astype(int)
    filtered = filtered.sort_values(["_active_rank", "Score", "Symbol"], ascending=[False, False, True])
    filtered = filtered.drop(columns="_active_rank")
    return filtered


def highest_score_summary(results: pd.DataFrame, min_score: int) -> str:
    """Return a concise summary of the strongest available setup."""
    if results.empty:
        return "None"

    source = results[(results["Setup"] != "NONE") & (results["Score"] >= min_score)].copy()
    if source.empty:
        return "None"

    source = source.sort_values(["Score", "Symbol"], ascending=[False, True])
    best = source.iloc[0]
    return f"{best['Symbol']} {int(best['Score'])}"


def render_kpis(results: pd.DataFrame, timestamp: str, min_score: int) -> None:
    """Render the dashboard KPI cards."""
    active_count = int((results["Setup"] != "NONE").sum()) if not results.empty else 0
    cards = st.columns(5, gap="small")
    card_data = [
        ("Last Scan Time", timestamp, "Latest completed market pass"),
        ("Coins Scanned", str(len(results)), "Current tracked universe"),
        ("Active Setups", str(active_count), "Rows with actionable setups"),
        ("Top Setup by Score", highest_score_summary(results, min_score), "Best qualified signal"),
        ("Minimum Score", str(min_score), "Current ranking threshold"),
    ]
    for column, (label, value, note) in zip(cards, card_data):
        with column:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def initialize_active_trades() -> Path:
    """Create the active trades CSV with headers if it does not exist."""
    if not ACTIVE_TRADES_PATH.exists():
        with ACTIVE_TRADES_PATH.open("w", newline="", encoding="utf-8") as trades_file:
            writer = DictWriter(trades_file, fieldnames=ACTIVE_TRADES_HEADERS)
            writer.writeheader()
    return ACTIVE_TRADES_PATH


def load_active_trades() -> pd.DataFrame:
    """Load tracked trades from disk."""
    if not ACTIVE_TRADES_PATH.exists():
        return pd.DataFrame(columns=ACTIVE_TRADES_HEADERS)

    trades = pd.read_csv(ACTIVE_TRADES_PATH)
    if trades.empty:
        return pd.DataFrame(columns=ACTIVE_TRADES_HEADERS)

    if "signal_entry" not in trades.columns and "entry_price" in trades.columns:
        trades["signal_entry"] = trades["entry_price"]
    if "actual_entry" not in trades.columns:
        trades["actual_entry"] = trades["signal_entry"]

    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    for column in ["signal_entry", "actual_entry", "stop_loss", "target_price", "score"]:
        trades[column] = pd.to_numeric(trades[column], errors="coerce")
    return trades[ACTIVE_TRADES_HEADERS]


def save_active_trade(row: pd.Series, actual_entry: float) -> bool:
    """Append a trade to the active trades CSV if it is not already tracked."""
    initialize_active_trades()
    existing_trades = load_active_trades()

    duplicate_mask = (
        (existing_trades["symbol"] == row["Symbol"])
        & (existing_trades["actual_entry"] == actual_entry)
        & (existing_trades["stop_loss"] == row["Stop Loss"])
        & (existing_trades["target_price"] == row["Target (2R)"])
        & (existing_trades["setup"] == row["Setup"])
    )
    if not existing_trades.empty and duplicate_mask.any():
        return False

    trade_record = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "symbol": row["Symbol"],
        "signal_entry": row["Entry Price"],
        "actual_entry": actual_entry,
        "stop_loss": row["Stop Loss"],
        "target_price": row["Target (2R)"],
        "score": row["Score"],
        "setup": row["Setup"],
    }
    with ACTIVE_TRADES_PATH.open("a", newline="", encoding="utf-8") as trades_file:
        writer = DictWriter(trades_file, fieldnames=ACTIVE_TRADES_HEADERS)
        writer.writerow(trade_record)
    return True


def _status_color(status: str) -> str:
    """Return a row style for a live trade status."""
    colors = {
        "ACTIVE": "background-color: #dbeafe; color: #1d4ed8;",
        "TARGET HIT": "background-color: #dcfce7; color: #166534;",
        "STOP HIT": "background-color: #fee2e2; color: #991b1b;",
    }
    return colors.get(status, "")


def format_time_open(timestamp: pd.Timestamp) -> str:
    """Format the elapsed time since the trade was tracked."""
    if pd.isna(timestamp):
        return "n/a"

    delta = pd.Timestamp.utcnow() - timestamp
    total_minutes = int(delta.total_seconds() // 60)
    if total_minutes < 60:
        return f"{total_minutes}m"

    hours, minutes = divmod(total_minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"

    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def enrich_live_trades(trades: pd.DataFrame, refresh_key: str, force_refresh: bool) -> pd.DataFrame:
    """Add current price, status, and R metrics to tracked trades."""
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Status",
                "Entry",
                "Current Price",
                "Stop",
                "Target",
                "Current R",
                "Time Open",
            ]
        )

    enriched_rows = []
    for _, trade in trades.iterrows():
        snapshot = get_chart_snapshot(str(trade["symbol"]), refresh_key, force_refresh=force_refresh)
        current_price = pd.NA
        if not snapshot["data_15m"].empty:
            current_price = snapshot["data_15m"].iloc[-1]["close"]

        status = "ACTIVE"
        if pd.notna(current_price):
            if str(trade["setup"]) == "PULLBACK_SHORT":
                if current_price <= trade["target_price"]:
                    status = "TARGET HIT"
                elif current_price >= trade["stop_loss"]:
                    status = "STOP HIT"
            else:
                if current_price >= trade["target_price"]:
                    status = "TARGET HIT"
                elif current_price <= trade["stop_loss"]:
                    status = "STOP HIT"

        risk_unit = trade["actual_entry"] - trade["stop_loss"]
        current_r = pd.NA
        if pd.notna(current_price) and pd.notna(risk_unit) and risk_unit != 0:
            current_r = (current_price - trade["actual_entry"]) / risk_unit

        enriched_rows.append(
            {
                "Symbol": trade["symbol"],
                "Status": status,
                "Entry": trade["actual_entry"],
                "Current Price": current_price,
                "Stop": trade["stop_loss"],
                "Target": trade["target_price"],
                "Current R": current_r,
                "Time Open": format_time_open(trade["timestamp"]),
            }
        )

    return pd.DataFrame(enriched_rows)


def render_track_trade_form(row: pd.Series) -> None:
    """Render a confirmation form to capture the user's actual entry price."""
    form_key = f"track-form-{row['Symbol']}-{row['Setup']}"
    st.markdown("**Track This Trade**")
    st.caption(
        f"{row['Symbol']} | Signal Entry: {format_display_value(row['Entry Price'], 4)} | "
        f"Stop: {format_display_value(row['Stop Loss'], 4)} | "
        f"Target: {format_display_value(row['Target (2R)'], 4)} | "
        f"Score: {format_display_value(row['Score'], 0)}"
    )
    with st.form(form_key, clear_on_submit=False):
        actual_entry = st.number_input(
            "Actual Entry Price",
            min_value=0.0,
            value=float(row["Entry Price"]),
            step=max(float(row["Entry Price"]) * 0.0001, 0.0001),
            format="%.4f",
            key=f"actual-entry-{row['Symbol']}-{row['Setup']}",
        )
        confirm = st.form_submit_button("Confirm Trade")
        cancel = st.form_submit_button("Cancel")

    if confirm:
        saved = save_active_trade(row, actual_entry)
        st.session_state.pop("trade_form_key", None)
        if saved:
            st.success("Trade added to Live Trades.")
        else:
            st.info("Trade is already being tracked.")
        st.rerun()

    if cancel:
        st.session_state.pop("trade_form_key", None)
        st.rerun()


def render_live_trades(timestamp: str, refresh_requested: bool) -> None:
    """Render the live trades monitoring panel."""
    render_section_header("Live Trades", "Monitor tracked positions against live market movement.")
    trades = load_active_trades()
    if trades.empty:
        st.info("No trades are being monitored yet.")
        return

    live_trades = enrich_live_trades(trades, timestamp, refresh_requested)
    styled_trades = (
        live_trades.style.apply(
            lambda row: [_status_color(str(row["Status"]))] * len(row),
            axis=1,
        ).format(
            {
                "Entry": "{:,.4f}",
                "Current Price": "{:,.4f}",
                "Stop": "{:,.4f}",
                "Target": "{:,.4f}",
                "Current R": "{:,.2f}",
            },
            na_rep="n/a",
        )
    )
    st.dataframe(styled_trades, use_container_width=True, hide_index=True)


def render_top_setups(results: pd.DataFrame, min_score: int, max_open_trades: int) -> None:
    """Render the top three setups ranked by score."""
    render_section_header("Top Setups", "Highest-ranked active setups above the current score threshold.")

    active_results = results[results["Status"] == "Actionable"].copy()
    ranked = active_results.sort_values(["Score Total", "Symbol"], ascending=[False, True]).head(3)
    if ranked.empty:
        st.info("No active setups at the current threshold.")
        return

    if len(active_results) > max_open_trades:
        st.caption(f"{len(active_results)} setups qualify above your threshold. Max open trades is set to {max_open_trades}.")

    cards = st.columns(3)
    for column, (_, row) in zip(cards, ranked.iterrows()):
        trade_form_key = f"{row['Symbol']}-{row['Setup']}-{int(row['Score'])}"
        with column:
            st.markdown(f"**{row['Symbol']}**")
            st.caption(f"{row['Setup']} | {row['1H Trend']} / {row['15m Trend']}")
            st.metric("Score", int(row["Score Total"]))
            st.write(
                f"Entry: {row['Entry Price']} | Stop: {row['Stop Loss']} | Target: {row['Target (2R)']}"
            )
            if st.button("Track Trade", key=f"track-trade-{trade_form_key}"):
                st.session_state["trade_form_key"] = trade_form_key

            if st.session_state.get("trade_form_key") == trade_form_key:
                render_track_trade_form(row)


def render_ranked_table(results: pd.DataFrame) -> None:
    """Render the full ranked market board."""
    display_results = results[RANKED_TABLE_COLUMNS].copy()
    styled_results = (
        display_results.style.apply(style_result_row, axis=1)
        .map(style_status_cell, subset=["Status"])
        .map(style_signal_cell, subset=["1H Trend", "15m Trend", "Setup"])
        .map(style_score_cell, subset=["Score Total"])
        .format(
            {
                "Score Total": "{:.0f}",
                "Entry Price": "{:,.4f}",
                "Stop Loss": "{:,.4f}",
                "Target Price": "{:,.4f}",
                "Trend Strength": "{:,.2f}",
            },
            na_rep="n/a",
        )
    )
    st.dataframe(styled_results, use_container_width=True, hide_index=True)


def load_signal_log() -> pd.DataFrame:
    """Load the signal log from disk when available."""
    if not SIGNAL_LOG_PATH.exists():
        return pd.DataFrame()

    signal_log = pd.read_csv(SIGNAL_LOG_PATH)
    if signal_log.empty:
        return signal_log

    signal_log["timestamp"] = pd.to_datetime(signal_log["timestamp"], utc=True, errors="coerce")
    signal_log["score"] = pd.to_numeric(signal_log["score"], errors="coerce")
    for column in [
        "R_multiple",
        "max_price_after_entry",
        "min_price_after_entry",
        "time_to_resolution_minutes",
    ]:
        if column in signal_log.columns:
            signal_log[column] = pd.to_numeric(signal_log[column], errors="coerce")
    return signal_log


def render_strategy_performance() -> None:
    """Render expectancy and resolution metrics from the signal log."""
    render_section_header("Strategy Performance", "Hypothetical outcome tracking for logged signals.")
    signal_log = load_signal_log()
    if signal_log.empty:
        st.info("No signals logged yet")
        return

    performance = calculate_expectancy()
    cards = st.columns(5, gap="small")
    metrics = [
        ("Expectancy (R)", f"{performance['expectancy_R']:+.2f}R"),
        ("Win Rate", f"{performance['win_rate']:.0%}"),
        ("Average Win (R)", f"{performance['avg_win_R']:.2f}R"),
        ("Average Loss (R)", f"-{performance['avg_loss_R']:.2f}R"),
        ("Signals Evaluated", str(performance["total_trades"])),
    ]
    for column, (label, value) in zip(cards, metrics):
        column.metric(label, value)

    if performance["total_trades"] == 0:
        st.caption("No logged signals have fully resolved yet.")


def build_score_distribution(signal_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate logged signals into score buckets."""
    buckets = [
        ("90+", signal_log["score"] >= 90),
        ("80-89", signal_log["score"].between(80, 89.9999, inclusive="both")),
        ("70-79", signal_log["score"].between(70, 79.9999, inclusive="both")),
        ("60-69", signal_log["score"].between(60, 69.9999, inclusive="both")),
    ]
    rows = [{"Bucket": label, "Signals": int(mask.fillna(False).sum())} for label, mask in buckets]
    return pd.DataFrame(rows)


def render_signal_analytics() -> None:
    """Render analytics for logged signals."""
    render_section_header("Signal Analytics", "Logged-signal distribution for ongoing strategy evaluation.")
    signal_log = load_signal_log()
    if signal_log.empty:
        st.info("No signals logged yet")
        return

    now_utc = pd.Timestamp.utcnow()
    last_24h_count = int((signal_log["timestamp"] >= now_utc - pd.Timedelta(hours=24)).fillna(False).sum())
    total_signals = len(signal_log)
    score_distribution = build_score_distribution(signal_log)
    top_symbols = (
        signal_log["symbol"]
        .value_counts()
        .head(5)
        .rename_axis("Symbol")
        .reset_index(name="Signals")
    )

    cards = st.columns(4)
    cards[0].metric("Total Signals", total_signals)
    cards[1].metric("Last 24 Hours", last_24h_count)
    cards[2].metric("Top Symbol", top_symbols.iloc[0]["Symbol"] if not top_symbols.empty else "n/a")
    cards[3].metric("Unique Coins", signal_log["symbol"].nunique())

    summary_column, chart_column = st.columns([1.1, 1.9])
    with summary_column:
        st.markdown("**Score Buckets**")
        st.dataframe(score_distribution, use_container_width=True, hide_index=True)
        st.markdown("**Top Symbols**")
        st.dataframe(top_symbols, use_container_width=True, hide_index=True)

    with chart_column:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=score_distribution["Bucket"],
                    y=score_distribution["Signals"],
                    marker_color=["#166534", "#65a30d", "#d97706", "#6b7280"],
                )
            ]
        )
        figure.update_layout(
            title="Score Distribution",
            template="plotly_white",
            height=320,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
            xaxis_title="Score Bucket",
            yaxis_title="Signals",
        )
        st.plotly_chart(figure, use_container_width=True)


def build_chart(snapshot: dict[str, object], selected_coin: str) -> go.Figure:
    """Build a Plotly candlestick chart with EMA and setup overlays."""
    price_data = snapshot["data_15m"].copy()
    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=price_data["timestamp"],
            open=price_data["open"],
            high=price_data["high"],
            low=price_data["low"],
            close=price_data["close"],
            name="Price",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=price_data["timestamp"],
            y=price_data["ema20"],
            mode="lines",
            name="EMA20",
            line={"color": "#2563eb", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=price_data["timestamp"],
            y=price_data["ema50"],
            mode="lines",
            name="EMA50",
            line={"color": "#f59e0b", "width": 2},
        )
    )

    row = snapshot["row"]
    if row["Setup"] != "NONE":
        figure.add_hline(y=row["Entry Price"], line_dash="dash", line_color="#2563eb", annotation_text="Entry")
        figure.add_hline(y=row["Stop Loss"], line_dash="dot", line_color="#dc2626", annotation_text="Stop")
        figure.add_hline(y=row["Target (2R)"], line_dash="dot", line_color="#16a34a", annotation_text="Target")

    figure.update_layout(
        title=f"{selected_coin} 15m Chart",
        template="plotly_white",
        height=520,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return figure


def render_chart_section(selected_coin: str, timestamp: str, refresh_requested: bool) -> None:
    """Render the interactive chart section for one selected coin."""
    render_section_header("Interactive Chart", "Recent 15-minute price action with EMA and trade overlays.")
    snapshot = get_chart_snapshot(selected_coin, timestamp, force_refresh=refresh_requested)
    if snapshot["data_15m"].empty:
        error_text = snapshot.get("error", "Chart data is unavailable for this symbol.")
        st.warning(error_text)
        return

    st.plotly_chart(build_chart(snapshot, selected_coin), use_container_width=True)


def render_setup_summary(selected_coin: str, results: pd.DataFrame) -> None:
    """Render the setup summary panel for the selected chart coin."""
    matching_rows = results[results["Symbol"] == selected_coin]
    if matching_rows.empty:
        st.info("No summary available for the selected coin.")
        return

    row = matching_rows.iloc[0]
    snapshot = get_chart_snapshot(selected_coin, row["Last Updated"], force_refresh=False)
    setup_details = snapshot.get("setup") or {}
    score_breakdown = setup_details.get("score_breakdown", {})
    qualification_failures = setup_details.get("qualification_failures", [])
    summary_fields = [
        ("Symbol", row["Symbol"]),
        ("1H Trend", row["1H Trend"]),
        ("15m Trend", row["15m Trend"]),
        ("Setup", row["Setup"]),
        ("Score", format_display_value(row["Score"], 0)),
        ("ATR14", format_display_value(row["ATR14"], 4)),
        ("ATR14 20-period average", format_display_value(row["ATR14_20_avg"], 4)),
        ("Volatility OK", row["Volatility OK"]),
        ("Entry", format_display_value(row["Entry Price"], 4)),
        ("Stop", format_display_value(row["Stop Loss"], 4)),
        ("Target", format_display_value(row["Target (2R)"], 4)),
        ("Risk Per Trade ($)", format_display_value(row["Risk Per Trade ($)"], 2)),
        ("Position Size", format_display_value(row["Position Size"], 4)),
        ("Trend Strength", format_display_value(row["Trend Strength"], 2, "%")),
        ("Volume Confirmation", row["Volume Confirmation"]),
    ]
    for label, value in summary_fields:
        st.markdown(f"**{label}:** {value}")

    if qualification_failures:
        st.markdown("**Setup Qualification**")
        for failure in qualification_failures:
            st.markdown(f"- {failure}")

    st.markdown("**Score Breakdown**")
    if not score_breakdown:
        st.markdown(f"- `Total`: {format_display_value(row['Score'], 0)}")
        return

    st.markdown(f"- `Total`: {format_display_value(score_breakdown.get('total', row['Score']), 0)}")
    breakdown_sections = [
        ("Market Structure", score_breakdown.get("market_structure", {})),
        ("Entry Quality", score_breakdown.get("entry_quality", {})),
        ("Market Conditions", score_breakdown.get("market_conditions", {})),
        ("Bonus Confluence", score_breakdown.get("bonus_confluence", {})),
    ]
    for section_title, section_values in breakdown_sections:
        if not section_values:
            continue
        st.markdown(f"**{section_title}**")
        for label, value in section_values.items():
            pretty_label = label.replace("_", " ").title()
            st.markdown(f"- `{pretty_label}`: {format_display_value(value, 0)}")


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title="Crypto Scanner", layout="wide")
    initialize_ui_state()
    inject_dashboard_styles()
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {AUTO_REFRESH_MS});
        </script>
        """,
        height=0,
    )

    st.markdown("<div class='dashboard-title'>Crypto Scanner</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dashboard-subtitle'>Coinbase multi-timeframe pullback scanner for liquid crypto markets on the 15-minute and 1-hour timeframes.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Filters")
        tracked_coins = st.multiselect(
            "Tracked Coins",
            options=LIQUID_UNIVERSE,
            key="tracked_coins",
        )
        min_score = st.slider("Minimum score", min_value=0, max_value=100, step=1, key="min_score")
        active_only = st.checkbox("Show only active setups", key="active_only")
        account_size = st.number_input("Account Size ($)", min_value=100.0, step=500.0, key="account_size")
        risk_percent = st.number_input(
            "Risk Per Trade (%)",
            min_value=0.1,
            max_value=10.0,
            step=0.1,
            key="risk_percent",
        )
        max_open_trades = st.number_input(
            "Max Open Trades",
            min_value=1,
            max_value=20,
            step=1,
            key="max_open_trades",
        )
        refresh_requested = st.button("Refresh Data", use_container_width=True)
        results, timestamp = get_results(tracked_coins, account_size, risk_percent, force_refresh=refresh_requested)

        chart_options = sorted(tracked_coins)
        if chart_options and st.session_state["chart_coin"] not in chart_options:
            st.session_state["chart_coin"] = chart_options[0]
        if not chart_options:
            st.session_state["chart_coin"] = None
        chart_coin = (
            st.selectbox("Chart Coin", options=chart_options, key="chart_coin")
            if chart_options
            else None
        )
        persist_ui_state()

    evaluate_signal_outcomes()
    ranked_results = build_ranked_results(results, timestamp, min_score)

    render_kpis(results, timestamp, min_score)
    st.caption(f"Last scan: {timestamp}")
    with st.container(border=True):
        render_top_setups(ranked_results, min_score, int(max_open_trades))
    with st.container(border=True):
        render_section_header("All Coins Ranked", "Live ranked scanner board across the full tracked universe.")
        if ranked_results.empty:
            st.info("No tracked coins are currently selected.")
        else:
            render_ranked_table(ranked_results)
    with st.container(border=True):
        render_live_trades(timestamp, refresh_requested)
    with st.container(border=True):
        render_signal_analytics()
    with st.container(border=True):
        render_strategy_performance()
    if chart_coin:
        chart_column, summary_column = st.columns([3, 1.2])
        with chart_column:
            with st.container(border=True):
                render_chart_section(chart_coin, timestamp, refresh_requested)
        with summary_column:
            with st.container(border=True):
                render_section_header("Setup Summary", "Focused detail for the selected chart symbol.")
                render_setup_summary(chart_coin, results)
    else:
        st.info("Select at least one tracked coin to view chart details.")


if __name__ == "__main__":
    main()
