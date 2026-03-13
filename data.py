"""Market data access helpers."""

from __future__ import annotations

from typing import Optional

import ccxt
import pandas as pd


class SymbolUnavailableError(Exception):
    """Raised when a requested base asset is not available on the exchange."""


def create_exchange(exchange_id: str):
    """Create a ccxt exchange instance for public market data."""
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def resolve_market_symbol(exchange, base_symbol: str, preferred_quotes: list[str]) -> str:
    """Resolve the best matching exchange symbol for a base asset."""
    markets = exchange.markets or exchange.load_markets()

    for quote_symbol in preferred_quotes:
        unified_symbol = f"{base_symbol}/{quote_symbol}"
        if unified_symbol in markets:
            return unified_symbol

    supported_quotes = sorted(
        {
            market["quote"]
            for market in markets.values()
            if market.get("base") == base_symbol and market.get("quote")
        }
    )
    supported_text = ", ".join(supported_quotes) if supported_quotes else "none"
    raise SymbolUnavailableError(
        f"{base_symbol} is not available on {exchange.id} with preferred quotes "
        f"({', '.join(preferred_quotes)}). Supported quotes: {supported_text}."
    )


def fetch_ohlcv(
    exchange,
    symbol: str,
    timeframe: str,
    limit: int = 150,
    since: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles and return them as a pandas DataFrame."""
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
