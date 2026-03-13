"""Project configuration for the crypto scanner."""

EXCHANGE_ID = "coinbase"
TIMEFRAME = "15m"
HIGHER_TIMEFRAME = "1h"
OHLCV_LIMIT = 150
SCAN_INTERVAL_SECONDS = 300

BASE_SYMBOLS = ["BTC", "ETH", "SOL", "LINK", "AVAX", "SUI"]
LIQUID_UNIVERSE = [
    "BTC",
    "ETH",
    "SOL",
    "AVAX",
    "LINK",
    "TON",
    "MATIC",
    "ATOM",
    "DOGE",
    "NEAR",
    "APT",
    "INJ",
    "FIL",
    "LTC",
    "SUI",
]
PREFERRED_QUOTES = ["USD", "USDC", "USDT"]
