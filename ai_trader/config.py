from dataclasses import dataclass


@dataclass(frozen=True)
class TradingConfig:
    symbol: str = "XAUUSD"
    timeframe: str = "H1"
    months: int = 12
    horizon: int = 3
    target_return: float = 0.0015


DEFAULT_CONFIG = TradingConfig()

TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}
