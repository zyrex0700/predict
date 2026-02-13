from __future__ import annotations

from dataclasses import dataclass

try:
    import MetaTrader5 as mt5
except Exception:  # noqa: BLE001
    mt5 = None


ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_IOC = 1


@dataclass
class TradeConfig:
    lot: float = 0.01
    deviation: int = 20
    magic: int = 20260213
    stop_loss_points: int = 400
    take_profit_points: int = 800


def _require_mt5() -> None:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not available.")


def initialize_mt5() -> None:
    _require_mt5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def shutdown_mt5() -> None:
    if mt5 is not None:
        mt5.shutdown()


def ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol not found: {symbol}")
    if not info.visible:
        ok = mt5.symbol_select(symbol, True)
        if not ok:
            raise RuntimeError(f"Could not select symbol in Market Watch: {symbol}")


def has_open_position(symbol: str) -> bool:
    positions = mt5.positions_get(symbol=symbol)
    return bool(positions)


def send_market_order(symbol: str, side: str, cfg: TradeConfig) -> dict:
    ensure_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        raise RuntimeError(f"No tick/info available for symbol: {symbol}")

    point = info.point
    if side == "BUY":
        price = tick.ask
        order_type = ORDER_TYPE_BUY
        sl = price - cfg.stop_loss_points * point
        tp = price + cfg.take_profit_points * point
    elif side == "SELL":
        price = tick.bid
        order_type = ORDER_TYPE_SELL
        sl = price + cfg.stop_loss_points * point
        tp = price - cfg.take_profit_points * point
    else:
        raise ValueError(f"Unsupported side: {side}")

    request = {
        "action": TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": cfg.lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": cfg.deviation,
        "magic": cfg.magic,
        "comment": "ai_trader_live_bot",
        "type_time": ORDER_TIME_GTC,
        "type_filling": ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send failed: {mt5.last_error()}")

    result_dict = result._asdict()
    if result_dict.get("retcode") not in (10008, 10009):
        raise RuntimeError(f"Order rejected: {result_dict}")

    return result_dict
