"""A 股日频回测系统：数据 → 策略目标权重 → 撮合执行 → 绩效。"""

from .data import MarketData
from .engine import Backtester, Costs, BacktestResult
from .metrics import summarize

__all__ = ["MarketData", "Backtester", "Costs", "BacktestResult", "summarize"]
