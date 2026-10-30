"""平台用的策略接口：写一个类，实现 on_bar(ctx)，每个交易日收盘调用一次。

    from bt.api import Strategy

    class MyStrategy(Strategy):
        def on_bar(self, ctx):
            ret = ctx.history("close", 6).pct_change(5).iloc[-1]     # 只能看到今天收盘及以前
            return ctx.equal_weight(ret.nsmallest(10).index)

返回值（目标仓位，下一个交易日开盘执行）：
    {代码: 权重}   目标权重，合计 ≤ 1，剩下的是现金；没写到的已持仓股票目标为 0
    None           今天不调仓，持仓不动
    {}             清仓

ctx 只会把「今天收盘及以前」的数据交给策略，所以策略写不出前视偏差。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MarketData
from .engine import Backtester, BacktestResult, Costs

PRICE_FIELDS = ("open", "high", "low", "close")


class Strategy:
    """策略基类。子类实现 on_bar；需要初始化状态可以覆盖 __init__（记得调用 super().__init__()）。

    trade_band：调仓容忍带（占组合比例，默认 0）。持有几百上千只股票时建议设 0.0002，
    单只调整金额小于组合 0.02% 就不交易，避免为价格漂移付 5 元最低佣金。"""

    trade_band = 0.0

    def __init__(self, **params):
        self.params = params

    def on_start(self, ctx: "Context") -> None:      # 回测开始前调用一次（可选）
        pass

    def on_bar(self, ctx: "Context"):
        raise NotImplementedError("请实现 on_bar(self, ctx)")


class Context:
    def __init__(self, data: MarketData, pool: np.ndarray):
        self._d = data
        self._pool = pool                      # 布尔数组：用户选的股票池
        self._t = 0
        self._t0 = 0
        self._exclude_st = False
        self._state = {"shares": np.zeros(len(data.symbols)), "cash": 0.0, "nav": 0.0}

    # ---------------------------------------------------------------- 时间
    @property
    def date(self) -> pd.Timestamp:
        return self._d.dates[self._t]

    @property
    def bar_index(self) -> int:
        """从回测起点算起的第几个交易日（0 起）。"""
        return self._t - self._t0

    # ---------------------------------------------------------------- 行情（只到今天收盘）
    def history(self, field: str = "close", n: int = 20, adjusted: bool = True) -> pd.DataFrame:
        """最近 n 个交易日（含今天）的数据，行 = 日期，列 = 股票池内的代码。
        价格字段默认为后复权价（可直接算收益率）；停牌日为 NaN。field 可选 open/high/low/close/amt。"""
        if field not in PRICE_FIELDS + ("amt",):
            raise ValueError(f"field 只能是 {PRICE_FIELDS + ('amt',)}")
        lo = max(0, self._t - n + 1)
        a = getattr(self._d, field)[lo:self._t + 1][:, self._pool]
        if field in PRICE_FIELDS and adjusted:
            a = a * self._d.adj[lo:self._t + 1][:, self._pool]
        a = np.where(self._d.traded[lo:self._t + 1][:, self._pool], a, np.nan)
        return pd.DataFrame(a, index=self._d.dates[lo:self._t + 1], columns=self._d.symbols[self._pool])

    def current(self, field: str = "close", adjusted: bool = False) -> pd.Series:
        """今天的数据（默认不复权价，即真实成交价位）。"""
        return self.history(field, 1, adjusted).iloc[-1]

    @property
    def universe(self) -> list[str]:
        """股票池里今天有成交的股票（开启「排除 ST」时不含 ST）。"""
        m = self._pool & self._d.traded[self._t]
        if self._exclude_st:
            m &= ~self._d.st[self._t]
        return list(self._d.symbols[m])

    @property
    def is_st(self) -> pd.Series:
        """今天是否为（推断的）ST 股票。数据没有 ST 标记，按 ±5% 涨跌幅行为推断，见 docs/ST_INFERENCE.md。"""
        return pd.Series(self._d.st[self._t][self._pool], index=self._d.symbols[self._pool])

    @property
    def listed_days(self) -> pd.Series:
        """截至今天，各股票上市后已有成交的交易日数。"""
        return pd.Series(self._d.age[self._t][self._pool], index=self._d.symbols[self._pool])

    @property
    def limit_status(self) -> pd.Series:
        """今天收盘是否封板：1 涨停、-1 跌停、0 其他（新股前 5 日为 0）。"""
        c = self._d.close[self._t]
        up = (c >= self._d.limit_up[self._t] - 0.005) & (self._d.age[self._t] > 5)
        dn = (c <= self._d.limit_down[self._t] + 0.005) & (self._d.age[self._t] > 5)
        s = np.where(up, 1, np.where(dn, -1, 0))
        return pd.Series(s[self._pool], index=self._d.symbols[self._pool])

    # ---------------------------------------------------------------- 账户
    @property
    def positions(self) -> dict:
        """当前持股数 {代码: 股数}（今天收盘后）。"""
        sh = self._state["shares"]
        return {self._d.symbols[i]: float(sh[i]) for i in np.flatnonzero(sh > 0)}

    @property
    def weights(self) -> dict:
        """当前持仓权重 {代码: 权重}（按今天收盘价）。"""
        sh, nav = self._state["shares"], self._state["nav"]
        px = self._d.mark[self._t]
        return {self._d.symbols[i]: float(sh[i] * px[i] / nav) for i in np.flatnonzero(sh > 0)} if nav else {}

    @property
    def cash(self) -> float:
        return float(self._state["cash"])

    @property
    def nav(self) -> float:
        return float(self._state["nav"])

    # ---------------------------------------------------------------- 工具
    @staticmethod
    def equal_weight(symbols, invest: float = 1.0) -> dict:
        symbols = list(symbols)
        return {s: invest / len(symbols) for s in symbols} if symbols else {}


@dataclass
class RunConfig:
    start: str
    end: str
    capital: float = 1_000_000
    symbols: list[str] | None = None        # None = 全部
    costs: Costs | None = None
    lots: bool = True
    limits: bool = True
    retry: bool = True
    band: float = 0.0                        # 调仓容忍带（占组合比例）；策略类可用 trade_band 属性自带
    exclude_st: bool = False                 # 不持有 ST：目标里的 ST 股票权重清零，其余按比例放大到原合计
    max_participation: float | None = None   # 单只股票当日成交 ≤ 前一日成交额 × 该比例
    liquidate_at_end: bool = False           # 最后一天收盘清仓
    first_trade_on_start: bool = False       # True：开始日期当天开盘就建仓（信号来自前一个交易日收盘）


def _to_row(target, data: MarketData, pool: np.ndarray) -> np.ndarray | None:
    if target is None:
        return None
    if not isinstance(target, dict):
        raise TypeError(f"on_bar 必须返回 dict 或 None，收到 {type(target).__name__}")
    row = np.zeros(len(data.symbols))
    for sym, w in target.items():
        if sym not in data.symbols:
            raise KeyError(f"未知代码：{sym}")
        i = data.col(sym)
        if not pool[i]:
            raise KeyError(f"{sym} 不在股票池里")
        w = float(w)
        if not np.isfinite(w) or w < 0:
            raise ValueError(f"{sym} 的权重 {w} 无效：A 股不能做空，权重需 ≥ 0")
        row[i] = w
    if row.sum() > 1 + 1e-6:
        raise ValueError(f"目标权重合计 {row.sum():.4f} > 1（不允许加杠杆）")
    return row


def run_strategy(data: MarketData, strategy: Strategy, cfg: RunConfig, progress=None) -> tuple[BacktestResult, int, int]:
    """逐日运行：第 t 日收盘调用 strategy.on_bar → 第 t+1 日开盘由引擎撮合。返回 (结果, 起点下标, 终点下标)。"""
    i0 = int(data.dates.searchsorted(pd.Timestamp(cfg.start)))
    i1 = int(data.dates.searchsorted(pd.Timestamp(cfg.end), side="right"))
    if cfg.first_trade_on_start:
        if i0 == 0:
            raise ValueError("开始日期之前没有交易日，无法在开始当天开盘建仓")
        i0 -= 1          # 从前一个交易日收盘出信号，开始日期当天开盘成交
    if i1 - i0 < 2:
        raise ValueError("回测区间至少需要 2 个交易日")
    pool = np.ones(len(data.symbols), dtype=bool)
    if cfg.symbols:
        unknown = [s for s in cfg.symbols if s not in data.symbols]
        if unknown:
            raise KeyError(f"数据里没有这些代码：{', '.join(unknown[:10])}")
        pool[:] = False
        pool[[data.col(s) for s in cfg.symbols]] = True
    ctx = Context(data, pool)
    ctx._exclude_st = cfg.exclude_st
    ctx._t0 = i0
    ctx._t = i0
    ctx._state = {"shares": np.zeros(len(data.symbols)), "cash": cfg.capital, "nav": cfg.capital}
    strategy.on_start(ctx)
    total = i1 - i0

    def on_close(t, state):
        ctx._t, ctx._state = t, state
        if progress is not None and (t - i0) % 20 == 0:
            progress((t - i0) / total, ctx.date)
        row = _to_row(strategy.on_bar(ctx), data, pool)
        if row is not None and cfg.exclude_st:
            total_w = row.sum()
            row = np.where(data.st[t], 0.0, row)
            if row.sum() > 0:
                row = row * total_w / row.sum()
        return row

    band = max(cfg.band, float(getattr(strategy, "trade_band", 0.0)))
    bt = Backtester(data, cfg.costs if cfg.costs is not None else Costs(), capital=cfg.capital,
                    lots=cfg.lots, limits=cfg.limits, retry=cfg.retry, keep_positions=True, band=band,
                    max_participation=cfg.max_participation, liquidate_at_end=cfg.liquidate_at_end)
    res = bt.run(start=i0, end=i1, on_close=on_close)
    if progress is not None:
        progress(1.0, data.dates[i1 - 1])
    return res, i0, i1


def compile_strategy(code: str) -> type[Strategy]:
    """执行用户代码，找出其中继承 Strategy 的类（只在本机运行，等同于运行自己写的脚本）。"""
    ns: dict = {"__name__": "user_strategy", "np": np, "pd": pd}
    exec(compile(code, "<策略代码>", "exec"), ns)
    classes = [v for v in ns.values() if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy]
    if not classes:
        raise ValueError("代码里没有找到继承 Strategy 的类")
    return classes[-1]
