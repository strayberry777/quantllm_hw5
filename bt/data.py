"""
行情数据：读取 7 个 Feather 日频矩阵（交易日 × 证券），派生交易约束。

口径（均已用数据核对，见 docs/DATA_NOTES.md）：
- 后复权价 = 价格 × cumadj（平安银行 2020-05-28 除息日：原始涨幅 0.54%，× cumadj 后 2.27%，与含息收益一致）
- 可交易：当日有开盘价、收盘价且成交额 > 0；成交额为 0 的格子是停牌日沿用的前收盘价
- 涨跌停参考价 = 上一个交易日收盘价 × 上一日 cumadj / 当日 cumadj（即除权后前收盘）
- 涨跌幅：主板 10%；科创板（688/689）20%；创业板（300/301）2020-08-24 起 20%、之前 10%；
  上市后前 5 个交易日不设涨跌幅
- 封板：开盘价触及涨停价、或「一字板」（开高低收同价）且高开，视为买不进；跌停对称视为卖不出。
  一字板规则同时覆盖了数据里没有标识的 ST 股（5% 涨跌幅）
- ST：数据没有 ST 标记，按「涨跌幅被限制在 ±5%」的价格行为推断（见 _derive 里的规则与 docs/ST_INFERENCE.md）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "amt", "num_trades", "cumadj")
GEM_REFORM = pd.Timestamp("2020-08-24")
NO_LIMIT_DAYS = 5
TICK_EPS = 0.005
ST_WINDOW = 60


def board_of(symbol: str) -> str:
    code = symbol.split(".")[0]
    if code.startswith(("688", "689")):
        return "STAR"
    if code.startswith(("300", "301")):
        return "GEM"
    return "MAIN"


@dataclass
class MarketData:
    """全部字段都是 numpy 二维数组 [日期, 证券]，与 dates / symbols 对齐。"""
    dates: pd.DatetimeIndex
    symbols: pd.Index
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    amt: np.ndarray
    num_trades: np.ndarray
    cumadj: np.ndarray
    derived: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ 读取
    @classmethod
    def load(cls, data_dir: str | Path, symbols: list[str] | None = None,
             start: str | None = None, end: str | None = None) -> "MarketData":
        data_dir = Path(data_dir)
        frames = {}
        for f in FIELDS:
            df = pd.read_feather(data_dir / f"{f}.feather").set_index("date")
            df.index = pd.DatetimeIndex(df.index)
            if symbols is not None:
                df = df.reindex(columns=symbols)
            frames[f] = df
        idx = frames["close"].index
        mask = np.ones(len(idx), dtype=bool)
        if start:
            mask &= idx >= pd.Timestamp(start)
        if end:
            mask &= idx <= pd.Timestamp(end)
        md = cls(dates=idx[mask], symbols=frames["close"].columns,
                 **{f: frames[f].to_numpy(dtype=np.float64)[mask] for f in FIELDS})
        md._derive()
        return md

    @classmethod
    def from_arrays(cls, dates, symbols, **arrays) -> "MarketData":
        """测试用：直接由数组构造。缺省的 high/low/amt/num_trades/cumadj 自动补齐。"""
        o = np.asarray(arrays["open"], dtype=float)
        c = np.asarray(arrays["close"], dtype=float)
        md = cls(dates=pd.DatetimeIndex(dates), symbols=pd.Index(symbols), open=o, close=c,
                 high=np.asarray(arrays.get("high", np.fmax(o, c)), dtype=float),
                 low=np.asarray(arrays.get("low", np.fmin(o, c)), dtype=float),
                 amt=np.asarray(arrays.get("amt", np.where(np.isnan(c), np.nan, 1e8)), dtype=float),
                 num_trades=np.asarray(arrays.get("num_trades", np.where(np.isnan(c), np.nan, 1e3)), dtype=float),
                 cumadj=np.asarray(arrays.get("cumadj", np.where(np.isnan(c), np.nan, 1.0)), dtype=float))
        md._derive()
        return md

    # ------------------------------------------------------------------ 派生
    def _derive(self) -> None:
        T, N = self.close.shape
        traded = np.isfinite(self.open) & np.isfinite(self.close) & (np.nan_to_num(self.amt) > 0)

        def ffill(a: np.ndarray) -> np.ndarray:
            out = a.copy()
            for t in range(1, T):
                m = ~np.isfinite(out[t])
                out[t, m] = out[t - 1, m]
            return out

        last_close = ffill(np.where(traded, self.close, np.nan))
        adj = ffill(self.cumadj)
        last_adj = ffill(np.where(traded, self.cumadj, np.nan))
        ref = np.full((T, N), np.nan)
        ref[1:] = last_close[:-1] * last_adj[:-1] / adj[1:]

        boards = np.array([board_of(s) for s in self.symbols])
        pct = np.full((T, N), 0.10)
        pct[:, boards == "STAR"] = 0.20
        gem_after = (self.dates >= GEM_REFORM)[:, None] & (boards == "GEM")[None, :]
        pct[gem_after] = 0.20
        age = np.cumsum(traded, axis=0)                      # 第几个交易日（含当日）
        limited = age > NO_LIMIT_DAYS
        up = np.round(ref * (1 + pct) + 1e-9, 2)
        dn = np.round(ref * (1 - pct) + 1e-9, 2)
        one_price = traded & (np.abs(self.high - self.low) < 1e-9)
        o = self.open
        buy_blocked = traded & limited & ((o >= up - TICK_EPS) | (one_price & (o > ref + TICK_EPS)))
        sell_blocked = traded & limited & ((o <= dn + TICK_EPS) | (one_price & (o < ref - TICK_EPS)))

        # ---- 推断 ST（数据里没有 ST 标记）：ST 股涨跌幅为 ±5%，而主板普通股为 ±10%
        #   触发：近 60 个交易日内至少 2 天收盘恰好收在 ±5% 限价上，且这 60 天价格从未超出 ±5% 区间
        #   维持：一旦标记为 ST，直到某天价格超出 ±5% 区间（说明涨跌幅已恢复为 10%，即摘帽）才解除
        #   只用当日及以前的数据，不会前视；只适用于主板（以及 2020-08-24 前的创业板）
        main5 = (boards == "MAIN")[None, :] | ((boards == "GEM")[None, :] & (self.dates < GEM_REFORM)[:, None])
        up5 = np.round(ref * 1.05 + 1e-9, 2)
        dn5 = np.round(ref * 0.95 + 1e-9, 2)
        seasoned = traded & (age > NO_LIMIT_DAYS)
        close_at5 = seasoned & main5 & ((np.abs(self.close - up5) < 0.0051) | (np.abs(self.close - dn5) < 0.0051))
        beyond5 = seasoned & ((self.high > up5 + 0.0051) | (self.low < dn5 - 0.0051))

        def roll(a: np.ndarray, w: int) -> np.ndarray:
            cs = np.cumsum(a, axis=0)
            out = cs.copy()
            out[w:] = cs[w:] - cs[:-w]
            return out

        trigger = (roll(close_at5, ST_WINDOW) >= 2) & (roll(beyond5, ST_WINDOW) == 0) & traded & main5
        st = np.zeros((T, N), dtype=bool)
        state = np.zeros(N, dtype=bool)
        for t in range(T):
            state = (state | trigger[t]) & ~beyond5[t] & main5[t]
            st[t] = state

        # 已被推断为 ST 的股票（以前一交易日状态为准，开盘前即可知道），涨跌停按 ±5% 判断封板
        st_prev = np.zeros_like(st)
        st_prev[1:] = st[:-1]
        pct = np.where(st_prev, 0.05, pct)
        up = np.round(ref * (1 + pct) + 1e-9, 2)
        dn = np.round(ref * (1 - pct) + 1e-9, 2)
        buy_blocked = traded & limited & ((o >= up - TICK_EPS) | (one_price & (o > ref + TICK_EPS)))
        sell_blocked = traded & limited & ((o <= dn + TICK_EPS) | (one_price & (o < ref - TICK_EPS)))

        self.derived = {
            "traded": traded, "age": age, "ref": ref, "limit_pct": pct, "limit_up": up, "limit_down": dn,
            "buy_blocked": buy_blocked, "sell_blocked": sell_blocked,
            "adj": adj, "adj_close": self.close * adj, "adj_open": self.open * adj,
            "mark": ffill(np.where(traded, self.close, np.nan)),   # 停牌按最近收盘价估值
            "boards": boards,
            "st": st,                                           # 推断的 ST 状态
        }

    # ------------------------------------------------------------------ 便捷
    def __getattr__(self, name):
        d = self.__dict__.get("derived", {})
        if name in d:
            return d[name]
        raise AttributeError(name)

    def select(self, symbols) -> "MarketData":
        """按列取子集（保留全部日期，所以上市天数等派生量不变）。"""
        idx = [self.col(s) for s in symbols]
        md = MarketData(dates=self.dates, symbols=pd.Index(list(symbols)),
                        **{f: getattr(self, f)[:, idx] for f in FIELDS})
        md._derive()
        return md

    def frame(self, arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=self.dates, columns=self.symbols)

    def col(self, symbol: str) -> int:
        return int(self.symbols.get_loc(symbol))
