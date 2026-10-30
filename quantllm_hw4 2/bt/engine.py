"""
撮合执行引擎（日频）。

时间线（无前视）：
    第 t 日收盘后  策略只用 ≤ t 的数据给出目标权重 weights[t]
    第 t+1 日开盘  引擎按 weights[t] 调仓：先卖后买，按开盘价 ± 滑点成交
    第 t+1 日收盘  按收盘价估值，记录净值

A 股约束：
    T+1        每日只在开盘调仓一次，先卖后买，当日买入的股份最早次日才能卖出
    整手        主板 / 创业板 100 股整数倍（卖出清仓可卖零股）；科创板单笔不少于 200 股，超出按 1 股递增
    停牌        当日不可交易的证券保持原持仓，按最近收盘价估值
    涨跌停      开盘封涨停不能加仓，封跌停不能减仓（MarketData.buy_blocked / sell_blocked）
    顺延        因停牌 / 封板没成交的证券，目标顺延到下一个交易日继续执行
    除权除息    持股数按 cumadj 比值调整（等价于红利再投资），使持仓价值与后复权价一致
    资金        买入金额不得超过可用现金；不足时按比例缩减
    成交规模    （可选）单只股票当日成交 ≤ 前一交易日成交额 × 上限比例，超出部分顺延
    离场        （可选）最后一天收盘按收盘价清空持仓

费用（Costs）：佣金（双边，有最低收费）、过户费（双边）、印花税（卖出，2023-08-28 起减半）、滑点（双边）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import MarketData

STAMP_CUT = pd.Timestamp("2023-08-28")


@dataclass
class Costs:
    commission: float = 0.0003       # 佣金费率（双边）
    min_commission: float = 5.0      # 单笔最低佣金（元）
    transfer: float = 0.00001        # 过户费（双边）
    stamp_old: float = 0.001         # 印花税（卖出，2023-08-28 之前）
    stamp_new: float = 0.0005        # 印花税（卖出，2023-08-28 起）
    slippage: float = 0.0005         # 滑点（双边，买价上浮 / 卖价下调）

    @classmethod
    def zero(cls) -> "Costs":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def stamp(self, date: pd.Timestamp) -> float:
        return self.stamp_old if date < STAMP_CUT else self.stamp_new

    def fee(self, value: np.ndarray, side: str, date: pd.Timestamp) -> np.ndarray:
        """value 为成交金额数组；返回每笔费用（不含滑点，滑点已体现在成交价）。"""
        value = np.asarray(value, dtype=float)
        comm = np.where(value > 0, np.maximum(value * self.commission, self.min_commission), 0.0)
        f = comm + value * self.transfer
        if side == "sell":
            f = f + value * self.stamp(date)
        return f


@dataclass
class BacktestResult:
    daily: pd.DataFrame                  # 日度：nav / cash / exposure / turnover / fees / positions
    trades: pd.DataFrame                 # 逐笔：date / symbol / side / qty / price / value / fee
    positions: pd.DataFrame | None = None
    rejects: pd.DataFrame | None = None  # 未成交记录：date / symbol / reason
    meta: dict = field(default_factory=dict)

    @property
    def nav(self) -> pd.Series:
        return self.daily["nav"]

    @property
    def returns(self) -> pd.Series:
        return self.daily["nav"].pct_change().fillna(0.0)


class Backtester:
    def __init__(self, data: MarketData, costs: Costs | None = None, capital: float = 1_000_000,
                 lots: bool = True, limits: bool = True, keep_positions: bool = False, band: float = 0.0,
                 retry: bool = True, max_participation: float | None = None, liquidate_at_end: bool = False):
        self.d = data
        # 成交规模上限：单只股票当日成交金额 ≤ max_participation × 前一交易日该股成交额（用前一日，避免看到当日成交量）
        self.max_participation = max_participation
        # 回测最后一天收盘时按收盘价清空全部持仓（收盘跌停或停牌的股票卖不出，记入未成交）
        self.liquidate_at_end = liquidate_at_end
        self.costs = costs if costs is not None else Costs()
        self.capital = float(capital)
        self.lots = lots
        self.limits = limits
        self.keep_positions = keep_positions
        self.retry = retry        # 停牌 / 封板未成交的目标是否顺延到下一交易日
        self.band = band          # 调仓容忍带：单只调整金额 < band × 组合价值时不交易（避免为漂移付最低佣金）

    # ------------------------------------------------------------------ 整手
    def _round_target(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        if not self.lots:
            return np.maximum(target, 0.0)
        star = self.d.boards == "STAR"
        out = np.where(star, np.floor(target), np.floor(target / 100) * 100)
        out = np.where(star & (out < 200), 0.0, out)
        # 只减不清时，主板减仓后余量按整手；清仓（目标 0）允许卖出全部零股
        return np.maximum(out, 0.0)

    # ------------------------------------------------------------------ 主循环
    def run(self, weights: np.ndarray | pd.DataFrame | None = None, start: int = 0, end: int | None = None,
            on_close=None) -> BacktestResult:
        """weights：事先算好的目标权重矩阵；或者传 on_close(t, state) -> 一行目标权重，
        引擎在第 t 日收盘估值后调用它（state 含当日持股、现金、净值），用于平台里的逐日策略。"""
        d, c = self.d, self.costs
        T, N = d.close.shape
        if weights is None:
            W = np.full((T, N), np.nan)
        else:
            W = weights.to_numpy(dtype=float) if isinstance(weights, pd.DataFrame) else np.array(weights, dtype=float)
        end = T if end is None else end
        rejects = []
        traded, mark, adj = d.traded, d.mark, d.adj
        shares = np.zeros(N)
        cash = self.capital
        pend, pend_w = np.zeros(N, dtype=bool), np.zeros(N)
        rows, trades, pos_rows = [], [], []

        for t in range(start, end):
            date = d.dates[t]
            fees_today, traded_value = 0.0, 0.0
            # 1) 除权除息：持股数按 cumadj 比值调整
            if t > start:
                ratio = adj[t] / adj[t - 1]
                held = shares > 0
                ok = held & np.isfinite(ratio)
                shares[ok] = shares[ok] * ratio[ok]
            # 2) 开盘调仓（用前一交易日收盘给出的目标权重；整行 NaN = 当日不调仓）
            #    因停牌 / 封板没能执行的那几只，目标保留到下一个交易日继续执行（其余证券不动）
            if t > start and np.isfinite(W[t - 1]).any():
                w, active = np.nan_to_num(W[t - 1], nan=0.0), np.ones(N, dtype=bool)
                pend_w = w.copy()
            elif t > start and pend.any():
                w, active = pend_w, pend.copy()
            else:
                w, active = None, None
            if w is not None:
                tradable = traded[t].copy()
                o = np.where(tradable, d.open[t], np.nan)
                val_px = np.where(tradable, d.open[t], mark[t - 1])
                value_open = cash + np.nansum(shares * np.nan_to_num(val_px))
                buy_px = o * (1 + c.slippage)
                sell_px = o * (1 - c.slippage)
                with np.errstate(invalid="ignore", divide="ignore"):
                    target = np.where(tradable, w * value_open / buy_px, shares)
                target = np.where(tradable, self._round_target(np.nan_to_num(target), shares), shares)
                delta = np.where(active & tradable, target - shares, 0.0)
                blocked = np.zeros(N, dtype=bool)
                if self.limits:
                    blocked = ((delta > 0) & d.buy_blocked[t]) | ((delta < 0) & d.sell_blocked[t])
                    delta = np.where(blocked, 0.0, delta)
                capped = np.zeros(N, dtype=bool)
                if self.max_participation is not None:
                    prev_amt = np.nan_to_num(d.amt[t - 1]) if t > 0 else np.zeros(N)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        cap = np.floor(np.nan_to_num(self.max_participation * prev_amt / o))
                    capped = np.abs(delta) > cap + 1e-9
                    if self.lots:   # 截断后仍按整手
                        cap = np.where(self.d.boards == "STAR", cap, np.floor(cap / 100) * 100)
                    delta = np.where(capped, np.sign(delta) * np.minimum(np.abs(delta), cap), delta)
                # 停牌：想清仓 / 想建仓但当日不能交易
                want = active & ~tradable & (((w == 0) & (shares > 0)) | ((w > 0) & (shares == 0)))
                pend = (active & (blocked | capped)) | want if self.retry else np.zeros(N, dtype=bool)
                for i in np.flatnonzero(blocked):
                    rejects.append((date, d.symbols[i], "开盘涨停，买不进" if d.buy_blocked[t, i] else "开盘跌停，卖不出"))
                for i in np.flatnonzero(want):
                    rejects.append((date, d.symbols[i], "停牌，无法交易"))
                for i in np.flatnonzero(capped & ~blocked):
                    rejects.append((date, d.symbols[i], "超过成交额上限，部分成交"))
                if self.band > 0:
                    small = np.abs(delta) * np.nan_to_num(o) < self.band * value_open
                    # 只对「已持有且继续持有」的股票生效：新建仓和清仓永远执行
                    delta = np.where(small & (target > 0) & (shares > 0), 0.0, delta)

                # 卖出
                sell_q = np.where(delta < 0, -delta, 0.0)
                if sell_q.any():
                    sv = sell_q * np.nan_to_num(sell_px)
                    sf = c.fee(sv, "sell", date)
                    cash += sv.sum() - sf.sum()
                    shares -= sell_q
                    fees_today += sf.sum()
                    traded_value += sv.sum()
                    for i in np.flatnonzero(sell_q):
                        trades.append((date, d.symbols[i], "SELL", sell_q[i], sell_px[i], sv[i], sf[i]))
                # 买入（现金约束）
                buy_q = np.where(delta > 0, delta, 0.0)
                if buy_q.any():
                    bv = buy_q * np.nan_to_num(buy_px)
                    bf = c.fee(bv, "buy", date)
                    need = bv.sum() + bf.sum()
                    if need > cash + 1e-9:
                        scale = max(cash, 0.0) / need
                        buy_q = self._round_target(buy_q * scale, np.zeros(N)) if self.lots else buy_q * scale
                        bv = buy_q * np.nan_to_num(buy_px)
                        bf = c.fee(bv, "buy", date)
                    cash -= bv.sum() + bf.sum()
                    shares += buy_q
                    fees_today += bf.sum()
                    traded_value += bv.sum()
                    for i in np.flatnonzero(buy_q):
                        trades.append((date, d.symbols[i], "BUY", buy_q[i], buy_px[i], bv[i], bf[i]))
            # 3) 最后一天收盘清仓
            if self.liquidate_at_end and t == end - 1 and (shares > 0).any():
                held = shares > 0
                ok = held & traded[t]
                if self.limits:
                    ok &= ~(d.close[t] <= d.limit_down[t] + 0.005) | (d.age[t] <= 5)
                if self.max_participation is not None:
                    cap = np.floor(np.nan_to_num(self.max_participation * np.nan_to_num(d.amt[t - 1]) / d.close[t]))
                else:
                    cap = np.full(N, np.inf)
                q = np.where(ok, np.minimum(shares, np.where(cap >= shares, shares, cap)), 0.0)
                px = d.close[t] * (1 - c.slippage)
                sv = q * np.nan_to_num(px)
                sf = c.fee(sv, "sell", date)
                cash += sv.sum() - sf.sum()
                shares -= q
                fees_today += sf.sum()
                traded_value += sv.sum()
                for i in np.flatnonzero(q):
                    trades.append((date, d.symbols[i], "SELL", q[i], px[i], sv[i], sf[i]))
                for i in np.flatnonzero(held & (shares > 1e-9)):
                    why = "停牌，无法清仓" if not traded[t, i] else ("收盘跌停，无法清仓" if q[i] == 0 else "超过成交额上限，未清完")
                    rejects.append((date, d.symbols[i], why))
            # 4) 收盘估值
            mv = np.nansum(shares * np.nan_to_num(mark[t]))
            nav = cash + mv
            rows.append((date, nav, cash, mv / nav if nav else 0.0, traded_value / nav if nav else 0.0,
                         fees_today, int((shares > 0).sum())))
            if self.keep_positions:
                pos_rows.append(shares.copy())
            if on_close is not None and t + 1 < end:
                row = on_close(t, {"shares": shares.copy(), "cash": cash, "nav": nav})
                if row is not None:
                    W[t] = row

        daily = pd.DataFrame(rows, columns=["date", "nav", "cash", "exposure", "turnover", "fees", "positions"]).set_index("date")
        tr = pd.DataFrame(trades, columns=["date", "symbol", "side", "qty", "price", "value", "fee"])
        pos = pd.DataFrame(pos_rows, index=daily.index, columns=d.symbols) if self.keep_positions else None
        rj = pd.DataFrame(rejects, columns=["date", "symbol", "reason"])
        return BacktestResult(daily=daily, trades=tr, positions=pos, rejects=rj,
                              meta={"capital": self.capital, "costs": c, "lots": self.lots, "limits": self.limits,
                                    "max_participation": self.max_participation, "liquidate_at_end": self.liquidate_at_end})


def frictionless_nav(data: MarketData, weights: np.ndarray, start: int = 0, end: int | None = None) -> np.ndarray:
    """独立的向量化参照实现（不经过引擎的逐股循环）：零费用、分数股、无涨跌停，全程用后复权价。

    第 t 日开盘按 weights[t-1] 调仓（整行 NaN 表示不调仓，沿用漂移后的持仓）：
        隔夜增长 g_on = 现金权重 + Σ ω_i · O_i,t / C_i,t-1     （ω 为上一日收盘的漂移后权重）
        日内增长 g_in = 现金权重 + Σ w_i · C_i,t / O_i,t
        NAV_t = NAV_t-1 · g_on · g_in
    它假设任何证券在调仓日都能成交，所以与引擎对比时，目标权重只能落在当日可交易的证券上。
    """
    W = np.asarray(weights, dtype=float)
    # 停牌日按最近收盘价估值（与引擎相同）：收盘价前向填充，停牌日开盘价取填充后的收盘价
    C = data.mark * data.adj
    O = np.where(data.traded, data.adj_open, C)
    T, N = O.shape
    end = T if end is None else end
    nav = np.ones(end - start)
    omega, cash = np.zeros(N), 1.0          # 收盘时的组合权重
    for k, t in enumerate(range(start + 1, end), start=1):
        on = omega * O[t] / C[t - 1]
        g_on = cash + np.nansum(on)
        row = W[t - 1]
        if np.isfinite(row).any():          # 调仓：开盘权重 = 目标权重
            w = np.nan_to_num(row)
            cw = 1.0 - w.sum()
        else:                                # 不调仓：开盘权重 = 隔夜漂移后的权重
            w = np.nan_to_num(on) / g_on
            cw = cash / g_on
        intraday = np.nan_to_num(w * C[t] / O[t])
        g_in = cw + intraday.sum()
        nav[k] = nav[k - 1] * g_on * g_in
        omega, cash = intraday / g_in, cw / g_in
    return nav
