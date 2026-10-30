"""策略：只负责在每个信号日给出目标权重矩阵（行 = 第 t 日收盘做的决定；整行 NaN = 不调仓）。"""

from __future__ import annotations

import numpy as np

from .data import MarketData


def past_return(d: MarketData, n: int) -> np.ndarray:
    C = np.where(d.traded, d.adj_close, np.nan)
    C = _ffill(C)
    r = np.full_like(C, np.nan)
    r[n:] = C[n:] / C[:-n] - 1
    return r


def _ffill(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for t in range(1, len(out)):
        m = ~np.isfinite(out[t])
        out[t, m] = out[t - 1, m]
    return out


def universe(d: MarketData, min_age: int = 60, liq_drop: float = 0.2, win: int = 20,
             exclude_st: bool = False) -> np.ndarray:
    """可选股票池：当日有成交、上市满 min_age 日、20 日均成交额不在后 liq_drop、收盘不是一字板。
    exclude_st=True 时再剔除推断为 ST 的股票（先剔除 ST，再算流动性分位）。"""
    amt = np.where(d.traded, d.amt, 0.0)
    cs = np.cumsum(amt, axis=0)
    avg = np.full_like(amt, np.nan)
    avg[win:] = (cs[win:] - cs[:-win]) / win
    avg[win - 1] = cs[win - 1] / win
    one_price = d.traded & (np.abs(d.high - d.low) < 1e-9)
    ok = d.traded & (d.age >= min_age) & ~one_price & np.isfinite(avg)
    if exclude_st:
        ok &= ~d.st
    thr = np.nanquantile(np.where(ok, avg, np.nan), liq_drop, axis=1)
    return ok & (avg >= thr[:, None])


def rebalance_days(T: int, every: int, first: int = 0) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    m[first::every] = True
    return m


def decile_weights(score: np.ndarray, pool: np.ndarray, q_lo: float, q_hi: float,
                   every: int = 5, first: int = 0) -> np.ndarray:
    """score 在 [q_lo, q_hi) 分位区间内的股票等权；非调仓日为 NaN。"""
    T, N = score.shape
    W = np.full((T, N), np.nan)
    s = np.where(pool & np.isfinite(score), score, np.nan)
    for t in np.flatnonzero(rebalance_days(T, every, first)):
        row = s[t]
        ok = np.isfinite(row)
        if ok.sum() < 50:
            continue
        rank = np.full(N, np.nan)
        rank[ok] = (np.argsort(np.argsort(row[ok])) + 0.5) / ok.sum()
        sel = ok & (rank >= q_lo) & (rank < q_hi)
        W[t] = 0.0
        if sel.any():
            W[t, sel] = 1.0 / sel.sum()
    return W


def reversal_5d(d: MarketData, every: int = 5, frac: float = 0.1, first: int = 60) -> np.ndarray:
    """5DR：过去 5 日跌得最多的 frac 等权做多。"""
    return decile_weights(past_return(d, 5), universe(d), 0.0, frac, every, first)


def momentum_5d(d: MarketData, every: int = 5, frac: float = 0.1, first: int = 60) -> np.ndarray:
    """5DR 的镜像：过去 5 日涨得最多的 frac 等权做多。"""
    return decile_weights(past_return(d, 5), universe(d), 1 - frac, 1.0 + 1e-9, every, first)


def pool_equal_weight(d: MarketData, every: int = 5, first: int = 60) -> np.ndarray:
    return decile_weights(past_return(d, 5), universe(d), 0.0, 1.0 + 1e-9, every, first)


def ts_momentum(d: MarketData, L: int) -> np.ndarray:
    """时间序列动量：过去 L 日收益 > 0 满仓，否则空仓（每日判断；只有状态变化时才交易）。"""
    r = past_return(d, L)
    T, N = r.shape
    want = np.where(np.isfinite(r), (r > 0).astype(float), 0.0)
    W = np.full((T, N), np.nan)
    prev = np.zeros(N)
    for t in range(T):
        row = want[t]
        if (row != prev).any():
            W[t] = row / max(row.sum(), 1.0)   # 多只股票时等分资金；单只时即 0 / 1
            prev = row
    return W


def buy_and_hold(d: MarketData, col: int | None = None) -> np.ndarray:
    T, N = d.close.shape
    W = np.full((T, N), np.nan)
    first = np.flatnonzero(d.traded[:, col if col is not None else 0])[0]
    W[first] = 0.0
    W[first, col if col is not None else 0] = 1.0
    return W
