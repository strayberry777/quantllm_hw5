"""绩效指标。年化按每年 244 个交易日。"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 244


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def summarize(nav: pd.Series, bench: pd.Series | None = None, turnover: pd.Series | None = None,
              fees: pd.Series | None = None, capital: float | None = None) -> dict:
    """nav：净值序列（第一天为起点）。bench：同期基准净值。"""
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    n = len(r)
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann = (1 + total) ** (DAYS / n) - 1 if n > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(DAYS) if n > 1 else np.nan
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(DAYS) if n > 1 and r.std() > 0 else np.nan
    mdd = max_drawdown(nav)
    out = {
        "days": n, "total_return": total, "ann_return": ann, "ann_vol": vol, "sharpe": sharpe,
        "max_drawdown": mdd, "calmar": ann / -mdd if mdd < 0 else np.nan,
        "win_rate": float((r > 0).mean()) if n else np.nan,
    }
    if bench is not None:
        b = bench.reindex(nav.index).dropna()
        br = b.pct_change().dropna()
        ex = (r - br.reindex(r.index)).dropna()
        out["bench_total"] = b.iloc[-1] / b.iloc[0] - 1
        out["excess_total"] = (1 + total) / (1 + out["bench_total"]) - 1
        out["excess_ann"] = ex.mean() * DAYS
        out["info_ratio"] = ex.mean() / ex.std(ddof=1) * np.sqrt(DAYS) if len(ex) > 1 and ex.std() > 0 else np.nan
    if turnover is not None:
        out["turnover_ann"] = float(turnover.reindex(r.index).sum() / n * DAYS) if n else np.nan
    if fees is not None:
        base = capital if capital else nav.iloc[0]
        out["fees_total"] = float(fees.sum())
        out["fee_drag_ann"] = float(fees.reindex(r.index).fillna(0).div(nav.shift(1).reindex(r.index)).sum() / n * DAYS) if n else np.nan
    return out


def by_year(nav: pd.Series, bench: pd.Series | None = None) -> pd.DataFrame:
    r = nav.pct_change().dropna()
    g = (1 + r).groupby(r.index.year).prod() - 1
    df = pd.DataFrame({"strategy": g})
    if bench is not None:
        br = bench.pct_change().dropna()
        df["bench"] = (1 + br).groupby(br.index.year).prod() - 1
        df["excess"] = (1 + df["strategy"]) / (1 + df["bench"]) - 1
    return df
