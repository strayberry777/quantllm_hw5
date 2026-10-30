"""因子评价模块测试（合成数据）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bt import MarketData                                                     # noqa: E402
from bt.factor_eval import (EvalConfig, corr_rows, evaluate, lookahead_check,  # noqa: E402
                            newey_west_t, xs_rank)


def market(T=200, N=80, seed=0):
    rng = np.random.default_rng(seed)
    c = 10 * np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0)
    o = c * (1 + rng.normal(0, 0.003, (T, N)))
    syms = [f"60{i:04d}.SH" for i in range(N)]
    return MarketData.from_arrays(pd.bdate_range("2023-01-02", periods=T), syms, open=o, close=c,
                                  high=np.fmax(o, c) * 1.01, low=np.fmin(o, c) * 0.99)


def test_corr_rows_matches_numpy():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(5, 100)), rng.normal(size=(5, 100))
    x[0, :10] = np.nan
    got = corr_rows(x, y)
    for i in range(5):
        ok = np.isfinite(x[i])
        assert got[i] == pytest.approx(np.corrcoef(x[i, ok], y[i, ok])[0, 1], abs=1e-12)


def test_xs_rank_is_percentile_within_mask():
    a = np.arange(100, dtype=float)[None, :].repeat(2, 0)
    m = np.ones_like(a, dtype=bool); m[1, 50:] = False
    r = xs_rank(a, m)
    assert r[0, 0] == pytest.approx(0.005) and r[0, -1] == pytest.approx(0.995)
    assert np.isnan(r[1, 60]) and r[1, 49] == pytest.approx(0.99)


def test_newey_west_t_of_constant_mean_series():
    x = np.r_[np.ones(50), -np.ones(50)] + 0.5
    assert newey_west_t(x, 0) == pytest.approx(0.5 / (np.std(x) / np.sqrt(100)))


def test_lookahead_check_catches_future_data():
    d = market()
    ok, _ = lookahead_check(d, lambda d: -d.adj_close)
    assert ok
    bad, diff = lookahead_check(d, lambda d: np.roll(d.adj_close, -1, axis=0))
    assert not bad and diff > 0


def test_perfect_factor_scores_high_and_noise_scores_low():
    d = market(T=260, N=120)
    # 「作弊」因子：直接用未来 5 日收益（只用来检验统计量的方向与量级）
    C = d.adj_close; O = d.adj_open
    fut = np.full_like(C, np.nan); fut[:-6] = O[6:] / O[1:-5] - 1
    noise = np.random.default_rng(3).normal(size=C.shape)
    cfg = EvalConfig(start="2023-01-02", backtest=False, min_listed=1, liq_drop=0.0, exclude_st=False)
    r = evaluate(d, {"CHEAT": fut, "NOISE": noise}, cfg)
    s = r.summary
    assert s.loc["CHEAT", "rankic_mean"] > 0.99 and s.loc["CHEAT", "monotonic"] == pytest.approx(1.0)
    assert abs(s.loc["NOISE", "rankic_mean"]) < 0.03 and abs(s.loc["NOISE", "t_nw"]) < 3
    assert s.loc["NOISE", "verdict"] in ("无预测力", "方向相反", "有效，但不单调")


def test_deep_dive_runs_and_breakeven_logic():
    from bt.factor_eval import BUILTIN, deep_dive
    d = market(T=260, N=120)
    cfg = EvalConfig(start="2023-01-02", backtest=False, min_listed=1, liq_drop=0.0, exclude_st=False)
    dd = deep_dive(d, "REV5", BUILTIN["REV5"], cfg)
    assert set(dd.subsets["group"]) >= {"全样本", "流动性", "市场环境"}
    assert list(dd.sweep.index) == [1, 2, 3, 5, 10, 20]
    be = dd.breakeven
    if be["top_ex_bp"] > 0:
        assert be["breakeven_rt_bp"] == pytest.approx(be["top_ex_bp"] / be["turnover"])
    else:
        assert np.isnan(be["breakeven_rt_bp"])
    assert be["cost_rt_bp"] == pytest.approx(17.0)
