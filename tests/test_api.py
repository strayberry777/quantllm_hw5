"""平台策略接口测试：逐日回调与矩阵方式结果一致、策略看不到未来、返回值校验。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bt import Backtester, Costs, MarketData                     # noqa: E402
from bt.api import RunConfig, Strategy, compile_strategy, run_strategy   # noqa: E402
from bt.templates import TEMPLATES                                # noqa: E402

DATES = pd.bdate_range("2024-01-02", periods=40)
SYMS = ["600000.SH", "600001.SH", "688001.SH", "300001.SZ"]


def market(seed=0):
    rng = np.random.default_rng(seed)
    c = 20 * np.cumprod(1 + rng.normal(0, 0.02, (40, 4)), axis=0)
    o = c * (1 + rng.normal(0, 0.005, (40, 4)))
    return MarketData.from_arrays(DATES, SYMS, open=o, close=c)


def test_callback_equals_matrix():
    d = market()
    rng = np.random.default_rng(1)
    W = np.full((40, 4), np.nan)
    for t in (3, 9, 10, 20, 31):
        w = rng.random(4); W[t] = w / w.sum() * 0.95

    class Replay(Strategy):
        def on_bar(self, ctx):
            t = ctx._t
            return None if not np.isfinite(W[t]).any() else dict(zip(SYMS, W[t]))

    res, i0, i1 = run_strategy(d, Replay(), RunConfig(start="2024-01-02", end=str(DATES[-1].date())))
    ref = Backtester(d, Costs()).run(W, start=i0, end=i1)
    assert np.allclose(res.nav.to_numpy(), ref.nav.to_numpy(), rtol=1e-12)
    assert len(res.trades) == len(ref.trades)


def test_strategy_cannot_see_future():
    d = market()
    seen = []

    class Spy(Strategy):
        def on_bar(self, ctx):
            h = ctx.history("close", 5)
            seen.append((ctx.date, h.index[-1]))
            return None

    run_strategy(d, Spy(), RunConfig(start="2024-01-10", end="2024-02-09"))
    assert all(a == b for a, b in seen)


def test_invalid_returns_are_rejected():
    d = market()

    class Short(Strategy):
        def on_bar(self, ctx):
            return {"600000.SH": -0.5}

    class Lever(Strategy):
        def on_bar(self, ctx):
            return {"600000.SH": 0.8, "600001.SH": 0.8}

    for s in (Short(), Lever()):
        with pytest.raises(ValueError):
            run_strategy(d, s, RunConfig(start="2024-01-02", end="2024-02-09"))


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_templates_compile_and_run(name):
    d = market()
    cls = compile_strategy(TEMPLATES[name])
    res, _, _ = run_strategy(d, cls(), RunConfig(start="2024-01-02", end=str(DATES[-1].date())))
    assert np.isfinite(res.nav).all()
