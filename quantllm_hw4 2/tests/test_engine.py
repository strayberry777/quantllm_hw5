"""引擎单元测试：全部用合成数据，不依赖真实行情。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bt import Backtester, Costs, MarketData           # noqa: E402
from bt.engine import frictionless_nav                 # noqa: E402

DATES = pd.bdate_range("2024-01-02", periods=12)


def flat(o, c=None, **kw):
    o = np.asarray(o, float)
    c = o if c is None else np.asarray(c, float)
    syms = kw.pop("symbols", [f"60000{i}.SH" for i in range(o.shape[1])])
    return MarketData.from_arrays(DATES[: o.shape[0]], syms, open=o, close=c, **kw)


def hold_weights(T, N, col=0, from_t=0):
    W = np.full((T, N), np.nan)
    W[from_t] = 0.0
    W[from_t, col] = 1.0
    return W


# ---------------------------------------------------------------- 时间线与估值
def test_signal_executes_next_open():
    o = np.array([[10, 10, 11, 12, 12, 12, 12, 12, 12, 12]], float).T
    c = np.array([[10, 10.5, 11.5, 12, 12, 12, 12, 12, 12, 12]], float).T
    d = flat(o, c)
    W = hold_weights(10, 1, from_t=1)          # 第 1 日收盘给信号
    r = Backtester(d, Costs.zero(), capital=11_000, lots=False).run(W)
    tr = r.trades
    assert len(tr) == 1 and tr.date.iloc[0] == DATES[2] and tr.price.iloc[0] == 11
    assert r.nav.iloc[1] == 11_000            # 信号当日不受影响
    assert r.nav.iloc[2] == pytest.approx(11_000 / 11 * 11.5)


def test_buy_and_hold_equals_adjusted_price_ratio():
    rng = np.random.default_rng(0)
    c = 10 * np.cumprod(1 + rng.normal(0, 0.02, 12))[:, None]
    o = c * (1 + rng.normal(0, 0.005, (12, 1)))
    adj = np.ones((12, 1)); adj[6:] = 1.1            # 第 6 日除权：原始价下调，cumadj 上调
    o[6:] /= 1.1; c[6:] /= 1.1
    d = flat(o, c, cumadj=adj)
    r = Backtester(d, Costs.zero(), lots=False).run(hold_weights(12, 1))
    expect = d.adj_close[-1, 0] / d.adj_open[1, 0]
    assert r.nav.iloc[-1] / r.nav.iloc[0] == pytest.approx(expect, rel=1e-12)


# ---------------------------------------------------------------- 费用
def test_fees_buy_and_sell():
    o = np.full((5, 1), 10.0)
    d = flat(o)
    W = np.full((5, 1), np.nan); W[0] = 1.0; W[2] = 0.0
    costs = Costs(slippage=0.0)
    r = Backtester(d, costs, capital=100_000).run(W)
    buy, sell = r.trades.iloc[0], r.trades.iloc[1]
    assert buy.qty == 9900                         # 整手且留出费用
    assert buy.fee == pytest.approx(max(99_000 * 0.0003, 5) + 99_000 * 0.00001)
    # 2024 年已在 2023-08-28 之后：印花税 0.05%
    assert sell.fee == pytest.approx(99_000 * (0.0003 + 0.00001 + 0.0005))
    assert r.nav.iloc[-1] == pytest.approx(100_000 - buy.fee - sell.fee)


def test_stamp_duty_cut():
    c = Costs()
    assert c.stamp(pd.Timestamp("2023-08-25")) == 0.001
    assert c.stamp(pd.Timestamp("2023-08-28")) == 0.0005
    f = c.fee(np.array([100_000.0]), "sell", pd.Timestamp("2024-01-02"))[0]
    assert f == pytest.approx(100_000 * (0.0003 + 0.00001 + 0.0005))
    assert c.fee(np.array([1000.0]), "buy", pd.Timestamp("2024-01-02"))[0] == pytest.approx(5 + 0.01)


def test_slippage_moves_price_against_trader():
    d = flat(np.full((4, 1), 10.0))
    W = np.full((4, 1), np.nan); W[0] = 0.5; W[1] = 0.0
    r = Backtester(d, Costs.zero().__class__(0, 0, 0, 0, 0, 0.001), lots=False).run(W)
    assert r.trades.price.tolist() == pytest.approx([10.01, 9.99])


# ---------------------------------------------------------------- 整手
def test_lot_rounding_main_and_star():
    o = np.full((3, 2), 33.0)
    d = flat(o, symbols=["600000.SH", "688001.SH"])
    W = np.full((3, 2), np.nan); W[0] = [0.5, 0.5]
    r = Backtester(d, Costs.zero(), capital=20_000).run(W)
    q = dict(zip(r.trades.symbol, r.trades.qty))
    assert q["600000.SH"] == 300                   # 10000/33=303 → 300
    assert q["688001.SH"] == 303                   # 科创板 ≥200 后按 1 股递增
    W2 = np.full((3, 2), np.nan); W2[0] = [0.0, 0.2]
    r2 = Backtester(d, Costs.zero(), capital=20_000).run(W2)
    assert r2.trades.empty                         # 4000/33=121 股 < 200，科创板买不了


# ---------------------------------------------------------------- 涨跌停 / 停牌
def test_limit_up_open_blocks_buy_limit_down_blocks_sell():
    # 第 0~5 日正常（满足上市满 5 日），第 6 日开盘一字涨停，第 8 日开盘一字跌停
    px = [10, 10, 10, 10, 10, 10, 11, 11, 9.9, 9.9, 9.9, 9.9]
    o = np.array(px, float)[:, None]
    d = flat(o)
    assert d.buy_blocked[6, 0] and d.sell_blocked[8, 0]
    W = np.full((12, 1), np.nan); W[5] = 1.0; W[7] = 0.0
    r = Backtester(d, Costs.zero(), lots=False, retry=False).run(W)
    assert r.trades.empty                          # 第 6 日封涨停买不进（不顺延）
    r = Backtester(d, Costs.zero(), lots=False).run(W)
    assert r.trades.date.tolist() == [DATES[7], DATES[9]]   # 顺延：第 7 日买入；第 8 日卖不出，第 9 日卖
    W = np.full((12, 1), np.nan); W[4] = 1.0; W[7] = 0.0
    r = Backtester(d, Costs.zero(), lots=False).run(W)
    assert r.trades.side.tolist() == ["BUY", "SELL"]
    assert r.trades.date.tolist() == [DATES[5], DATES[9]]   # 第 8 日封跌停卖不出，顺延到第 9 日
    r = Backtester(d, Costs.zero(), lots=False, retry=False).run(W)
    assert r.trades.side.tolist() == ["BUY"]                 # 关闭顺延：目标作废


def test_suspension_holds_and_marks_last_close():
    o = np.array([10, 10, 10, np.nan, np.nan, 12, 12, 12], float)[:, None]
    c = np.array([10, 10, 11, np.nan, np.nan, 12, 12, 12], float)[:, None]
    d = flat(o, c)
    W = np.full((8, 1), np.nan); W[0] = 1.0; W[2] = 0.0     # 第 3 日停牌，卖出指令无法执行
    r = Backtester(d, Costs.zero(), lots=False).run(W)
    assert r.nav.iloc[3] == r.nav.iloc[2] == pytest.approx(1e6 * 1.1)
    assert r.trades.date.tolist() == [DATES[0 + 1], DATES[5]]   # 复牌第一天执行卖出


def test_first_five_days_have_no_limit():
    o = np.array([10, 14, 18, 25, 30, 36, 40, 40], float)[:, None]
    d = flat(o, symbols=["688999.SH"])
    assert not d.buy_blocked[:5].any()
    assert d.buy_blocked[6, 0] or d.limit_up[6, 0] == pytest.approx(36 * 1.2)


# ---------------------------------------------------------------- 与独立实现交叉验证
def test_engine_matches_vectorized_formula_frictionless():
    rng = np.random.default_rng(42)
    T, N = 12, 6
    c = 20 * np.cumprod(1 + rng.normal(0, 0.03, (T, N)), axis=0)
    o = c * (1 + rng.normal(0, 0.01, (T, N)))
    adj = np.ones((T, N)); adj[7:, 2] = 1.05; o[7:, 2] /= 1.05; c[7:, 2] /= 1.05
    d = flat(o, c, cumadj=adj, symbols=[f"60000{i}.SH" for i in range(N)])
    W = np.full((T, N), np.nan)
    for t in (0, 3, 4, 8):                          # 不定期调仓，中间按漂移持有
        w = rng.random(N); w[rng.random(N) < 0.3] = 0
        W[t] = w / w.sum() * 0.9
    eng = Backtester(d, Costs.zero(), lots=False, limits=False).run(W).nav.to_numpy()
    ref = frictionless_nav(d, W)
    assert np.allclose(eng / eng[0], ref, rtol=1e-10)


# ---------------------------------------------------------------- ST 推断 / 成交规模 / 期末清仓
def test_st_inference_flags_5pct_limits_and_clears_on_10pct_move():
    px = [10.0] * 8
    for _ in range(4):                       # 连续收在 +5% 限价
        px.append(round(px[-1] * 1.05 + 1e-9, 2))
    px += [px[-1]] * 3
    px.append(round(px[-1] * 1.08, 2))       # 涨 8%：只有 10% 涨跌幅的股票才可能，说明已摘帽
    px += [px[-1]] * 2
    o = np.array(px)[:, None]
    dates = pd.bdate_range("2024-01-02", periods=len(px))
    d = MarketData.from_arrays(dates, ["600000.SH"], open=o, close=o)
    st = d.st[:, 0]
    assert not st[:9].any()
    assert st[9:15].all()                    # 第二次收在 5% 限价起标记为 ST，并一直保持
    assert not st[15:].any()                 # 价格超出 ±5% 后解除


def test_st_not_inferred_on_star_board():
    px = [10.0] * 8
    for _ in range(4):
        px.append(round(px[-1] * 1.05 + 1e-9, 2))
    o = np.array(px)[:, None]
    d = MarketData.from_arrays(pd.bdate_range("2024-01-02", periods=len(px)), ["688001.SH"], open=o, close=o)
    assert not d.st.any()


def test_participation_cap_spreads_order_over_days():
    o = np.full((6, 1), 10.0)
    d = flat(o, amt=np.full((6, 1), 1e6))     # 每天成交额 100 万
    W = np.full((6, 1), np.nan); W[0] = 1.0
    r = Backtester(d, Costs.zero(), capital=1e6, max_participation=0.1).run(W)
    buys = r.trades[r.trades.side == "BUY"]
    assert (buys.value <= 1e5 + 1e-6).all() and len(buys) == 5     # 每天最多买 10 万，顺延 5 天


def test_liquidate_at_end_sells_at_close():
    o = np.full((5, 1), 10.0); c = np.full((5, 1), 10.5)
    d = flat(o, c)
    W = np.full((5, 1), np.nan); W[0] = 1.0
    r = Backtester(d, Costs(commission=0.0001, min_commission=0, transfer=0, stamp_old=0.0005, stamp_new=0.0005, slippage=0),
                   capital=1e6, liquidate_at_end=True).run(W)
    last = r.trades.iloc[-1]
    assert last.side == "SELL" and last.date == DATES[4] and last.price == 10.5
    assert r.daily.exposure.iloc[-1] == 0
    assert last.fee == pytest.approx(last.value * 0.0006)
