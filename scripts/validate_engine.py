"""真实数据上的引擎交叉验证：零费用、分数股、关闭涨跌停时，引擎净值 == 独立向量化公式。

公式假设调仓日所有证券都能成交，所以只在样本期内从未停牌的股票上比较（这样两边面对的是同一个问题）。
"""
import numpy as np
import pandas as pd

from _common import CXMT, OUT, RAW
from bt import Backtester, Costs, MarketData
from bt import strategies as S
from bt.engine import frictionless_nav

full = MarketData.load(RAW)
rows = []
never = list(full.symbols[full.traded.all(0)])
d = full.select(never)
for name, W in (("5DR 每 5 日", S.reversal_5d(d, every=5)), ("5DR 每日", S.reversal_5d(d, every=1)),
                ("股票池等权 每 5 日", S.pool_equal_weight(d, every=5)), ("5 日动量 每 5 日", S.momentum_5d(d, every=5))):
    e = Backtester(d, Costs.zero(), lots=False, limits=False, retry=False).run(W).nav.to_numpy()
    ref = frictionless_nav(d, W)
    rows.append({"case": name, "stocks": len(never), "days": len(d.dates),
                 "max_rel_diff": float(np.max(np.abs(e / e[0] / ref - 1))), "final_engine": e[-1] / e[0], "final_formula": ref[-1]})
# 买入持有 == 后复权价之比（长鑫 + 一只有多次分红的股票）
for sym in (CXMT, "600000.SH"):
    dd = full.select([sym])
    f = int(np.flatnonzero(dd.traded[:, 0])[0])
    W = np.full((len(dd.dates), 1), np.nan); W[f] = 1.0
    e = Backtester(dd, Costs.zero(), lots=False, limits=False).run(W, start=f).nav.to_numpy()
    expect = dd.adj_close[-1, 0] / dd.adj_open[f + 1, 0]
    rows.append({"case": f"买入持有 {sym}", "stocks": 1, "days": len(e), "max_rel_diff": abs(e[-1] / e[0] / expect - 1),
                 "final_engine": e[-1] / e[0], "final_formula": expect})
df = pd.DataFrame(rows)
(OUT / "validation").mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "validation" / "engine_vs_formula.csv", index=False, float_format="%.3e")
print(df.to_string(index=False))
assert (df.max_rel_diff < 1e-10).all()
