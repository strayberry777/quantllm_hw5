"""事后诊断（不参与任何判定）：5DR 的收益被「收盘信号 → 次日开盘成交」吃掉了多少？

把每个信号日之后 5 天的收益拆成两段（等权、零费用、不考虑涨跌停）：
    隔夜段  C_t → O_t+1   —— 收盘信号来不及吃到的部分
    持有段  O_t+1 → O_t+6 —— 次日开盘成交后实际持有的部分（与回测口径一致）
"""
import numpy as np
import pandas as pd

from _common import OUT, RAW
from bt import MarketData
from bt import strategies as S

d = MarketData.load(RAW)
score, pool = S.past_return(d, 5), S.universe(d)
C = S._ffill(np.where(d.traded, d.adj_close, np.nan))
O = np.where(d.traded, d.adj_open, C)
T = len(d.dates)
rows = []
for t in range(60, T - 6, 5):
    s = np.where(pool[t] & np.isfinite(score[t]), score[t], np.nan)
    ok = np.isfinite(s)
    rank = np.full(len(s), np.nan)
    rank[ok] = (np.argsort(np.argsort(s[ok])) + 0.5) / ok.sum()
    on = O[t + 1] / C[t] - 1
    hold = O[t + 6] / O[t + 1] - 1
    for name, sel in (("D1_losers", ok & (rank < 0.1)), ("D10_winners", ok & (rank >= 0.9)), ("POOL", ok)):
        rows.append({"date": d.dates[t], "group": name, "overnight": np.nanmean(on[sel]), "hold": np.nanmean(hold[sel])})
df = pd.DataFrame(rows)
df["year"] = df.date.dt.year
piv = df.pivot_table(index="date", columns="group", values=["overnight", "hold"])
res = []
for g in ("D1_losers", "D10_winners"):
    for seg in ("overnight", "hold"):
        x = piv[(seg, g)] - piv[(seg, "POOL")]
        res.append({"group": g, "segment": seg, "mean_excess_bp_per_week": x.mean() * 1e4,
                    "t_stat": x.mean() / x.std(ddof=1) * np.sqrt(len(x)), "weeks": len(x)})
res = pd.DataFrame(res)
res.to_csv(OUT / "5dr" / "decomposition.csv", index=False, float_format="%.4f")
yr = (df.assign(total=lambda x: (1 + x.overnight) * (1 + x.hold) - 1)
        .pivot_table(index="date", columns="group", values=["overnight", "hold"]))
by = []
for y, g in piv.groupby(piv.index.year):
    by.append({"year": y,
               "D1_overnight_bp": (g[("overnight", "D1_losers")] - g[("overnight", "POOL")]).mean() * 1e4,
               "D1_hold_bp": (g[("hold", "D1_losers")] - g[("hold", "POOL")]).mean() * 1e4,
               "D10_hold_bp": (g[("hold", "D10_winners")] - g[("hold", "POOL")]).mean() * 1e4})
pd.DataFrame(by).to_csv(OUT / "5dr" / "decomposition_by_year.csv", index=False, float_format="%.2f")
print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print(pd.DataFrame(by).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
