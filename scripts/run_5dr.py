"""5DR 全市场验证：扣费前后、两种调仓频率、镜像动量、股票池等权基准、十分位、分年。"""
import pandas as pd

from _common import OUT, RAW
from bt import Backtester, Costs, MarketData, summarize
from bt import strategies as S
from bt.metrics import by_year

CAPITAL = 1e8          # 300+ 只等权：100 万会被整手和 5 元最低佣金扭曲（见协议修订 1）
out = OUT / "5dr"
out.mkdir(parents=True, exist_ok=True)
d = MarketData.load(RAW)
score = S.past_return(d, 5)
pool = S.universe(d)

rows, navs, years = [], {}, []
for every in (5, 1):
    W = {
        "5DR": S.decile_weights(score, pool, 0.0, 0.1, every, 60),
        "MOM5": S.decile_weights(score, pool, 0.9, 1.0 + 1e-9, every, 60),
        "POOL": S.decile_weights(score, pool, 0.0, 1.0 + 1e-9, every, 60),
    }
    for cost_name, costs in (("gross", Costs.zero()), ("net", Costs())):
        res = {k: Backtester(d, costs, capital=CAPITAL).run(w, start=60) for k, w in W.items()}
        for k, r in res.items():
            m = summarize(r.nav, bench=res["POOL"].nav, turnover=r.daily.turnover, fees=r.daily.fees)
            rows.append({"strategy": k, "rebalance_every": every, "costs": cost_name, **m})
            navs[f"{k}_{every}d_{cost_name}"] = r.nav / r.nav.iloc[0]
            if k != "POOL":
                y = by_year(r.nav, res["POOL"].nav)
                y["strategy"], y["rebalance_every"], y["costs"] = k, every, cost_name
                years.append(y)
        print(f"every={every} {cost_name} done", flush=True)

summary = pd.DataFrame(rows)
summary.to_csv(out / "summary.csv", index=False, float_format="%.6f")
pd.DataFrame(navs).to_csv(out / "nav.csv", float_format="%.6f")
pd.concat(years).rename_axis("year").reset_index().to_csv(out / "by_year.csv", index=False, float_format="%.6f")

# 十分位（零费用、每 5 日调仓）：反转效应是否单调
dec = []
for k in range(10):
    w = S.decile_weights(score, pool, k / 10, (k + 1) / 10 + (1e-9 if k == 9 else 0), 5, 60)
    r = Backtester(d, Costs.zero(), lots=False).run(w, start=60)
    m = summarize(r.nav)
    dec.append({"decile": k + 1, "ann_return": m["ann_return"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]})
pd.DataFrame(dec).to_csv(out / "deciles.csv", index=False, float_format="%.6f")

cols = ["strategy", "rebalance_every", "costs", "ann_return", "sharpe", "max_drawdown", "excess_ann",
        "info_ratio", "turnover_ann", "fee_drag_ann"]
print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(pd.DataFrame(dec).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
