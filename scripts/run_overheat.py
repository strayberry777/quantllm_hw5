"""追加策略：剔除过热股的等权组合。2020-04 ~ 2023-12 选规则，2024-01 ~ 2026-10 留出。见 research/PROTOCOL.md §5。"""
import pandas as pd

from _common import OUT, RAW
from bt import Backtester, Costs, MarketData, summarize
from bt import strategies as S
from bt.metrics import by_year

CAPITAL, BAND, EVERY = 1e8, 0.0002, 5
RULES = {
    "A 剔除涨幅前 10%": (0.0, 0.9),
    "B 剔除涨幅前 20%": (0.0, 0.8),
    "C 剔除涨幅前 10% 与跌幅前 10%": (0.1, 0.9),
}
out = OUT / "overheat"
out.mkdir(parents=True, exist_ok=True)
d = MarketData.load(RAW)
score, pool = S.past_return(d, 5), S.universe(d)
split = int(d.dates.searchsorted(pd.Timestamp("2024-01-02")))
periods = {"train": (60, split), "holdout": (split, len(d.dates))}


def run(lo, hi, a, b, costs):
    W = S.decile_weights(score, pool, lo, hi, EVERY, a)
    return Backtester(d, costs, capital=CAPITAL, band=BAND).run(W, start=a, end=b)


rows, navs, years = [], {}, []
for period, (a, b) in periods.items():
    for cname, costs in (("gross", Costs.zero()), ("net", Costs())):
        base = run(0.0, 1.0 + 1e-9, a, b, costs)
        navs[f"POOL_{period}_{cname}"] = base.nav / CAPITAL
        rows.append({"period": period, "rule": "股票池等权（基准）", "costs": cname,
                     **summarize(base.nav, turnover=base.daily.turnover, fees=base.daily.fees)})
        for name, (lo, hi) in RULES.items():
            if period == "holdout" and name != chosen:
                continue
            r = run(lo, hi, a, b, costs)
            navs[f"{name[0]}_{period}_{cname}"] = r.nav / CAPITAL
            rows.append({"period": period, "rule": name, "costs": cname,
                         **summarize(r.nav, bench=base.nav, turnover=r.daily.turnover, fees=r.daily.fees)})
            y = by_year(r.nav, base.nav)
            y["period"], y["rule"], y["costs"] = period, name, cname
            years.append(y)
    if period == "train":
        tr = pd.DataFrame(rows)
        tr = tr[(tr.costs == "net") & tr.rule.isin(RULES)]
        chosen = tr.sort_values("excess_ann", ascending=False).rule.iloc[0]
        print("选定：", chosen, flush=True)

summary = pd.DataFrame(rows)
summary.to_csv(out / "summary.csv", index=False, float_format="%.6f")
pd.concat(years).rename_axis("year").reset_index().to_csv(out / "by_year.csv", index=False, float_format="%.6f")
pd.DataFrame(navs).to_csv(out / "nav.csv", float_format="%.6f")
(out / "chosen.txt").write_text(chosen)
cols = ["period", "rule", "costs", "ann_return", "sharpe", "max_drawdown", "excess_ann", "info_ratio", "turnover_ann", "fee_drag_ann"]
print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(pd.concat(years).to_string(float_format=lambda x: f"{x:.4f}"))
