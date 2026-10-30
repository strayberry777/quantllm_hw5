"""课程作业规格下的全市场选股回测。

作业要求（老师补充说明）：
    1. 在非 ST 股票上做全市场选股
    2. 初始资金 10 亿元
    3. 交易费用：佣金单边万一；印花税万五（卖出）
    4. 2025 年第一个交易日开始建仓，2026 年 9 月最后一个交易日清空离场

实现：
    - 2024-12-31 收盘出信号 → 2025-01-02 开盘建仓；之后每 5 个交易日收盘出信号、次日开盘调仓
    - 2026-09-30 收盘按收盘价清空全部持仓（收盘跌停或停牌的卖不出，单独列出）
    - 费用只有作业规定的两项：佣金 0.01%（双边、无最低收费）、印花税 0.05%（卖出）；主结果不加滑点，另做滑点敏感性
    - 10 亿元资金需要考虑容量：单只股票每天成交 ≤ 前一交易日成交额的 10%，超出部分顺延
"""
import numpy as np
import pandas as pd

from _common import OUT, RAW
from bt import Backtester, Costs, MarketData, summarize
from bt import strategies as S

CAPITAL = 1e9
BUILD, EXIT = "2025-01-02", "2026-09-30"
EVERY = 5
PARTICIPATION = 0.10
BAND = 0.0002


def spec_costs(slippage: float = 0.0) -> Costs:
    return Costs(commission=0.0001, min_commission=0.0, transfer=0.0, stamp_old=0.0005, stamp_new=0.0005,
                 slippage=slippage)


out = OUT / "assignment"
out.mkdir(parents=True, exist_ok=True)
d = MarketData.load(RAW)
first = int(d.dates.searchsorted(pd.Timestamp(BUILD)))
last = int(d.dates.searchsorted(pd.Timestamp(EXIT), side="right")) - 1
assert d.dates[first] == pd.Timestamp(BUILD) and d.dates[last] == pd.Timestamp(EXIT)
s0, end = first - 1, last + 1                       # s0 = 2024-12-31：信号日；end：清仓日之后
print("信号日", d.dates[s0].date(), "建仓日", d.dates[first].date(), "清仓日", d.dates[last].date(), "交易日数", last - first + 1)

score = S.past_return(d, 5)
pool = S.universe(d, exclude_st=True)
pool_all = S.universe(d, exclude_st=False)
pool_wide = S.universe(d, min_age=1, liq_drop=0.0, exclude_st=True)   # 不做上市天数与流动性筛选的全部非 ST 股票


def weights(lo, hi, p=pool):
    W = S.decile_weights(score, p, lo, hi, EVERY, s0)
    W[last:] = np.nan                                # 清仓日不再开盘调仓
    return W


STRATS = {
    "剔除过热股（主策略）": (0.1, 0.9),
    "5DR 五日反转": (0.0, 0.1),
    "非 ST 股票池等权（基准）": (0.0, 1.0 + 1e-9),
}


def run(W, slippage=0.0, part=PARTICIPATION, capital=CAPITAL):
    bt = Backtester(d, spec_costs(slippage), capital=capital, band=BAND, max_participation=part,
                    liquidate_at_end=True, keep_positions=False)
    return bt.run(W, start=s0, end=end)


res = {k: run(weights(*v)) for k, v in STRATS.items()}
res["全部非 ST 等权（对照，不做流动性筛选）"] = run(weights(0.0, 1.0 + 1e-9, p=pool_wide))
bench = res["非 ST 股票池等权（基准）"].nav
rows = []
for k, r in res.items():
    m = summarize(r.nav, bench=bench if "基准" not in k and "对照" not in k else None,
                  turnover=r.daily.turnover, fees=r.daily.fees)
    held = r.daily.positions.iloc[1:-1]
    rej = r.rejects
    rows.append({"strategy": k, **m,
                 "avg_holdings": held.mean(), "trades": len(r.trades),
                 "final_cash_share": r.daily.cash.iloc[-1] / r.nav.iloc[-1],
                 "capped_orders": int((rej.reason.str.contains("成交额上限")).sum()),
                 "blocked_orders": int((rej.reason.str.contains("涨停|跌停")).sum()),
                 "unsold_at_exit": int((rej.date == d.dates[last]).sum())})
summary = pd.DataFrame(rows)
summary.to_csv(out / "summary.csv", index=False, float_format="%.6f")

# 月度收益
mon = pd.DataFrame({k: r.nav for k, r in res.items()})
me = mon.resample("ME").last()                    # 第一行是 2024-12-31（= 初始资金）
monthly = me.pct_change().iloc[1:]
monthly.index = monthly.index.strftime("%Y-%m")
monthly.to_csv(out / "monthly.csv", float_format="%.6f")
(mon / CAPITAL).to_csv(out / "nav.csv", float_format="%.6f")

# 清仓日与未成交
for k, r in res.items():
    tag = "main" if "主策略" in k else ("5dr" if "5DR" in k else ("pool" if "基准" in k else "wide"))
    r.trades.to_csv(out / f"trades_{tag}.csv.gz", index=False, float_format="%.4f",
                    compression={"method": "gzip", "mtime": 0})   # mtime=0：重跑得到完全相同的文件
    r.rejects.to_csv(out / f"rejects_{tag}.csv", index=False)
    r.daily.to_csv(out / f"daily_{tag}.csv", float_format="%.4f")

# 敏感性（主策略）
sens = []
Wm = weights(*STRATS["剔除过热股（主策略）"])
Wb = weights(*STRATS["非 ST 股票池等权（基准）"])
for name, kw in (("作业口径（无滑点，成交额上限 10%）", {}),
                 ("滑点 5bp", {"slippage": 0.0005}), ("滑点 10bp", {"slippage": 0.001}),
                 ("成交额上限 5%", {"part": 0.05}), ("不设成交额上限", {"part": None})):
    rm, rb = run(Wm, **kw), run(Wb, **kw)
    m = summarize(rm.nav, bench=rb.nav, fees=rm.daily.fees)
    sens.append({"case": name, "total_return": m["total_return"], "bench_total": m["bench_total"],
                 "excess_total": m["excess_total"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                 "fees_total": m["fees_total"]})
# 不剔除 ST 时会怎样
r_inc = run(weights(*STRATS["剔除过热股（主策略）"], p=pool_all))
m = summarize(r_inc.nav, bench=bench, fees=r_inc.daily.fees)
sens.append({"case": "对照：不剔除 ST", "total_return": m["total_return"], "bench_total": m["bench_total"],
             "excess_total": m["excess_total"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
             "fees_total": m["fees_total"]})
# 股票池不做流动性筛选时，同一规则相对同口径等权
r_w = run(weights(0.1, 0.9, p=pool_wide))
m = summarize(r_w.nav, bench=res["全部非 ST 等权（对照，不做流动性筛选）"].nav, fees=r_w.daily.fees)
sens.append({"case": "对照：股票池不做流动性筛选（基准同口径）", "total_return": m["total_return"], "bench_total": m["bench_total"],
             "excess_total": m["excess_total"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
             "fees_total": m["fees_total"]})
pd.DataFrame(sens).to_csv(out / "sensitivity.csv", index=False, float_format="%.6f")

# ST 推断统计
st_cnt = pd.Series(d.st[s0:end].sum(1), index=d.dates[s0:end])
pool_cnt = pd.Series(pool[s0:end].sum(1), index=d.dates[s0:end])
pd.DataFrame({"st_inferred": st_cnt, "pool_size": pool_cnt}).to_csv(out / "universe_counts.csv")
ever = d.st[s0:end].any(0)
pd.Series(d.symbols[ever]).to_csv(out / "st_inferred_symbols.csv", index=False, header=["symbol"])

cols = ["strategy", "total_return", "ann_return", "ann_vol", "sharpe", "max_drawdown", "excess_total", "info_ratio",
        "turnover_ann", "fees_total", "avg_holdings", "trades", "capped_orders", "blocked_orders", "unsold_at_exit",
        "final_cash_share"]
pd.set_option("display.width", 250)
print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(pd.DataFrame(sens).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(monthly.to_string(float_format=lambda x: f"{x:+.2%}"))
print("ST 推断：区间内平均每天", round(st_cnt.mean()), "只，区间内出现过", int(ever.sum()), "只；股票池平均", round(pool_cnt.mean()), "只")
