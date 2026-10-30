"""长鑫科技（688825.SH）趋势策略：同类股上选 L → 两个留出样本 → 冻结后用在长鑫。见 research/PROTOCOL.md §4。"""
import numpy as np
import pandas as pd

from _common import CXMT, OUT, RAW
from bt import Backtester, Costs, MarketData, summarize
from bt import strategies as S
from peers import MEMORY, PEERS

LS = (5, 10, 20)
WINDOW = 54
CAPITAL = 1e6
out = OUT / "cxmt"
out.mkdir(parents=True, exist_ok=True)
full = MarketData.load(RAW)
T = len(full.dates)
di = lambda s: int(full.dates.searchsorted(pd.Timestamp(s)))


def one(sym, s0, end, L=None):
    """单只股票在 [s0, end) 区间：L=None 为买入持有，否则为 L 日时间序列动量。两者都在 s0+1 开盘起成交。"""
    d = full.select([sym])
    if L is None:
        W = np.full((T, 1), np.nan); W[s0] = 1.0
    else:
        W = S.ts_momentum(d, L)
        state = pd.DataFrame(W).ffill().fillna(0.0).to_numpy()   # 每天的持仓状态（信号只在变化日给出）
        W[:s0] = np.nan
        W[s0] = state[s0]                                            # 区间起点按当日状态建仓
    r = Backtester(d, Costs(), capital=CAPITAL).run(W, start=s0, end=end)
    m = summarize(r.nav, turnover=r.daily.turnover, fees=r.daily.fees)
    m["trades"] = len(r.trades)
    m["exposure_avg"] = r.daily.exposure.mean()
    return m, r


def stock_rows(sample, syms, a, b):
    rows = []
    for s in syms:
        first = int(np.flatnonzero(full.traded[:, full.col(s)])[0])
        s0 = max(a, first)
        if b - s0 < 60:
            continue
        bh, _ = one(s, s0, b)
        for L in LS:
            m, _ = one(s, s0, b, L)
            rows.append({"sample": sample, "symbol": s, "name": PEERS.get(s, ""), "group": "存储" if s in MEMORY else "半导体",
                         "L": L, "start": full.dates[s0].date(), "days": m["days"],
                         "ann": m["ann_return"], "bh_ann": bh["ann_return"], "diff_ann": m["ann_return"] - bh["ann_return"],
                         "sharpe": m["sharpe"], "bh_sharpe": bh["sharpe"], "mdd": m["max_drawdown"], "bh_mdd": bh["max_drawdown"],
                         "trades": m["trades"], "exposure": m["exposure_avg"], "fee_drag_ann": m["fee_drag_ann"]})
    return rows


def agg(df):
    return df.groupby("L").agg(n=("symbol", "size"), median_diff_ann=("diff_ann", "median"),
                               share_beat_bh=("diff_ann", lambda x: (x > 0).mean()),
                               median_sharpe=("sharpe", "median"), median_bh_sharpe=("bh_sharpe", "median"),
                               median_mdd=("mdd", "median"), median_bh_mdd=("bh_mdd", "median"),
                               median_trades=("trades", "median"))


# ---------- 1) 选择样本：同类股 2020-01-02 ~ 2025-12-31
train = pd.DataFrame(stock_rows("train", PEERS, 0, di("2026-01-01")))
ag_train = agg(train)
best = ag_train["median_diff_ann"]
L_star = int(best[best == best.max()].index.max())          # 并列取较大的 L
print("选择样本\n", ag_train.round(4), "\n冻结 L =", L_star, flush=True)

# ---------- 2) 留出 1：同类股 2026
hold = pd.DataFrame(stock_rows("holdout_2026", PEERS, di("2026-01-01"), T))
ag_hold = agg(hold)
print("留出 1：同类股 2026\n", ag_hold.round(4), flush=True)

# ---------- 3) 留出 2：科创板首日成交额最大的 20 只新股，上市后 54 天
firsts = []
for j, s in enumerate(full.symbols):
    if not s.startswith(("688", "689")) or s == CXMT:
        continue
    tr = np.flatnonzero(full.traded[:, j])
    if len(tr) == 0 or tr[0] == 0 or tr[0] + WINDOW > T:
        continue
    firsts.append((s, tr[0], full.amt[tr[0], j]))
ipo = sorted(firsts, key=lambda x: -x[2])[:20]
rows = []
for s, f, amt in ipo:
    bh, _ = one(s, f, f + WINDOW)
    for L in LS:
        m, _ = one(s, f, f + WINDOW, L)
        rows.append({"sample": "star_ipo_54d", "symbol": s, "name": PEERS.get(s, ""), "L": L, "start": full.dates[f].date(),
                     "first_day_amt_yi": amt / 1e8, "ann": m["ann_return"], "bh_ann": bh["ann_return"],
                     "total": m["total_return"], "bh_total": bh["total_return"],
                     "diff_total": (1 + m["total_return"]) / (1 + bh["total_return"]) - 1,
                     "diff_ann": m["ann_return"] - bh["ann_return"], "sharpe": m["sharpe"], "bh_sharpe": bh["sharpe"],
                     "mdd": m["max_drawdown"], "bh_mdd": bh["max_drawdown"], "trades": m["trades"]})
ipo_df = pd.DataFrame(rows)
ag_ipo = ipo_df.groupby("L").agg(n=("symbol", "size"), median_diff_total=("diff_total", "median"),
                                 share_beat_bh=("diff_total", lambda x: (x > 0).mean()),
                                 median_total=("total", "median"), median_bh_total=("bh_total", "median"),
                                 median_mdd=("mdd", "median"), median_bh_mdd=("bh_mdd", "median"))
print("留出 2：科创板大型新股前 54 天\n", ag_ipo.round(4), flush=True)

# ---------- 4) 长鑫
f = int(np.flatnonzero(full.traded[:, full.col(CXMT)])[0])
cx, navs = [], {}
bh, rbh = one(CXMT, f, T)
navs["买入持有"] = rbh.nav / CAPITAL
cx.append({"rule": "买入持有", "L": None, **bh})
for L in LS:
    m, r = one(CXMT, f, T, L)
    navs[f"动量 L={L}"] = r.nav / CAPITAL
    cx.append({"rule": f"动量 L={L}" + ("（冻结）" if L == L_star else ""), "L": L, **m})
    r.trades.to_csv(out / f"cxmt_trades_L{L}.csv", index=False, float_format="%.4f")
cx = pd.DataFrame(cx)
print("长鑫\n", cx[["rule", "total_return", "max_drawdown", "sharpe", "trades", "exposure_avg", "fees_total"]].round(4))

train.to_csv(out / "peers_train.csv", index=False, float_format="%.6f")
hold.to_csv(out / "peers_holdout_2026.csv", index=False, float_format="%.6f")
ipo_df.to_csv(out / "star_ipo_54d.csv", index=False, float_format="%.6f")
pd.concat({"train": ag_train, "holdout_2026": ag_hold}).to_csv(out / "peers_summary.csv", float_format="%.6f")
ag_ipo.to_csv(out / "star_ipo_summary.csv", float_format="%.6f")
cx.to_csv(out / "cxmt_summary.csv", index=False, float_format="%.6f")
pd.DataFrame(navs).to_csv(out / "cxmt_nav.csv", float_format="%.6f")
(out / "L_star.txt").write_text(str(L_star))
