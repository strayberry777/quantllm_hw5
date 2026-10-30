"""报告图。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import CXMT, OUT, RAW, ROOT
from bt import MarketData

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9893"
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK TC", "PingFang SC", "Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=160)
    plt.close(fig)


# 1) 5DR 净值
nav = pd.read_csv(OUT / "5dr" / "nav.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(9, 4.2))
for col, lab, c, ls in (("POOL_5d_net", "股票池等权（基准）", GREY, "-"), ("5DR_5d_gross", "5DR 扣费前", BLUE, "--"),
                        ("5DR_5d_net", "5DR 扣费后", BLUE, "-"), ("MOM5_5d_net", "5 日动量 扣费后", ORANGE, "-")):
    ax.plot(nav.index, nav[col], color=c, ls=ls, lw=1.6)
    ax.text(nav.index[-1], nav[col].iloc[-1], f"  {lab} {nav[col].iloc[-1]:.2f}", va="center", color=c, fontsize=9)
ax.axhline(1, color=INK2, lw=0.6)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_xlim(nav.index[0], nav.index[-1] + pd.Timedelta(days=420).to_numpy())
ax.set_ylabel("净值")
ax.set_title("5DR（每 5 日调仓，T+1 开盘成交）：扣费前就没跑赢等权股票池", loc="left", fontsize=11)
save(fig, "5dr_nav.png")

# 2) 十分位
dec = pd.read_csv(OUT / "5dr" / "deciles.csv")
fig, ax = plt.subplots(figsize=(8, 3.8))
colors = [BLUE if k == 1 else ORANGE if k == 10 else GREY for k in dec.decile]
ax.bar(dec.decile, dec.ann_return * 100, color=colors, width=0.62)
for x, v in zip(dec.decile, dec.ann_return * 100):
    ax.text(x, v + (0.6 if v >= 0 else -0.6), f"{v:.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(dec.decile, ["1\n跌最多\n(5DR)"] + [str(k) for k in range(2, 10)] + ["10\n涨最多"])
ax.set_ylabel("年化收益（零费用）%")
ax.set_ylim(dec.ann_return.min() * 100 - 4, dec.ann_return.max() * 100 + 3)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_title("按过去 5 日收益分十组：两头都差，「涨最多」的一组差得多", loc="left", fontsize=11)
save(fig, "5dr_deciles.png")

# 3) 长鑫：价格 + 冻结策略持仓 + 净值
L = int((OUT / "cxmt" / "L_star.txt").read_text())
d = MarketData.load(RAW, symbols=[CXMT], start="2026-07-27")
px = pd.Series(d.close[:, 0], index=d.dates)
cn = pd.read_csv(OUT / "cxmt" / "cxmt_nav.csv", index_col=0, parse_dates=True)
tr = pd.read_csv(OUT / "cxmt" / f"cxmt_trades_L{L}.csv", parse_dates=["date"])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
a1.plot(px.index, px, color=INK, lw=1.4)
buys, sells = tr[tr.side == "BUY"].date.tolist(), tr[tr.side == "SELL"].date.tolist()
for b, s in zip(buys, sells + [px.index[-1]] * (len(buys) - len(sells))):
    a1.axvspan(b, s, color=BLUE, alpha=0.12, lw=0)
a1.axvline(px.index[L], color=INK2, lw=0.8, ls=":")
a1.text(px.index[L], px.max(), f" 第 {L+1} 天起才有 {L} 日信号", color=INK2, fontsize=8.5, va="top")
a1.set_ylabel("收盘价（元）")
a1.yaxis.grid(True, color=GRID, lw=0.8)
a1.set_title(f"长鑫科技 688825.SH：蓝色区间为 {L} 日动量持仓（冻结参数，T+1 开盘成交）", loc="left", fontsize=11)
for col, c in (("买入持有", GREY), (f"动量 L={L}", BLUE)):
    a2.plot(cn.index, cn[col], color=c, lw=1.8)
    a2.text(cn.index[-1], cn[col].iloc[-1], f"  {col} {cn[col].iloc[-1] - 1:+.1%}", color=c, va="center", fontsize=9)
a2.axhline(1, color=INK2, lw=0.6)
a2.set_ylabel("净值（扣费）")
a2.yaxis.grid(True, color=GRID, lw=0.8)
a2.set_xlim(px.index[0], px.index[-1] + pd.Timedelta(days=12).to_numpy())
a2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
save(fig, "cxmt.png")

# 4) 三个样本里：收益差与回撤
ps = pd.read_csv(OUT / "cxmt" / "peers_summary.csv").rename(columns={"Unnamed: 0": "sample"})
ipo = pd.read_csv(OUT / "cxmt" / "star_ipo_54d.csv")
cx = pd.read_csv(OUT / "cxmt" / "cxmt_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
labels = ["同类股\n2020–2025\n(选参)", "同类股\n2026\n(留出)", "科创板大新股\n前 54 天\n(留出)", "长鑫\n54 天"]
tr_, ho = ps[(ps["sample"] == "train") & (ps.L == L)].iloc[0], ps[(ps["sample"] == "holdout_2026") & (ps.L == L)].iloc[0]
ip = ipo[ipo.L == L]
c_s, c_b = cx[cx.L == L].iloc[0], cx[cx.rule == "买入持有"].iloc[0]
mdd_s = [tr_.median_mdd, ho.median_mdd, ip.mdd.median(), c_s.max_drawdown]
mdd_b = [tr_.median_bh_mdd, ho.median_bh_mdd, ip.bh_mdd.median(), c_b.max_drawdown]
beat = [tr_.share_beat_bh, ho.share_beat_bh, (ip.diff_total > 0).mean(), float(c_s.total_return > c_b.total_return)]
x = np.arange(4)
ax = axes[0]
ax.bar(x - 0.18, np.array(mdd_b) * 100, 0.34, color=GREY, label="买入持有")
ax.bar(x + 0.18, np.array(mdd_s) * 100, 0.34, color=BLUE, label=f"动量 L={L}")
for xi, v in zip(x - 0.18, mdd_b):
    ax.text(xi, v * 100 - 1, f"{v:.0%}", ha="center", va="top", fontsize=8)
for xi, v in zip(x + 0.18, mdd_s):
    ax.text(xi, v * 100 - 1, f"{v:.0%}", ha="center", va="top", fontsize=8, color=BLUE)
ax.set_xticks(x, labels, fontsize=8.5)
ax.set_ylabel("最大回撤（中位数）%")
ax.set_ylim(min(mdd_b + mdd_s) * 100 - 10, 0)
ax.set_xlim(-0.6, 3.9)
ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.0))
ax.set_title("回撤：四个样本里都更小", loc="left", fontsize=11)
ax = axes[1]
ax.bar(x, np.array(beat) * 100, 0.5, color=[BLUE if b > 0.5 else ORANGE for b in beat])
for xi, v in zip(x, beat):
    ax.text(xi, v * 100 + 2, f"{v:.0%}" if xi < 3 else ("跑赢" if v else "没跑赢\n(只有 1 只)"), ha="center", fontsize=9)
ax.axhline(50, color=INK2, lw=0.8, ls="--")
ax.set_xticks(x, labels, fontsize=8.5)
ax.set_ylim(0, 105)
ax.set_ylabel("收益跑赢买入持有的比例 %")
ax.set_title("收益：跑不跑赢取决于行情", loc="left", fontsize=11)
save(fig, "trend_samples.png")
print("charts ok")

# 5) 剔除过热股：留出期净值与相对强弱
ov = pd.read_csv(OUT / "overheat" / "nav.csv", index_col=0, parse_dates=True)
h = ov[["C_holdout_net", "POOL_holdout_net"]].dropna()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
for col, lab, c in (("POOL_holdout_net", "股票池等权（基准）", GREY), ("C_holdout_net", "剔除过热股（去两头 10%）", BLUE)):
    a1.plot(h.index, h[col], color=c, lw=1.8)
    a1.text(h.index[-1], h[col].iloc[-1], f"  {lab} {h[col].iloc[-1] - 1:+.1%}", color=c, va="center", fontsize=9)
a1.axhline(1, color=INK2, lw=0.6)
a1.yaxis.grid(True, color=GRID, lw=0.8)
a1.set_ylabel("净值（扣费）")
a1.set_xlim(h.index[0], h.index[-1] + pd.Timedelta(days=260).to_numpy())
a1.set_title("剔除过热股：留出期 2024-01 至 2026-10（参数只用 2020–2023 选定），扣费后", loc="left", fontsize=11)
rel = h["C_holdout_net"] / h["POOL_holdout_net"]
a2.plot(rel.index, rel, color=BLUE, lw=1.6)
a2.fill_between(rel.index, 1, rel, color=BLUE, alpha=0.12, lw=0)
a2.axhline(1, color=INK2, lw=0.6)
a2.yaxis.grid(True, color=GRID, lw=0.8)
a2.set_ylabel("相对基准")
a2.text(rel.index[-1], rel.iloc[-1], f"  {rel.iloc[-1] - 1:+.1%}", color=BLUE, va="center", fontsize=9)
save(fig, "overheat_holdout.png")
print("overheat chart ok")

# 6) 作业规格：2025-01-02 建仓 → 2026-09-30 清仓，10 亿元，非 ST
an = pd.read_csv(OUT / "assignment" / "nav.csv", index_col=0, parse_dates=True)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True, gridspec_kw={"height_ratios": [1.7, 1]})
series = (("非 ST 股票池等权（基准）", "非 ST 股票池等权（基准）", GREY, "-"),
          ("全部非 ST 等权（对照，不做流动性筛选）", "全部非 ST 等权（不筛流动性）", GREY, ":"),
          ("5DR 五日反转", "5DR 五日反转", ORANGE, "-"),
          ("剔除过热股（主策略）", "剔除过热股（主策略）", BLUE, "-"))
ends = []
for col, lab, c, ls in series:
    a1.plot(an.index, an[col], color=c, ls=ls, lw=2.0 if c == BLUE else 1.4)
    ends.append([an[col].iloc[-1], lab, c])
ends.sort(key=lambda x: x[0])
for k in range(1, len(ends)):                       # 末端标签避免重叠
    ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.045)
for y, lab, c in ends:
    a1.text(an.index[-1], y, f"  {lab}", color=c, va="center", fontsize=8.5)
for col, lab, c, ls in series:
    pass
a1.axhline(1, color=INK2, lw=0.6)
a1.yaxis.grid(True, color=GRID, lw=0.8)
a1.set_ylabel("净值（扣费）")
a1.set_xlim(an.index[0], an.index[-1] + pd.Timedelta(days=170).to_numpy())
a1.set_title("作业规格：非 ST 全市场 · 10 亿元 · 佣金万一 + 印花税万五 · 2025-01-02 建仓 → 2026-09-30 清仓",
             loc="left", fontsize=10.5)
rel = an["剔除过热股（主策略）"] / an["非 ST 股票池等权（基准）"]
a2.plot(rel.index, rel, color=BLUE, lw=1.6)
a2.fill_between(rel.index, 1, rel, color=BLUE, alpha=0.12, lw=0)
a2.axhline(1, color=INK2, lw=0.6)
a2.yaxis.grid(True, color=GRID, lw=0.8)
a2.set_ylabel("主策略 / 基准")
a2.text(rel.index[-1], rel.iloc[-1], f"  {rel.iloc[-1] - 1:+.1%}", color=BLUE, va="center", fontsize=9)
save(fig, "assignment_nav.png")

mo = pd.read_csv(OUT / "assignment" / "monthly.csv", index_col=0)
ex = (1 + mo["剔除过热股（主策略）"]) / (1 + mo["非 ST 股票池等权（基准）"]) - 1
fig, ax = plt.subplots(figsize=(9, 3.3))
ax.bar(range(len(ex)), ex * 100, color=[BLUE if v >= 0 else ORANGE for v in ex], width=0.65)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(len(ex)), [m[2:] for m in ex.index], rotation=90, fontsize=8)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_ylabel("月度超额 %")
ax.set_title(f"主策略相对基准的月度超额：{(ex > 0).sum()} / {len(ex)} 个月为正", loc="left", fontsize=11)
save(fig, "assignment_monthly.png")
print("assignment charts ok")
