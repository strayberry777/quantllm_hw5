"""研究迴圈的报告图：验证期全部候选的夏普—回撤散点、留出期冠军净值。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import EXAMS, HERE, LEDGER, ROOT, read_jsonl
from objectives import table

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#2f9e6e", "#9a9893"
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK TC", "PingFang SC", "Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
IMG = ROOT / "docs" / "img"
champ = {"EXT5": ("IC 目标", ORANGE), "IVOL60": ("夏普目标", BLUE), "LOWVOL60": ("回撤 / 50-50 混合目标", GREEN)}

t = table(read_jsonl(LEDGER), "valid")
fig, ax = plt.subplots(figsize=(8.5, 5))
for k, r in t.iterrows():
    c, s, z = (champ[k][1], 70, 3) if k in champ else (GREY, 30, 2)
    ax.scatter(-r.mdd * 100, r.sharpe, s=s, color=c, zorder=z, edgecolor=SURFACE, linewidth=1)
    if k in champ or k not in ("LOWVOL120", "DOWNVOL60", "RANGE20", "LV_EXT", "LVILQ"):
        ax.annotate(k, (-r.mdd * 100, r.sharpe), xytext=(5, 3 if k != "LOWVOL60" else 6), textcoords="offset points",
                    fontsize=8.5 if k in champ else 7.5, color=c if k in champ else INK2)
p = t.iloc[0]
ax.scatter(-p.pool_mdd * 100, p.pool_sharpe, marker="D", s=50, color=INK, zorder=3)
ax.annotate("股票池等权（基准）", (-p.pool_mdd * 100, p.pool_sharpe), xytext=(6, -12), textcoords="offset points", fontsize=8.5)
ax.invert_xaxis()
ax.set_xlabel("最大回撤 %（向右越小）")
ax.set_ylabel("夏普比率")
ax.grid(True, color=GRID, lw=0.8)
ax.set_title("验证期（2023-01 至 2024-11）：21 个候选因子的夏普与回撤\n彩色为各目标的冠军（右上角挤在一起的灰点是 LOWVOL120、DOWNVOL60、RANGE20）", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(IMG / "loop_valid_scatter.png", dpi=160); plt.close(fig)

nav = pd.read_csv(HERE / "exam_nav_EXT5_IVOL60_LOWVOL60.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(nav.index, nav["POOL"], color=INK, lw=1.6)
ax.text(nav.index[-1], nav["POOL"].iloc[-1], f"  股票池等权 {nav['POOL'].iloc[-1] - 1:+.1%}", va="center", fontsize=9)
ex = {e["name"]: e for e in read_jsonl(EXAMS)}
for k, (lab, c) in champ.items():
    ax.plot(nav.index, nav[k], color=c, lw=2)
    ax.text(nav.index[-1], nav[k].iloc[-1], f"  {k}（{lab}）{nav[k].iloc[-1] - 1:+.1%}", va="center", fontsize=9, color=c)
ax.axhline(1, color=INK2, lw=0.6)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_xlim(nav.index[0], nav.index[-1] + pd.Timedelta(days=260))
ax.set_ylabel("净值（扣费）")
ax.set_title("留出期考试（2025-01 至 2026-09，每个因子只考一次）：做多最高 10%", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(IMG / "loop_holdout.png", dpi=160); plt.close(fig)
print("ok")

# 留出期：全部考试（冠军 + 追加考试）
nav2 = pd.read_csv(HERE / "exam_nav_pools.csv", index_col=0, parse_dates=True)
nav3 = pd.read_csv(HERE / "exam_nav_A012_A042.csv", index_col=0, parse_dates=True)
allnav = pd.concat([nav, nav2.drop(columns="POOL"), nav3.drop(columns="POOL")], axis=1)
series = [("POOL", "股票池等权（基准）", INK, 1.6), ("POOL_R4", "因子池 22 个等权（含 Alpha101）", "#c0392b", 2.4),
          ("POOL_R3", "因子池 6 个等权（第 0–3 轮）", "#d98c6a", 1.8), ("IVOL60", "IVOL60（夏普冠军）", BLUE, 1.8),
          ("A042", "A042（Alpha101 扣费后最好）", "#8a6fd1", 1.5), ("EXT5", "EXT5（IC 冠军）", ORANGE, 1.2),
          ("LOWVOL60", "LOWVOL60（回撤冠军）", GREEN, 1.2), ("A012", "A012（Alpha101 IC 最强）", GREY, 1.2)]
fig, ax = plt.subplots(figsize=(10, 5.2))
ends = []
for k, lab, c, lw in series:
    ax.plot(allnav.index, allnav[k], color=c, lw=lw)
    ends.append([allnav[k].iloc[-1], f"  {lab} {allnav[k].iloc[-1] - 1:+.1%}", c])
ends.sort(key=lambda e: e[0])
for i in range(1, len(ends)):
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.035)
for y, t, c in ends:
    ax.text(allnav.index[-1], y, t, va="center", fontsize=8.5, color=c)
ax.axhline(1, color=INK2, lw=0.6)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_xlim(allnav.index[0], allnav.index[-1] + pd.Timedelta(days=330))
ax.set_ylabel("净值（扣费）")
ax.set_title("留出期全部考试（2025-01 至 2026-09，每个只考一次）：做多最高 10%", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(IMG / "loop_holdout_all.png", dpi=160); plt.close(fig)
print((allnav.iloc[-1] - 1).round(4).to_dict())
