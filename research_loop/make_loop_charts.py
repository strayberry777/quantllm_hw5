"""研究过程的可视化：全部候选表、各目标冠军的进化轨迹、因子池效用、最终因子池与组合信号。

因子池的晋升规则（固定、按日志顺序逐个判定，只用训练期和验证期）：
    1. 通过前视检查
    2. 验证期 RankIC t 值 ≥ 当时的 Bonferroni 门槛（按截至该实验的累计试验次数计算），且为正
    3. 训练期 t 值 ≥ 2 且同为正（两段样本方向一致）
    4. 与池内已有因子的横截面秩相关绝对值 < 0.7（否则视为重复）
池效用 = 池内因子排名等权平均后的组合信号，在验证期的平均 RankIC（不做回归拟合权重，避免在验证期上过拟合）。
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CACHE, HERE, LEDGER, ROOT, data, eval_cfg, load_factor, read_jsonl
from objectives import bonferroni_t, scores_with_novelty, table
from bt.factor_eval import _eval_universe, corr_rows, factor_ranks, forward_returns, rank_rows

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, PURPLE, GOLD, GREY = "#2a78d6", "#eb6834", "#2f9e6e", "#8a6fd1", "#c79a2b", "#9a9893"
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK TC", "PingFang SC", "Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
IMG = ROOT / "docs" / "img"
OUT = HERE / "report"
OUT.mkdir(exist_ok=True)

rows = read_jsonl(LEDGER)
for i, r in enumerate(rows):
    r["exp"] = i
tv = table(rows, "valid")
tt = table(rows, "train")

# ---------------------------------------------------------------- 1. 晋升判定
d = data()
cache = {k: np.load(CACHE / f"{k}.npy") for k in tv.index if (CACHE / f"{k}.npy").exists()}
pool, decisions = [], []
for r in rows:
    k = r["name"]
    n_so_far = r["exp"] + 1
    tstar = bonferroni_t(n_so_far)
    if not r.get("lookahead_ok"):
        decisions.append((k, "拒绝：用到未来数据", tstar)); continue
    v, t = r["valid"], r["train"]
    if v["t_nw"] is None or v["t_nw"] < tstar:
        decisions.append((k, "拒绝：验证期不显著" if (v["t_nw"] or 0) >= 0 else "拒绝：方向为负", tstar)); continue
    if t["t_nw"] is None or t["t_nw"] < 2:
        decisions.append((k, "拒绝：训练期不一致", tstar)); continue
    cmax, cwho = 0.0, None
    for m in pool:
        c = abs(float(np.nanmean(corr_rows(cache[k], cache[m]))))
        if c > cmax:
            cmax, cwho = c, m
    if cmax >= 0.7:
        decisions.append((k, f"拒绝：与 {cwho} 重复", tstar)); continue
    pool.append(k)
    decisions.append((k, "晋升", tstar))
dec = pd.DataFrame(decisions, columns=["name", "decision", "t_threshold"]).set_index("name")
dec.loc[[r["name"] for r in rows if r["round"] == 0], "decision"] = \
    dec.loc[[r["name"] for r in rows if r["round"] == 0], "decision"].map(lambda s: "基准 · " + s)

# ---------------------------------------------------------------- 2. 全部候选表
sc = scores_with_novelty(rows, tv)
champ_corr = {r["name"]: r.get("corr_champions") for r in rows}
recs = []
for r in rows:
    k = r["name"]
    rec = {"实验": f"e{r['exp']}", "轮次": r["round"], "因子": k, "假设": r["hypothesis"][:60],
           "决策": dec.loc[k, "decision"]}
    if k in tv.index:
        v = tv.loc[k]
        rec.update({"验证 RankIC": v.rankic, "验证 IC t": v.t_ic, "训练 IC t": tt.loc[k, "t_ic"], "夏普": v.sharpe,
                    "最大回撤": v.mdd, "年化超额": v.excess, "年换手（倍）": v.turnover, "MIX 得分": sc.loc[k, "MIX"]})
    recs.append(rec)
cand = pd.DataFrame(recs)
cand.to_csv(OUT / "candidates.csv", index=False, float_format="%.4f", encoding="utf-8-sig")

# ---------------------------------------------------------------- 3. 冠军进化轨迹（各目标的验证期得分，按实验顺序取历史最好）
objs = [("IC", "IC t 值", BLUE), ("SHARPE", "夏普", GREEN), ("MDD", "最大回撤（越接近 0 越好）", ORANGE),
        ("MIX", "50% 夏普 + 50% 回撤", PURPLE), ("NOVEL", "新颖度（MIX − 与旧冠军相关）", GOLD)]
order = [r["name"] for r in rows if r["name"] in sc.index]
fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
evo = {}
for ax, (o, lab, c) in zip(axes, objs):
    s = sc[o].reindex(order)
    best = s.cummax()
    evo[o] = best
    x = np.arange(len(s))
    ax.scatter(x, s, s=9, color=GREY, alpha=0.5, zorder=2)
    ax.step(x, best, where="post", color=c, lw=2, zorder=3)
    newbest = best.ne(best.shift())
    ax.scatter(x[newbest.to_numpy()], best[newbest], color=c, s=28, zorder=4, edgecolor=SURFACE, linewidth=1)
    k_fin = best[newbest].index[-1]
    ax.annotate(f"冠军：{k_fin}", (len(s) - 1, best.iloc[-1]), xytext=(0, 6), textcoords="offset points",
                fontsize=8.5, color=c, ha="right")
    for rr in sorted({r["round"] for r in rows})[1:]:
        xi = next(i for i, k in enumerate(order) if [r for r in rows if r["name"] == k][0]["round"] == rr)
        ax.axvline(xi - 0.5, color=GRID, lw=1)
    ax.set_title(lab, fontsize=9.5, loc="left")
    ax.set_xlabel("实验序号（竖线 = 新一轮）", fontsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
fig.suptitle("冠军进化轨迹：灰点为每个候选的验证期得分，彩线为截至当时的最好成绩（只在出现新冠军时上升）", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(IMG / "loop_incumbent.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 4. 因子池效用（验证期组合信号的 RankIC）
cfg = eval_cfg("valid")
_, mask = _eval_universe(d, cfg)
H = cfg.horizon
i0 = int(d.dates.searchsorted(pd.Timestamp(cfg.start)))
i1 = int(d.dates.searchsorted(pd.Timestamp(cfg.end), side="right"))
tt_idx = np.arange(i0, i1)
fwd = forward_returns(d, mask, (H,))[H]
fr = rank_rows(fwd[tt_idx])
member_rank = {}
src = {r["name"]: r["source"] for r in rows}
for k in pool:
    n, f, _ = load_factor(src[k])
    member_rank[k] = factor_ranks(d, {k: f}, mask)[k][tt_idx].astype(np.float32)


def combo_ic(members):
    if not members:
        return np.nan, None
    comb = np.nanmean(np.stack([member_rank[m] for m in members]), axis=0)
    ric = pd.Series(corr_rows(rank_rows(comb), fr), index=d.dates[tt_idx])
    return float(ric.mean()), ric


rounds = sorted({r["round"] for r in rows})
util, mean_ic, best_ic, size = [], [], [], []
for rr in rounds:
    members = [k for k in pool if [r for r in rows if r["name"] == k][0]["round"] <= rr]
    u, _ = combo_ic(members)
    util.append(u)
    ics = [tv.loc[m, "rankic"] for m in members]
    mean_ic.append(np.mean(ics) if ics else np.nan)
    best_ic.append(np.max(ics) if ics else np.nan)
    size.append(len(members))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(rounds, util, color=BLUE, lw=2.2, marker="o")
for x, y, n in zip(rounds, util, size):
    a1.annotate(f"{n} 个因子", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
a1.axhline(util[0], color=GREY, ls="--", lw=1)
a1.text(rounds[0] + 0.1, util[0], "第 0 轮水平", va="top", ha="left", fontsize=8.5, color=GREY)
a1.set_xticks(rounds, [f"第 {r} 轮" for r in rounds])
a1.set_title("因子池效用：池内因子等权组合后的验证期 RankIC", loc="left", fontsize=10.5)
a1.yaxis.grid(True, color=GRID, lw=0.8)
a2.plot(rounds, best_ic, color=ORANGE, lw=2, marker="o", label="池内最好的单因子 RankIC")
a2.plot(rounds, mean_ic, color=BLUE, lw=2, marker="o", label="池内单因子平均 RankIC")
a2.set_xticks(rounds, [f"第 {r} 轮" for r in rounds])
a2.legend(frameon=False, fontsize=9)
a2.set_title("池内单因子质量（验证期）", loc="left", fontsize=10.5)
a2.yaxis.grid(True, color=GRID, lw=0.8)
fig.tight_layout()
fig.savefig(IMG / "loop_pool_utility.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 5. 最终因子池与组合信号
u_final, ric_final = combo_ic(pool)
fig = plt.figure(figsize=(12, 4.2 + 0.3 * len(pool)))
gs = fig.add_gridspec(2, 1, height_ratios=[0.3 * len(pool), 3.6], hspace=0.25)
a1 = fig.add_subplot(gs[0])
vals = [tv.loc[k, "rankic"] for k in pool]
y = np.arange(len(pool))[::-1]
a1.barh(y, vals, color=[BLUE if [r for r in rows if r["name"] == k][0]["round"] < 4 else ORANGE for k in pool], height=0.6)
for yi, v in zip(y, vals):
    a1.text(v, yi, f" {v:.4f}", va="center", fontsize=8.5)
a1.set_yticks(y, [f"{k}（第 {[r for r in rows if r['name'] == k][0]['round']} 轮）" for k in pool], fontsize=8.5)
a1.set_xlim(0, max(vals) * 1.18)
a1.set_title(f"最终因子池：{len(pool)} 个晋升因子的验证期 RankIC（蓝 = AI 自提，橙 = Alpha101）", loc="left", fontsize=10.5)
a1.xaxis.grid(True, color=GRID, lw=0.8)
a2 = fig.add_subplot(gs[1])
a2.bar(ric_final.index, ric_final.values, color=GREY, alpha=0.45, width=1.5, label="每日 RankIC")
a2.plot(ric_final.index, ric_final.rolling(20, min_periods=10).mean(), color=BLUE, lw=2, label="20 日平均")
a2.axhline(0, color=INK2, lw=0.8)
a2.legend(frameon=False, fontsize=9, loc="upper left")
a2.set_title(f"组合信号在验证期的稳定性：平均 RankIC {u_final:.4f}，"
             f"{(ric_final > 0).mean():.0%} 的交易日为正", loc="left", fontsize=10.5)
a2.yaxis.grid(True, color=GRID, lw=0.8)
fig.savefig(IMG / "loop_final_pool.png", dpi=150, bbox_inches="tight")
plt.close(fig)

summary = {"pool": pool, "pool_util_by_round": dict(zip(map(str, rounds), util)), "pool_size_by_round": dict(zip(map(str, rounds), size)),
           "final_combo_rankic_valid": u_final, "final_combo_pos_days": float((ric_final > 0).mean()),
           "n_experiments": len(rows), "decisions": dec["decision"].value_counts().to_dict(),
           "incumbent_final": {o: sc[o].idxmax() for o, _, _ in objs}}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print(json.dumps(summary, ensure_ascii=False, indent=1))
