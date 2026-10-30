"""按四个目标给日志里的因子排名，写 leaderboard.md。"""
import pandas as pd

from common import HERE, LEDGER, read_jsonl
from objectives import bonferroni_t, scores_with_novelty, table

rows = read_jsonl(LEDGER)
tv, tt = table(rows, "valid"), table(rows, "train")
sc, sct = scores_with_novelty(rows, tv), scores_with_novelty(rows, tt)
n = len(rows)
tstar = bonferroni_t(n)
lines = [f"# 排行榜（{n} 个因子，{len({r['round'] for r in rows})} 轮）", "",
         f"多重检验：试了 {n} 个因子，Bonferroni 校正后 IC 显著门槛为 |t| > {tstar:.2f}（单次检验为 1.96）。", ""]
for obj, label in (("IC", "IC t 值"), ("SHARPE", "夏普"), ("MDD", "最大回撤"), ("MIX", "50% 夏普 + 50% 回撤"),
                   ("NOVEL", "新颖度：混合目标 − 与第 0–3 轮冠军的最大相关")):
    s = sc[obj].sort_values(ascending=False)
    lines += [f"## 目标：{label}", "", "| 排名 | 因子 | 轮次 | 验证期得分 | 训练期得分 | IC t | 夏普 | 最大回撤 | 年化超额 | 年换手 | 重复？ |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for i, (k, v) in enumerate(s.head(8).items(), 1):
        r = tv.loc[k]
        lines.append(f"| {i} | {k} | {int(r['round'])} | {v:.3f} | {sct[obj].get(k, float('nan')):.3f} | {r.t_ic:.1f} | {r.sharpe:.2f} | {r.mdd:.1%} | {r.excess:+.1%} | {r.turnover:.0f} | {'是' if r.dup else ''} |")
    lines.append("")
pool = tv.iloc[0]
lines.append(f"验证期基准（非 ST 股票池等权）：夏普 {pool.pool_sharpe:.2f}，最大回撤 {pool.pool_mdd:.1%}。")
(HERE / "leaderboard.md").write_text("\n".join(lines))
pd.set_option("display.width", 220)
print("\n".join(lines))
