"""因子池组合信号的留出期考试（每个组合只考一次）：
    POOL_R3：第 0–3 轮晋升的因子等权组合（加入 Alpha101 之前）
    POOL_R4：第 0–4 轮全部晋升因子等权组合（加入 Alpha101 之后）
组合方式与 make_loop_charts.py 相同：各因子在股票池内的百分位排名取平均。
"""
import datetime as dt
import json

import numpy as np

from common import EXAMS, LEDGER, data, eval_cfg, load_factor, read_jsonl, HERE
from bt.factor_eval import _eval_universe, evaluate, factor_ranks

summary = json.loads((HERE / "report" / "summary.json").read_text())
rows = read_jsonl(LEDGER)
rnd = {r["name"]: r["round"] for r in rows}
src = {r["name"]: r["source"] for r in rows}
pool = summary["pool"]
combos = {"POOL_R3": [k for k in pool if rnd[k] <= 3], "POOL_R4": pool}
taken = {e["name"] for e in read_jsonl(EXAMS)}
if taken & set(combos):
    raise SystemExit(f"已经考过：{taken & set(combos)}，不能重考。")
d = data()
cfg = eval_cfg("holdout")
_, mask = _eval_universe(d, cfg)
ranks = {}
for k in pool:
    _, f, _ = load_factor(src[k])
    ranks[k] = factor_ranks(d, {k: f}, mask)[k].astype(np.float32)
arrays = {name: np.nanmean(np.stack([ranks[m] for m in members]), axis=0) for name, members in combos.items()}
res = evaluate(d, arrays, cfg)
with EXAMS.open("a") as fh:
    for name, members in combos.items():
        s = res.summary.loc[name]
        row = {"time": dt.datetime.now().isoformat(timespec="seconds"), "name": name, "members": members,
               "why": "因子池等权组合（加入 Alpha101 前 / 后）", "t_nw": round(float(s.t_nw), 4),
               "rankic": round(float(s.rankic_mean), 5), "sharpe": round(float(s.bt_sharpe), 4), "mdd": round(float(s.bt_mdd), 4),
               "ann": round(float(s.bt_ann), 4), "excess_ann": round(float(s.bt_excess_ann), 4), "ir": round(float(s.bt_ir), 4),
               "turnover": round(float(s.bt_turnover), 2), "pool_ann": round(float(res.info["pool_ann"]), 4),
               "pool_sharpe": round(float(res.info["pool_sharpe"]), 4), "pool_mdd": round(float(res.info["pool_mdd"]), 4)}
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(row)
res.nav.to_csv(HERE / "exam_nav_pools.csv", float_format="%.5f")
