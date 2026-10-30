"""跑一轮：前视检查 → 训练期、验证期评价 → 追加写入研究日志。

    python run_round.py <轮次> builtin:REV5 factors/xxx.py ...
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from common import LEDGER, data, eval_cfg, load_factor, read_jsonl
from bt.factor_eval import corr_rows, evaluate, factor_ranks, lookahead_check

KEEP = ["rankic_mean", "icir_ann", "t_nw", "monotonic", "top_ex_bp", "bottom_ex_bp", "top_turnover",
        "bt_ann", "bt_sharpe", "bt_mdd", "bt_excess_ann", "bt_ir", "bt_turnover"]


def main():
    rnd = int(sys.argv[1])
    paths = sys.argv[2:]
    d = data()
    done = {r["name"] for r in read_jsonl(LEDGER)}
    items = []
    for p in paths:
        name, f, hyp = load_factor(p)
        if name in done:
            print(f"跳过 {name}：日志里已有（不重复评价）")
            continue
        ok, diff = lookahead_check(d, f.fn)
        code = Path(p).read_text() if not p.startswith("builtin:") else p
        items.append({"name": name, "source": p, "factor": f, "hypothesis": hyp, "lookahead_ok": ok, "lookahead_diff": diff,
                      "code_sha": hashlib.sha1(code.encode()).hexdigest()[:10]})
        print(f"{name}: 前视检查 {'通过' if ok else '未通过（最大差 %.3g）' % diff}")
    good = {it["name"]: it["factor"] for it in items if it["lookahead_ok"]}
    res = {}
    for period in ("train", "valid"):
        if good:
            print(f"评价 {period} …", flush=True)
            res[period] = evaluate(d, good, eval_cfg(period))
    # 与「更早」因子的最大相关：日志里已有的 + 本轮排在它前面的（验证期起点后每 20 日取一个截面）
    max_corr = {}
    if good:
        mask = d.traded & ~d.st
        i0 = int(d.dates.searchsorted(np.datetime64("2023-01-03")))
        samp = np.arange(i0, i0 + 400, 20)
        earlier = {}
        for r in read_jsonl(LEDGER):
            if r.get("lookahead_ok"):
                n2, f2, _ = load_factor(r["source"])
                earlier[n2] = factor_ranks(d, {n2: f2}, mask)[n2][samp]
        for k, f in good.items():
            rk = factor_ranks(d, {k: f}, mask)[k][samp]
            best, who = 0.0, None
            for k2, rk2 in earlier.items():
                c = float(np.nanmean(corr_rows(rk, rk2)))
                if abs(c) > abs(best):
                    best, who = c, k2
            max_corr[k] = (best, who)
            earlier[k] = rk
    with LEDGER.open("a") as fh:
        for it in items:
            row = {"round": rnd, "time": dt.datetime.now().isoformat(timespec="seconds"), "name": it["name"], "source": it["source"],
                   "hypothesis": it["hypothesis"], "code_sha": it["code_sha"], "lookahead_ok": it["lookahead_ok"]}
            if it["lookahead_ok"]:
                for period in ("train", "valid"):
                    s = res[period].summary.loc[it["name"]]
                    row[period] = {k: (None if not np.isfinite(float(s[k])) else round(float(s[k]), 6)) for k in KEEP}
                    row[period]["verdict"] = s["verdict"]
                    row[period]["pool_sharpe"] = round(res[period].info["pool_sharpe"], 6)
                    row[period]["pool_mdd"] = round(res[period].info["pool_mdd"], 6)
                c, who = max_corr.get(it["name"], (0.0, None))
                row["max_corr"], row["max_corr_with"] = round(c, 3), who
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已写入 {len(items)} 条日志。下一步：python leaderboard.py，并在 notes/round{rnd}.md 写反思。")


if __name__ == "__main__":
    main()
