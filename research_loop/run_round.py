"""跑一轮：前视检查 → 训练期、验证期评价 → 追加写入研究日志。

    python run_round.py <轮次> builtin:REV5 factors/xxx.py alpha101:A012 ...

每个因子只计算一次原始值，供两个时段的评价和相关性计算共用；横截面排名样本缓存在 cache/，用于和更早的因子比较相关性。
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from common import CACHE, LEDGER, data, eval_cfg, load_factor, read_jsonl
from bt.factor_eval import Factor, corr_rows, evaluate, factor_ranks, lookahead_check

KEEP = ["rankic_mean", "icir_ann", "t_nw", "monotonic", "top_ex_bp", "bottom_ex_bp", "top_turnover",
        "bt_ann", "bt_sharpe", "bt_mdd", "bt_excess_ann", "bt_ir", "bt_turnover"]
CHAMPIONS = ("EXT5", "IVOL60", "LOWVOL60")          # 第 0–3 轮各目标的冠军


def sample_rows(d):
    i0 = int(d.dates.searchsorted(np.datetime64("2023-01-03")))
    return np.arange(i0, i0 + 400, 20)               # 验证期内每 20 日一个截面


def rank_sample(d, name, obj):
    """obj：Factor 或原始值数组。返回验证期抽样截面上的排名（缓存）。"""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{name}.npy"
    if p.exists():
        return np.load(p)
    rk = factor_ranks(d, {name: obj}, d.traded & ~d.st)[name][sample_rows(d)].astype(np.float32)
    np.save(p, rk)
    return rk


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
        code = Path(p).read_text() if p.startswith("factors/") else p
        items.append({"name": name, "source": p, "factor": f, "hypothesis": hyp, "lookahead_ok": ok,
                      "code_sha": hashlib.sha1(code.encode()).hexdigest()[:10]})
        print(f"{name}: 前视检查 {'通过' if ok else '未通过（最大差 %.3g）' % diff}", flush=True)
    good = [it for it in items if it["lookahead_ok"]]
    # 每个因子只算一次原始值（带排名后变换的内建因子保留 Factor 对象）
    objs = {}
    for it in good:
        f = it["factor"]
        objs[it["name"]] = f if f.post_rank else np.asarray(f.fn(d), dtype=float)
    res = {}
    for period in ("train", "valid"):
        if objs:
            print(f"评价 {period} …", flush=True)
            res[period] = evaluate(d, objs, eval_cfg(period))
    # 与更早因子的相关性
    earlier = {}
    for r in read_jsonl(LEDGER):
        if r.get("lookahead_ok"):
            n2, f2, _ = load_factor(r["source"])
            earlier[n2] = rank_sample(d, n2, f2)
    champ = {k: earlier[k] for k in CHAMPIONS if k in earlier}
    stats = {}
    for it in good:
        k = it["name"]
        rk = rank_sample(d, k, objs[k])
        best, who = 0.0, None
        for k2, rk2 in earlier.items():
            c = float(np.nanmean(corr_rows(rk, rk2)))
            if abs(c) > abs(best):
                best, who = c, k2
        cc = max([abs(float(np.nanmean(corr_rows(rk, v)))) for k2, v in champ.items() if k2 != k] or [0.0])
        stats[k] = (best, who, cc)
        earlier[k] = rk
    with LEDGER.open("a") as fh:
        for it in items:
            row = {"round": rnd, "time": dt.datetime.now().isoformat(timespec="seconds"), "name": it["name"],
                   "source": it["source"], "hypothesis": it["hypothesis"], "code_sha": it["code_sha"],
                   "lookahead_ok": it["lookahead_ok"]}
            if it["lookahead_ok"]:
                for period in ("train", "valid"):
                    s = res[period].summary.loc[it["name"]]
                    row[period] = {k: (None if not np.isfinite(float(s[k])) else round(float(s[k]), 6)) for k in KEEP}
                    row[period]["verdict"] = s["verdict"]
                    row[period]["pool_sharpe"] = round(float(res[period].info["pool_sharpe"]), 6)
                    row[period]["pool_mdd"] = round(float(res[period].info["pool_mdd"]), 6)
                best, who, cc = stats[it["name"]]
                row["max_corr"], row["max_corr_with"], row["corr_champions"] = round(best, 3), who, round(cc, 3)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已写入 {len(items)} 条日志。下一步：python leaderboard.py，并在 notes/round{rnd}.md 写反思。")


if __name__ == "__main__":
    main()
