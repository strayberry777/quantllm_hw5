"""最终考试：在留出期（2025-01-02 至 2026-09-30）上评价指定因子，每个因子只能考一次。

    python final_exam.py <因子名> [<因子名> ...] --why "选中理由"
"""
import datetime as dt
import json
import sys

from common import EXAMS, LEDGER, data, eval_cfg, load_factor, read_jsonl
from bt.factor_eval import evaluate


def main():
    args = sys.argv[1:]
    why = ""
    if "--why" in args:
        i = args.index("--why")
        why = args[i + 1]
        args = args[:i] + args[i + 2:]
    taken = {e["name"] for e in read_jsonl(EXAMS)}
    src = {r["name"]: r["source"] for r in read_jsonl(LEDGER)}
    todo = []
    for n in args:
        if n in taken:
            sys.exit(f"{n} 已经考过一次，不能重考。")
        if n not in src:
            sys.exit(f"{n} 不在研究日志里：只有经过训练期、验证期评价的因子才能考试。")
        todo.append(n)
    factors = {}
    for n in todo:
        name, f, _ = load_factor(src[n])
        factors[name] = f
    res = evaluate(data(), factors, eval_cfg("holdout"))
    with EXAMS.open("a") as fh:
        for n in todo:
            s = res.summary.loc[n]
            row = {"time": dt.datetime.now().isoformat(timespec="seconds"), "name": n, "why": why,
                   "t_nw": round(float(s.t_nw), 4), "rankic": round(float(s.rankic_mean), 5),
                   "sharpe": round(float(s.bt_sharpe), 4), "mdd": round(float(s.bt_mdd), 4),
                   "ann": round(float(s.bt_ann), 4), "excess_ann": round(float(s.bt_excess_ann), 4),
                   "ir": round(float(s.bt_ir), 4), "turnover": round(float(s.bt_turnover), 2),
                   "pool_ann": round(res.info["pool_ann"], 4), "pool_sharpe": round(res.info["pool_sharpe"], 4),
                   "pool_mdd": round(res.info["pool_mdd"], 4)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row)
    res.nav.to_csv(f"exam_nav_{'_'.join(todo)}.csv", float_format="%.5f")


if __name__ == "__main__":
    main()
