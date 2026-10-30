"""为第 0–3 轮的因子补算相关性缓存（只读日志，不改日志）。"""
from common import LEDGER, data, load_factor, read_jsonl
from run_round import rank_sample

d = data()
for r in read_jsonl(LEDGER):
    if r.get("lookahead_ok"):
        n, f, _ = load_factor(r["source"])
        rank_sample(d, n, f)
        print("cached", n, flush=True)
