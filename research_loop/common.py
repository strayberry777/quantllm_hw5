import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from bt import Costs, MarketData                                   # noqa: E402
from bt.factor_eval import BUILTIN, HELPERS, EvalConfig, Factor     # noqa: E402

CFG = json.loads((HERE / "config.json").read_text())
LEDGER = HERE / "ledger.jsonl"
EXAMS = HERE / "exams.jsonl"
_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = MarketData.load(ROOT / "raw")
    return _DATA


def eval_cfg(period: str, backtest: bool = True) -> EvalConfig:
    p, e = CFG["periods"][period], CFG["eval"]
    return EvalConfig(start=p["start"], end=p["end"], horizon=e["horizon"], groups=e["groups"],
                      rebalance=e["rebalance"], exclude_st=e["exclude_st"], backtest=backtest,
                      top_frac=e["top_frac"], capital=e["capital"],
                      costs=Costs(commission=e["commission"], min_commission=0.0, transfer=0.0,
                                  stamp_old=e["stamp"], stamp_new=e["stamp"], slippage=e["slippage"]),
                      max_participation=e["max_participation"])


def load_factor(path: str) -> tuple[str, Factor, str]:
    """读取 factors/<名字>.py 或 builtin:<KEY>。返回 (名字, Factor, 假设)。"""
    if path.startswith("builtin:"):
        k = path.split(":", 1)[1]
        return k, BUILTIN[k], f"内建基准：{BUILTIN[k].desc}"
    code = Path(path).read_text()
    ns = dict(HELPERS)
    exec(compile(code, path, "exec"), ns)
    name = Path(path).stem
    return name, Factor(name, name, ns.get("HYPOTHESIS", ""), ns["factor"]), ns.get("HYPOTHESIS", "")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
