"""因子评价：6 个内建因子，全样本 2020–2026，非 ST 股票池。结果写到 outputs/factors/（与网页平台「因子评价」页用同一个模块）。"""
import json

from _common import OUT, RAW
from bt import MarketData
from bt.factor_eval import BUILTIN, EvalConfig, evaluate

out = OUT / "factors"
out.mkdir(parents=True, exist_ok=True)
d = MarketData.load(RAW)
res = evaluate(d, BUILTIN, EvalConfig(), progress=lambda p, m: print(f"{p:4.0%} {m}", flush=True))
res.summary.to_csv(out / "summary.csv", float_format="%.6f")
res.deciles.to_csv(out / "groups_bp.csv", float_format="%.3f")
res.decay.to_csv(out / "decay.csv", float_format="%.5f")
res.yearly.to_csv(out / "yearly_rankic.csv", float_format="%.5f")
res.corr.to_csv(out / "corr.csv", float_format="%.3f")
res.ic.to_csv(out / "rankic_daily.csv", float_format="%.5f")
res.nav.to_csv(out / "topgroup_nav.csv", float_format="%.5f")
(out / "info.json").write_text(json.dumps(res.info, ensure_ascii=False, indent=1))
cols = ["verdict", "rankic_mean", "icir_ann", "t_nw", "monotonic", "top_ex_bp", "bottom_ex_bp", "top_turnover", "bt_excess_ann", "bt_ir"]
print(res.summary[cols].round(4).to_string())
