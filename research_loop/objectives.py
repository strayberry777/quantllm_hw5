"""优化目标：都只用训练期 / 验证期的结果，越大越好。"""
import numpy as np
import pandas as pd


def table(rows: list[dict], period: str = "valid") -> pd.DataFrame:
    recs = []
    for r in rows:
        if not r.get("lookahead_ok") or period not in r:
            continue
        v = r[period]
        recs.append({"name": r["name"], "round": r["round"], "t_ic": v["t_nw"], "rankic": v["rankic_mean"],
                     "sharpe": v["bt_sharpe"], "mdd": v["bt_mdd"], "excess": v["bt_excess_ann"], "ann": v["bt_ann"],
                     "turnover": v["bt_turnover"], "top_ex_bp": v["top_ex_bp"], "monotonic": v["monotonic"],
                     "pool_sharpe": v["pool_sharpe"], "pool_mdd": v["pool_mdd"],
                     "max_corr": r.get("max_corr"), "dup": abs(r.get("max_corr") or 0) > 0.8})
    return pd.DataFrame(recs).set_index("name")


def z(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else x * 0


def scores(t: pd.DataFrame) -> pd.DataFrame:
    """四个目标。MIX = 0.5·z(夏普) + 0.5·z(−最大回撤)，z 分数在全部候选里计算。"""
    out = pd.DataFrame(index=t.index)
    out["IC"] = t["t_ic"]
    out["SHARPE"] = t["sharpe"]
    out["MDD"] = t["mdd"]                       # 回撤是负数，越接近 0 越好，所以直接越大越好
    out["MIX"] = 0.5 * z(t["sharpe"]) + 0.5 * z(t["mdd"])
    return out


def bonferroni_t(n_trials: int, alpha: float = 0.05) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha / 2 / max(n_trials, 1))


def novelty(rows: list[dict], t: pd.DataFrame) -> pd.Series:
    """与第 0–3 轮冠军（EXT5、IVOL60、LOWVOL60）的最大绝对相关。旧日志没有这一列时从缓存计算。"""
    import numpy as np
    from common import CACHE
    from bt.factor_eval import corr_rows
    champ = {k: np.load(CACHE / f"{k}.npy") for k in ("EXT5", "IVOL60", "LOWVOL60") if (CACHE / f"{k}.npy").exists()}
    out = {}
    for r in rows:
        k = r["name"]
        if k not in t.index:
            continue
        if r.get("corr_champions") is not None:
            out[k] = r["corr_champions"]
        elif (CACHE / f"{k}.npy").exists():
            rk = np.load(CACHE / f"{k}.npy")
            out[k] = max([abs(float(np.nanmean(corr_rows(rk, v)))) for k2, v in champ.items() if k2 != k] or [0.0])
    return pd.Series(out)


def scores_with_novelty(rows, t):
    s = scores(t)
    s["NOVEL"] = s["MIX"] - novelty(rows, t).reindex(t.index).fillna(0)
    return s
