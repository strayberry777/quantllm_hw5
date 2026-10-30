"""因子评价。

因子 = 每个交易日给每只股票打的一个分数（第 t 日收盘用 ≤ t 的数据计算），方向约定为「值越大，预期收益越高」。
评价回答六个问题：有没有预测力（IC / RankIC）、稳不稳（ICIR、t 值、分年）、是否单调（分组收益）、
能不能赚钱（多空与做多回测）、贵不贵（换手、衰减）、是否重复（因子相关性）。

远期收益：第 t 日收盘出信号，第 t+1 日开盘买入，持有 h 个交易日到第 t+1+h 日开盘（后复权）。
第 t+1 日开盘停牌或封涨停（买不进）的股票不参与当天评价，只评价真能买到的股票。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import strategies as S
from .data import FIELDS, MarketData
from .engine import Backtester, Costs
from .metrics import summarize

HORIZONS = (1, 2, 3, 5, 10, 20)


# ====================================================================== 工具函数（也提供给自定义因子）
def ffill(a: np.ndarray) -> np.ndarray:
    return S._ffill(a)


def roll_mean(a: np.ndarray, w: int, min_frac: float = 0.8) -> np.ndarray:
    """滚动均值（忽略 NaN；窗口内有效值少于 80% 时为 NaN）。"""
    x = np.nan_to_num(a)
    cs = np.cumsum(x, axis=0)
    cnt = np.cumsum(np.isfinite(a), axis=0).astype(float)
    s, n = cs.copy(), cnt.copy()
    s[w:] = cs[w:] - cs[:-w]
    n[w:] = cnt[w:] - cnt[:-w]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n >= w * min_frac, s / np.maximum(n, 1), np.nan)
    out[: w - 1] = np.nan
    return out


def roll_std(a: np.ndarray, w: int) -> np.ndarray:
    m = roll_mean(a, w)
    m2 = roll_mean(a * a, w)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def roll_max(a: np.ndarray, w: int) -> np.ndarray:
    out = pd.DataFrame(a).rolling(w, min_periods=int(np.ceil(w * 0.8))).max().to_numpy()
    return out


def daily_return(d: MarketData) -> np.ndarray:
    C = ffill(np.where(d.traded, d.adj_close, np.nan))
    r = np.full_like(C, np.nan)
    r[1:] = np.where(d.traded[1:], C[1:] / C[:-1] - 1, np.nan)
    return r


def past_return(d: MarketData, n: int) -> np.ndarray:
    return S.past_return(d, n)


HELPERS = {"np": np, "pd": pd, "ffill": ffill, "roll_mean": roll_mean, "roll_std": roll_std,
           "roll_max": roll_max, "daily_return": daily_return, "past_return": past_return}


# ====================================================================== 内建因子
@dataclass
class Factor:
    key: str
    name: str
    desc: str
    fn: object                   # fn(d) -> T×N 数组
    post_rank: bool = False      # True：在股票池内排名之后再做变换（如极端度）


def _ext5(d):
    return past_return(d, 5)


BUILTIN = {
    "REV5": Factor("REV5", "5 日反转（5DR）", "−(C_t / C_{t−5} − 1)：过去 5 日跌得越多，因子值越高",
                   lambda d: -past_return(d, 5)),
    "EXT5": Factor("EXT5", "5 日极端度（剔除过热股）", "−|rank(5 日收益) − 0.5|：过去 5 日涨跌越接近中间，因子值越高",
                   _ext5, post_rank=True),
    "MOM20": Factor("MOM20", "20 日动量（跳过近 5 日）", "C_{t−5} / C_{t−20} − 1：跳过最近 5 日，避开短期反转",
                    lambda d: (1 + past_return(d, 20)) / (1 + past_return(d, 5)) - 1),
    "LOWVOL20": Factor("LOWVOL20", "20 日低波动", "−std(过去 20 日日收益)",
                       lambda d: -roll_std(daily_return(d), 20)),
    "ILLIQ20": Factor("ILLIQ20", "20 日低成交额", "−log(过去 20 日日均成交额)；数据没有市值，用成交额近似「小而冷门」",
                      lambda d: -np.log(roll_mean(np.where(d.traded, d.amt, np.nan), 20))),
    "MAX20": Factor("MAX20", "20 日最大涨幅（反向）", "−max(过去 20 日单日收益)",
                    lambda d: -roll_max(daily_return(d), 20)),
}

CUSTOM_TEMPLATE = '''# 自定义因子：写一个 factor(d) 函数，返回「交易日 × 股票」的数组（与 d.close 同形状）
# 约定：值越大 = 预期收益越高；第 t 行只能用第 t 日收盘及以前的数据
# 可用：d.adj_close / d.adj_open / d.close / d.amt / d.traded（布尔）等，
#       以及 past_return(d, n)、daily_return(d)、roll_mean(a, w)、roll_std(a, w)、roll_max(a, w)、ffill(a)、np、pd

def factor(d):
    # 例：20 日价格位置——收盘价在过去 20 日最高最低价区间中的位置，越低越好（反转思路）
    C = ffill(np.where(d.traded, d.adj_close, np.nan))
    hi = pd.DataFrame(C).rolling(20, min_periods=16).max().to_numpy()
    lo = pd.DataFrame(C).rolling(20, min_periods=16).min().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (C - lo) / (hi - lo)
    return -pos
'''


def compile_custom(code: str):
    ns = dict(HELPERS)
    exec(compile(code, "<自定义因子>", "exec"), ns)
    fn = ns.get("factor")
    if not callable(fn):
        raise ValueError("代码里需要定义 factor(d) 函数")
    return fn


# ====================================================================== 向量化统计
def xs_rank(a: np.ndarray, mask: np.ndarray, min_n: int = 50) -> np.ndarray:
    """逐日横截面百分位排名 (rank − 0.5) / n，只在 mask 内排名；有效股票少于 min_n 的日子为 NaN。"""
    x = np.where(mask & np.isfinite(a), a, np.nan)
    df = pd.DataFrame(x)
    r = df.rank(axis=1, method="average").to_numpy()
    n = np.isfinite(x).sum(1, keepdims=True).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (r - 0.5) / n
    out[(n[:, 0] < min_n)] = np.nan
    return out


def rank_rows(a: np.ndarray) -> np.ndarray:
    return pd.DataFrame(a).rank(axis=1, method="average").to_numpy()


def corr_rows(x: np.ndarray, y: np.ndarray, min_n: int = 50) -> np.ndarray:
    """逐行 Pearson 相关（只用两边都有值的格子）。"""
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(1).astype(float)
    x0 = np.where(ok, x, 0.0)
    y0 = np.where(ok, y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = x0.sum(1) / n
        my = y0.sum(1) / n
        cov = np.einsum("ij,ij->i", x0, y0) / n - mx * my
        vx = np.einsum("ij,ij->i", x0, x0) / n - mx * mx
        vy = np.einsum("ij,ij->i", y0, y0) / n - my * my
        c = cov / np.sqrt(vx * vy)
    c[n < min_n] = np.nan
    return c


def newey_west_t(x: np.ndarray, lag: int) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    v = (e @ e) / n
    for L in range(1, lag + 1):
        v += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(v / n))


def verdict(r) -> str:
    if not np.isfinite(r["t_nw"]) or abs(r["t_nw"]) < 2:
        return "无预测力"
    if r["rankic_mean"] < 0:
        return "方向相反"
    if np.isfinite(r.get("bt_excess_ann", np.nan)) and r["bt_excess_ann"] > 0.02 and r["bt_ir"] > 0.5:
        return "可交易"
    if r["monotonic"] > 0.8:
        return "有效，但做多端弱"
    return "有效，但不单调"


# ====================================================================== 评价主函数
@dataclass
class EvalConfig:
    start: str = "2020-01-02"
    end: str | None = None
    horizon: int = 5               # 主评价持有期（日）
    groups: int = 10
    rebalance: int = 5             # 分组与回测的调仓间隔
    exclude_st: bool = True
    min_listed: int = 60
    liq_drop: float = 0.2
    backtest: bool = True
    top_frac: float = 0.1
    capital: float = 1e8
    costs: Costs = field(default_factory=lambda: Costs(commission=0.0001, min_commission=0.0, transfer=0.0,
                                                       stamp_old=0.0005, stamp_new=0.0005, slippage=0.0005))
    max_participation: float | None = 0.1


@dataclass
class EvalResult:
    summary: pd.DataFrame          # 每个因子一行
    deciles: pd.DataFrame          # 组号 × 因子（bp / 持有期，相对股票池平均）
    decay: pd.DataFrame            # 持有期 × 因子（平均 RankIC）
    yearly: pd.DataFrame           # 年 × 因子（平均 RankIC）
    ic: pd.DataFrame               # 日期 × 因子（主持有期 RankIC）
    corr: pd.DataFrame             # 因子 × 因子
    ls_nav: pd.DataFrame           # 日期 × 因子（最高组 − 最低组的累计净值，不可做空，仅衡量强度）
    nav: pd.DataFrame | None       # 日期 × 因子 + POOL（做多最高组的回测净值）
    info: dict


def _eval_universe(d: MarketData, cfg: EvalConfig):
    pool = S.universe(d, min_age=cfg.min_listed, liq_drop=cfg.liq_drop, exclude_st=cfg.exclude_st)
    buyable = np.zeros_like(pool)
    buyable[:-1] = d.traded[1:] & ~d.buy_blocked[1:]
    return pool, pool & buyable


def forward_returns(d: MarketData, mask: np.ndarray, horizons=HORIZONS) -> dict:
    T = d.close.shape[0]
    C = ffill(np.where(d.traded, d.adj_close, np.nan))
    O = np.where(d.traded, d.adj_open, C)
    out = {}
    for h in horizons:
        r = np.full(O.shape, np.nan)
        if T - 1 - h > 0:
            r[: T - 1 - h] = O[1 + h:] / O[1:T - h] - 1
        out[h] = np.where(mask, r, np.nan)
    return out


def factor_ranks(d: MarketData, factors: dict, mask: np.ndarray) -> dict:
    """factors: {key: Factor 或 T×N 数组}。返回股票池内的百分位排名。"""
    ranks = {}
    for k, f in factors.items():
        if isinstance(f, Factor):
            raw = np.asarray(f.fn(d), dtype=float)
            r = xs_rank(raw, mask)
            if f.post_rank:                                      # 极端度：离中位越远越差
                r = xs_rank(-np.abs(r - 0.5), mask)
        else:
            raw = np.asarray(f, dtype=float)
            r = xs_rank(raw, mask)
        if raw.shape != d.close.shape:
            raise ValueError(f"因子 {k} 的形状 {raw.shape} 与数据 {d.close.shape} 不一致")
        ranks[k] = r
    return ranks


def evaluate(d: MarketData, factors: dict, cfg: EvalConfig | None = None, progress=None) -> EvalResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _evaluate(d, factors, cfg, progress)


def _evaluate(d: MarketData, factors: dict, cfg: EvalConfig | None = None, progress=None) -> EvalResult:
    cfg = cfg or EvalConfig()
    T, N = d.close.shape
    names = {k: (f.name if isinstance(f, Factor) else k) for k, f in factors.items()}
    descs = {k: (f.desc if isinstance(f, Factor) else "自定义因子") for k, f in factors.items()}
    pool, mask = _eval_universe(d, cfg)
    hs = sorted(set(HORIZONS) | {cfg.horizon})
    i0 = max(int(d.dates.searchsorted(pd.Timestamp(cfg.start))), 60)
    i1 = T if cfg.end is None else int(d.dates.searchsorted(pd.Timestamp(cfg.end), side="right"))
    last_eval = min(i1, T - 1 - max(hs)) - 1          # 所有持有期的远期收益都可得的最后一天
    if last_eval - i0 < 40:
        raise ValueError("评价区间太短：至少需要约 40 个交易日，并且结束日之后要留出 20 个交易日计算远期收益")
    tt = np.arange(i0, last_eval + 1)

    step = lambda p, msg: progress(p, msg) if progress else None
    step(0.05, "计算远期收益")
    fwd = forward_returns(d, mask, hs)
    fwd_rank = {h: rank_rows(fwd[h][tt]) for h in hs}
    step(0.15, "计算因子并排名")
    F = factor_ranks(d, factors, mask)

    rows, decay, yearly, ic_series = [], {}, {}, {}
    H = cfg.horizon
    for j, (k, f) in enumerate(F.items()):
        step(0.2 + 0.35 * j / max(len(F), 1), f"统计 {names[k]}")
        fr = rank_rows(f[tt])
        for h in hs:
            ric = corr_rows(fr, fwd_rank[h])
            decay.setdefault(k, {})[h] = float(np.nanmean(ric))
            if h == H:
                s = pd.Series(ric, index=d.dates[tt])
                ic_series[k] = s
                ic_pearson = corr_rows(f[tt], fwd[h][tt])
                sub = s.iloc[::H].dropna()
                yearly[k] = s.groupby(s.index.year).mean()
                rows.append({"factor": k, "name": names[k], "definition": descs[k],
                             "ic_mean": float(np.nanmean(ic_pearson)), "rankic_mean": float(s.mean()),
                             "rankic_std": float(s.std()),
                             "icir_ann": float(sub.mean() / sub.std() * np.sqrt(244 / H)) if len(sub) > 2 else np.nan,
                             "t_nw": newey_west_t(ric, max(H - 1, 1)), "ic_pos_ratio": float((s.dropna() > 0).mean())})
    summ = pd.DataFrame(rows).set_index("factor")

    # 分组（不重叠，每 rebalance 天一次，用主持有期的远期收益）
    step(0.6, "分组收益与换手")
    G = cfg.groups
    reb = np.arange(i0, min(i1, T - 1 - H), cfg.rebalance)      # 只需要主持有期的远期收益可得
    groups, ls_nav = {}, {}
    for k, f in F.items():
        fr, rr = f[reb], fwd[H][reb]
        ok = np.isfinite(fr) & np.isfinite(rr)
        q = np.where(ok, np.minimum(np.floor(np.nan_to_num(fr) * G), G - 1), -1)
        base = np.nanmean(np.where(ok, rr, np.nan), 1)
        g = np.full((len(reb), G), np.nan)
        for j in range(G):
            sel = q == j
            with np.errstate(invalid="ignore"):
                g[:, j] = np.nanmean(np.where(sel, rr, np.nan), 1) - base
        gm = np.nanmean(g, 0)
        groups[k] = gm * 1e4
        ls = pd.Series(g[:, G - 1] - g[:, 0], index=d.dates[reb]).dropna()
        ls_nav[k] = (1 + ls).cumprod()
        top = q == G - 1
        prev = top[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            out_share = 1 - (top[1:] & prev).sum(1) / prev.sum(1)
        ac = corr_rows(f[reb[:-1]], f[reb[1:]])
        summ.loc[k, "monotonic"] = float(pd.Series(gm).corr(pd.Series(np.arange(G)), method="spearman"))
        summ.loc[k, "top_ex_bp"] = float(gm[-1] * 1e4)
        summ.loc[k, "bottom_ex_bp"] = float(gm[0] * 1e4)
        per_year = 244 / cfg.rebalance
        summ.loc[k, "ls_ann"] = float((1 + ls).prod() ** (per_year / max(len(ls), 1)) - 1) if len(ls) else np.nan
        summ.loc[k, "ls_sharpe"] = float(ls.mean() / ls.std() * np.sqrt(per_year)) if len(ls) > 2 else np.nan
        summ.loc[k, "autocorr"] = float(np.nanmean(ac))
        summ.loc[k, "top_turnover"] = float(np.nanmean(out_share))

    keys = list(F)
    corr = pd.DataFrame(index=keys, columns=keys, dtype=float)
    samp = reb[::4]
    for a in keys:
        for b in keys:
            corr.loc[a, b] = float(np.nanmean(corr_rows(F[a][samp], F[b][samp])))

    navs = None
    pool_nav = None
    if cfg.backtest:
        navs = {}
        end = i1
        kw = dict(capital=cfg.capital, band=0.0002, max_participation=cfg.max_participation)
        pw = S.decile_weights(np.where(pool, 0.0, np.nan), pool, 0.0, 1.0 + 1e-9, cfg.rebalance, i0)
        step(0.7, "回测：股票池等权")
        base = Backtester(d, cfg.costs, **kw).run(pw, start=i0, end=end)
        navs["POOL"] = base.nav / cfg.capital
        pool_nav = base.nav
        for j, (k, f) in enumerate(F.items()):
            step(0.72 + 0.26 * j / max(len(F), 1), f"回测：做多 {names[k]} 最高组")
            W = S.decile_weights(np.where(pool, f, np.nan), pool, 1 - cfg.top_frac, 1.0 + 1e-9, cfg.rebalance, i0)
            r = Backtester(d, cfg.costs, **kw).run(W, start=i0, end=end)
            navs[k] = r.nav / cfg.capital
            m = summarize(r.nav, bench=base.nav, turnover=r.daily.turnover, fees=r.daily.fees)
            summ.loc[k, "bt_ann"] = m["ann_return"]
            summ.loc[k, "bt_sharpe"] = m["sharpe"]
            summ.loc[k, "bt_mdd"] = m["max_drawdown"]
            summ.loc[k, "bt_excess_ann"] = m["excess_ann"]
            summ.loc[k, "bt_ir"] = m["info_ratio"]
            summ.loc[k, "bt_turnover"] = m["turnover_ann"]
            summ.loc[k, "bt_fee_drag"] = m["fee_drag_ann"]
        navs = pd.DataFrame(navs)
    summ["verdict"] = summ.apply(verdict, axis=1)
    step(1.0, "完成")
    info = {"start": str(d.dates[i0].date()), "end": str(d.dates[last_eval].date()), "days": int(len(tt)),
            "bt_end": str(d.dates[i1 - 1].date()),
            "pool_size": float(np.nanmean(mask[tt].sum(1))), "horizon": H, "groups": G, "rebalance": cfg.rebalance,
            "exclude_st": cfg.exclude_st}
    if pool_nav is not None:
        mb = summarize(pool_nav)
        info.update(pool_ann=mb["ann_return"], pool_sharpe=mb["sharpe"], pool_mdd=mb["max_drawdown"])
    return EvalResult(summary=summ, deciles=pd.DataFrame(groups, index=pd.RangeIndex(1, G + 1, name="group")),
                      decay=pd.DataFrame(decay).rename_axis("horizon"), yearly=pd.DataFrame(yearly).rename_axis("year"),
                      ic=pd.DataFrame(ic_series), corr=corr, ls_nav=pd.DataFrame(ls_nav), nav=navs, info=info)


# ====================================================================== 前视检查
def head(d: MarketData, k: int) -> MarketData:
    """只保留前 k 个交易日（派生量重新计算；起点相同，所以上市天数等不变）。"""
    md = MarketData(dates=d.dates[:k], symbols=d.symbols, **{f: getattr(d, f)[:k] for f in FIELDS})
    md._derive()
    return md


def lookahead_check(d: MarketData, fn, cut: int | None = None) -> tuple[bool, float]:
    """在第 cut 日把数据截断后重新计算因子：若第 cut−1 行与用全量数据算的不同，说明用到了未来数据。"""
    T = d.close.shape[0]
    cut = cut or int(T * 0.6)
    full = np.asarray(fn(d), dtype=float)[cut - 1]
    part = np.asarray(fn(head(d, cut)), dtype=float)[cut - 1]
    both = np.isfinite(full) & np.isfinite(part)
    same_nan = (~np.isfinite(full)) == (~np.isfinite(part))
    diff = float(np.nanmax(np.abs(full[both] - part[both]))) if both.any() else 0.0
    return bool(same_nan.all() and diff < 1e-9), diff


# ====================================================================== 单因子深度评价
@dataclass
class DeepDive:
    key: str
    name: str
    subsets: pd.DataFrame          # 分样本：板块 / 流动性 / 市场环境 → RankIC、t 值、天数
    legs: pd.DataFrame             # 日期 × [最高组, 最低组] 的累计超额净值（相对股票池平均）
    breakeven: dict                # 成本盈亏平衡
    sweep: pd.DataFrame | None     # 参数扫描（反转类因子的回看天数）
    rolling_ic: pd.Series          # 滚动 120 日平均 RankIC


def _subset_ic(f_raw_rank, fwd_h, mask_sub, tt, H, min_n=30):
    fr = rank_rows(np.where(mask_sub[tt], f_raw_rank[tt], np.nan))
    rr = rank_rows(np.where(mask_sub[tt], fwd_h[tt], np.nan))
    ric = corr_rows(fr, rr, min_n=min_n)
    s = ric[np.isfinite(ric)]
    return float(np.mean(s)) if len(s) else np.nan, newey_west_t(ric, max(H - 1, 1)), int(len(s)), ric


def _group_stats(fr, fwd_h, reb, G):
    rr = fwd_h[reb]
    f = fr[reb]
    ok = np.isfinite(f) & np.isfinite(rr)
    q = np.where(ok, np.minimum(np.floor(np.nan_to_num(f) * G), G - 1), -1)
    base = np.nanmean(np.where(ok, rr, np.nan), 1)
    g = np.full((len(reb), G), np.nan)
    for j in range(G):
        g[:, j] = np.nanmean(np.where(q == j, rr, np.nan), 1) - base
    top = q == G - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        out_share = 1 - (top[1:] & top[:-1]).sum(1) / top[:-1].sum(1)
    return g, float(np.nanmean(out_share))


def deep_dive(d: MarketData, key: str, factor, cfg: EvalConfig | None = None, progress=None) -> DeepDive:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _deep_dive(d, key, factor, cfg or EvalConfig(), progress)


def _deep_dive(d, key, factor, cfg, progress):
    step = lambda p, m: progress(p, m) if progress else None
    T = d.close.shape[0]
    H, G = cfg.horizon, cfg.groups
    pool, mask = _eval_universe(d, cfg)
    i0 = max(int(d.dates.searchsorted(pd.Timestamp(cfg.start))), 60)
    i1 = T if cfg.end is None else int(d.dates.searchsorted(pd.Timestamp(cfg.end), side="right"))
    last = min(i1, T - 1 - H) - 1
    tt = np.arange(i0, last + 1)
    reb = np.arange(i0, last + 1, cfg.rebalance)
    step(0.05, "计算因子")
    fwd = forward_returns(d, mask, (H,))[H]
    f = factor_ranks(d, {key: factor}, mask)[key]
    name = factor.name if isinstance(factor, Factor) else key

    rows = []
    # 全样本
    m, t, n, ric_all = _subset_ic(f, fwd, mask, tt, H, min_n=50)
    rows.append({"group": "全样本", "subset": "全部", "rankic": m, "t": t, "days": n})
    # 板块
    step(0.15, "按板块")
    boards = d.boards
    for b, lab in (("MAIN", "主板"), ("GEM", "创业板"), ("STAR", "科创板")):
        sub = mask & (boards == b)[None, :]
        m, t, n, _ = _subset_ic(f, fwd, sub, tt, H)
        rows.append({"group": "板块", "subset": lab, "rankic": m, "t": t, "days": n})
    # 流动性三分组（20 日均成交额，在股票池内排名）
    step(0.35, "按流动性")
    liq = xs_rank(roll_mean(np.where(d.traded, d.amt, np.nan), 20), mask)
    for lo, hi, lab in ((0, 1 / 3, "成交额低"), (1 / 3, 2 / 3, "成交额中"), (2 / 3, 1.01, "成交额高")):
        sub = mask & (liq >= lo) & (liq < hi)
        m, t, n, _ = _subset_ic(f, fwd, sub, tt, H)
        rows.append({"group": "流动性", "subset": lab, "rankic": m, "t": t, "days": n})
    # 市场环境：股票池等权过去 20 日收益的正负、过去 20 日波动的高低（都只用 ≤ t 的数据）
    step(0.55, "按市场环境")
    r1 = daily_return(d)
    mkt = np.nanmean(np.where(pool, r1, np.nan), 1)
    mkt20 = pd.Series(mkt).rolling(20, min_periods=15).sum().to_numpy()
    vol20 = pd.Series(mkt).rolling(20, min_periods=15).std().to_numpy()
    med_vol = np.nanmedian(vol20[tt])
    ric_s = pd.Series(ric_all, index=tt)
    for cond, g_, lab in ((mkt20[tt] > 0, "市场环境", "过去 20 日上涨"), (mkt20[tt] <= 0, "市场环境", "过去 20 日下跌"),
                          (vol20[tt] > med_vol, "市场环境", "高波动"), (vol20[tt] <= med_vol, "市场环境", "低波动")):
        x = ric_s.to_numpy()[cond]
        x = x[np.isfinite(x)]
        rows.append({"group": g_, "subset": lab, "rankic": float(x.mean()) if len(x) else np.nan,
                     "t": newey_west_t(x, max(H - 1, 1)), "days": int(len(x))})
    subsets = pd.DataFrame(rows)

    # 多空拆解与成本盈亏平衡
    step(0.7, "多空拆解与成本")
    g, turn = _group_stats(f, fwd, reb, G)
    legs = pd.DataFrame({"最高组": np.nancumprod(1 + np.nan_to_num(g[:, -1])),
                         "最低组": np.nancumprod(1 + np.nan_to_num(g[:, 0]))}, index=d.dates[reb])
    top_bp = float(np.nanmean(g[:, -1]) * 1e4)
    # 每次调仓换掉 turn 比例的持仓：卖出 turn、买入 turn，成本 = turn × 往返成本。超额刚好被吃光时的往返成本即盈亏平衡点
    be = top_bp / turn if turn > 0 and top_bp > 0 else np.nan
    rt = (cfg.costs.commission * 2 + cfg.costs.stamp_new + cfg.costs.slippage * 2 + cfg.costs.transfer * 2) * 1e4
    breakeven = {"top_ex_bp": top_bp, "turnover": turn, "breakeven_rt_bp": be, "cost_rt_bp": rt}

    # 参数扫描：反转类因子（REV*）的回看天数
    sweep = None
    if isinstance(factor, Factor) and key.startswith("REV"):
        out = []
        ns = (1, 2, 3, 5, 10, 20)
        for i, nlb in enumerate(ns):
            step(0.75 + 0.24 * i / len(ns), f"参数扫描：回看 {nlb} 日")
            fr = xs_rank(-past_return(d, nlb), mask)
            m, t, n, _ = _subset_ic(fr, fwd, mask, tt, H, min_n=50)
            gg, tv = _group_stats(fr, fwd, reb, G)
            gm = np.nanmean(gg, 0)
            out.append({"lookback": nlb, "rankic": m, "t": t, "monotonic": float(pd.Series(gm).corr(pd.Series(np.arange(G)), method="spearman")),
                        "top_ex_bp": float(gm[-1] * 1e4), "bottom_ex_bp": float(gm[0] * 1e4), "top_turnover": tv})
        sweep = pd.DataFrame(out).set_index("lookback")
    rolling = ric_s.rolling(120, min_periods=60).mean()
    rolling.index = d.dates[tt]
    step(1.0, "完成")
    return DeepDive(key=key, name=name, subsets=subsets, legs=legs, breakeven=breakeven, sweep=sweep, rolling_ic=rolling)
