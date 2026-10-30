"""Alpha101（Kakushadze, 2016, "101 Formulaic Alphas"）的运算子与可在本数据上计算的因子。

数据映射（课程数据没有成交股数、VWAP、市值、行业）：
    open/high/low/close  后复权价（时间序列运算不受除权影响）
    volume               成交额 / 当日典型价 (H+L+C)/3（原始价），即估算的成交股数
    vwap                 典型价 (H+L+C)/3（后复权）：没有逐笔成交时最常见的近似
    returns              收盘到收盘的后复权日收益
    adv{d}               过去 d 日平均成交额（元）
    cap、IndClass        没有：用到市值或行业中性化的因子（#36 #48 #56 #58 #59 #63 #67 #69 #70 #76 #79 #80 #82 #87 #89 #90 #91 #93 #97 #100）不实现

约定：窗口参数为小数时向下取整（与论文一致）；所有运算只用当日及以前的数据；停牌日为 NaN。
时间序列运算的窗口内有效值少于 80% 时结果为 NaN。
公式按论文附录逐条转写，个别因子的取值方向保持原样（论文的方向即「值越大越看多」）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bt.factor_eval import ffill, roll_mean

# ====================================================================== 运算子
_W = lambda d: max(int(math.floor(d)), 1)


def rank(x):
    return pd.DataFrame(x).rank(axis=1, pct=True).to_numpy()


def delay(x, d):
    d = _W(d)
    out = np.full_like(x, np.nan, dtype=float)
    out[d:] = x[:-d]
    return out


def delta(x, d):
    return x - delay(x, d)


def ts_sum(x, d):
    d = _W(d)
    return roll_mean(x, d) * d


def sma(x, d):
    return roll_mean(x, _W(d))


def stddev(x, d):
    d = _W(d)
    m = roll_mean(x, d)
    return np.sqrt(np.maximum(roll_mean(x * x, d) - m * m, 0))


def _pair(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.where(ok, x, np.nan), np.where(ok, y, np.nan)


def covariance(x, y, d):
    d = _W(d)
    x, y = _pair(x, y)
    return roll_mean(x * y, d) - roll_mean(x, d) * roll_mean(y, d)


def correlation(x, y, d):
    d = _W(d)
    x, y = _pair(x, y)
    mx, my = roll_mean(x, d), roll_mean(y, d)
    cov = roll_mean(x * y, d) - mx * my
    vx = roll_mean(x * x, d) - mx * mx
    vy = roll_mean(y * y, d) - my * my
    with np.errstate(invalid="ignore", divide="ignore"):
        c = cov / np.sqrt(vx * vy)
    c[~np.isfinite(c)] = np.nan
    return np.clip(c, -1, 1)


def _roll(x, d, how):
    d = _W(d)
    r = pd.DataFrame(x).rolling(d, min_periods=int(math.ceil(d * 0.8)))
    return getattr(r, how)().to_numpy()


def ts_min(x, d):
    return _roll(x, d, "min")


def ts_max(x, d):
    return _roll(x, d, "max")


def _ts_arg(x, d, fn):
    """窗口内最大 / 最小值出现的位置，1 = 最早一天，d = 今天（并列取较近的一天）。"""
    d = _W(d)
    best = np.full_like(x, -np.inf if fn == "max" else np.inf, dtype=float)
    pos = np.full_like(x, np.nan, dtype=float)
    cnt = np.zeros_like(x, dtype=float)
    for k in range(d - 1, -1, -1):                 # 从最早一天到今天
        y = delay(x, k) if k else x
        ok = np.isfinite(y)
        better = ok & ((y >= best) if fn == "max" else (y <= best))
        best = np.where(better, y, best)
        pos = np.where(better, d - k, pos)
        cnt += ok
    return np.where(cnt >= math.ceil(d * 0.8), pos, np.nan)


def ts_argmax(x, d):
    return _ts_arg(x, d, "max")


def ts_argmin(x, d):
    return _ts_arg(x, d, "min")


def ts_rank(x, d):
    """今天的值在过去 d 日中的排名，归一化到 (0, 1]。"""
    d = _W(d)
    less = np.zeros_like(x, dtype=float)
    eq = np.zeros_like(x, dtype=float)
    cnt = np.zeros_like(x, dtype=float)
    for k in range(d):
        y = delay(x, k) if k else x
        ok = np.isfinite(y)
        less += ok & (y < x)
        eq += ok & (y == x)
        cnt += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (less + (eq + 1) / 2) / cnt
    return np.where(np.isfinite(x) & (cnt >= math.ceil(d * 0.8)), r, np.nan)


def decay_linear(x, d):
    d = _W(d)
    num = np.zeros_like(x, dtype=float)
    den = np.zeros_like(x, dtype=float)
    for k in range(d):
        y = delay(x, k) if k else x
        w = d - k
        ok = np.isfinite(y)
        num += np.where(ok, y * w, 0.0)
        den += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den >= (d * (d + 1) / 2) * 0.8, out, np.nan)


def scale(x, a=1.0):
    s = np.nansum(np.abs(x), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return a * x / s


def signedpower(x, a):
    return np.sign(x) * np.abs(x) ** a


def product(x, d):
    d = _W(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp(ts_sum(np.log(x), d))


def log(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x)


def where(c, a, b):
    out = np.where(c, a, b).astype(float)
    return out


# ====================================================================== 输入
_CACHE: dict = {}


def inputs(d):
    key = (id(d), d.close.shape)
    if key in _CACHE:
        return _CACHE[key]
    tr = d.traded
    nanize = lambda a: np.where(tr, a, np.nan)
    O, H, L, C = (nanize(getattr(d, f) * d.adj) for f in ("open", "high", "low", "close"))
    typ_raw = (d.high + d.low + d.close) / 3
    amt = nanize(d.amt)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol = amt / typ_raw
    Cf = ffill(C)
    ret = np.full_like(C, np.nan)
    ret[1:] = np.where(tr[1:], Cf[1:] / Cf[:-1] - 1, np.nan)
    x = {"open": O, "high": H, "low": L, "close": C, "volume": vol, "vwap": (H + L + C) / 3,
         "returns": ret, "amt": amt}
    _CACHE.clear()
    _CACHE[key] = x
    return x


def adv(x, n):
    return sma(x["amt"], n)


# ====================================================================== 因子
def a001(x):
    r, c = x["returns"], x["close"]
    return rank(ts_argmax(signedpower(where(r < 0, stddev(r, 20), c), 2.0), 5)) - 0.5


def a002(x):
    return -correlation(rank(delta(log(x["volume"]), 2)), rank((x["close"] - x["open"]) / x["open"]), 6)


def a003(x):
    return -correlation(rank(x["open"]), rank(x["volume"]), 10)


def a004(x):
    return -ts_rank(rank(x["low"]), 9)


def a005(x):
    v = x["vwap"]
    return rank(x["open"] - ts_sum(v, 10) / 10) * (-np.abs(rank(x["close"] - v)))


def a006(x):
    return -correlation(x["open"], x["volume"], 10)


def a007(x):
    c = x["close"]
    d7 = delta(c, 7)
    return where(adv(x, 20) < x["amt"], -ts_rank(np.abs(d7), 60) * np.sign(d7), -1.0)


def a008(x):
    s = ts_sum(x["open"], 5) * ts_sum(x["returns"], 5)
    return -rank(s - delay(s, 10))


def a009(x):
    d1 = delta(x["close"], 1)
    return where(ts_min(d1, 5) > 0, d1, where(ts_max(d1, 5) < 0, d1, -d1))


def a010(x):
    d1 = delta(x["close"], 1)
    return rank(where(ts_min(d1, 4) > 0, d1, where(ts_max(d1, 4) < 0, d1, -d1)))


def a011(x):
    g = x["vwap"] - x["close"]
    return (rank(ts_max(g, 3)) + rank(ts_min(g, 3))) * rank(delta(x["volume"], 3))


def a012(x):
    return np.sign(delta(x["volume"], 1)) * (-delta(x["close"], 1))


def a013(x):
    return -rank(covariance(rank(x["close"]), rank(x["volume"]), 5))


def a014(x):
    return -rank(delta(x["returns"], 3)) * correlation(x["open"], x["volume"], 10)


def a015(x):
    return -ts_sum(rank(correlation(rank(x["high"]), rank(x["volume"]), 3)), 3)


def a016(x):
    return -rank(covariance(rank(x["high"]), rank(x["volume"]), 5))


def a017(x):
    c = x["close"]
    return (-rank(ts_rank(c, 10))) * rank(delta(delta(c, 1), 1)) * rank(ts_rank(x["amt"] / adv(x, 20), 5))


def a018(x):
    c, o = x["close"], x["open"]
    return -rank(stddev(np.abs(c - o), 5) + (c - o) + correlation(c, o, 10))


def a019(x):
    c = x["close"]
    return -np.sign((c - delay(c, 7)) + delta(c, 7)) * (1 + rank(1 + ts_sum(x["returns"], 250)))


def a020(x):
    o = x["open"]
    return -rank(o - delay(x["high"], 1)) * rank(o - delay(x["close"], 1)) * rank(o - delay(x["low"], 1))


def a022(x):
    return -(delta(correlation(x["high"], x["volume"], 5), 5) * rank(stddev(x["close"], 20)))


def a023(x):
    h = x["high"]
    return where(ts_sum(h, 20) / 20 < h, -delta(h, 2), 0.0)


def a024(x):
    c = x["close"]
    cond = delta(ts_sum(c, 100) / 100, 100) / delay(c, 100) <= 0.05
    return where(cond, -(c - ts_min(c, 100)), -delta(c, 3))


def a025(x):
    return rank(((-x["returns"]) * adv(x, 20)) * x["vwap"] * (x["high"] - x["close"]))


def a026(x):
    return -ts_max(correlation(ts_rank(x["volume"], 5), ts_rank(x["high"], 5), 5), 3)


def a028(x):
    return scale(correlation(adv(x, 20), x["low"], 5) + (x["high"] + x["low"]) / 2 - x["close"])


def a030(x):
    c = x["close"]
    s = np.sign(c - delay(c, 1)) + np.sign(delay(c, 1) - delay(c, 2)) + np.sign(delay(c, 2) - delay(c, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1.0 - rank(s)) * ts_sum(x["volume"], 5) / ts_sum(x["volume"], 20)


def a033(x):
    with np.errstate(invalid="ignore", divide="ignore"):
        return rank(-(1 - x["open"] / x["close"]))


def a034(x):
    r = x["returns"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return rank((1 - rank(stddev(r, 2) / stddev(r, 5))) + (1 - rank(delta(x["close"], 1))))


def a035(x):
    return ts_rank(x["volume"], 32) * (1 - ts_rank(x["close"] + x["high"] - x["low"], 16)) * (1 - ts_rank(x["returns"], 32))


def a037(x):
    oc = x["open"] - x["close"]
    return rank(correlation(delay(oc, 1), x["close"], 200)) + rank(oc)


def a038(x):
    with np.errstate(invalid="ignore", divide="ignore"):
        return -rank(ts_rank(x["close"], 10)) * rank(x["close"] / x["open"])


def a039(x):
    with np.errstate(invalid="ignore", divide="ignore"):
        dv = decay_linear(x["amt"] / adv(x, 20), 9)
    return -rank(delta(x["close"], 7) * (1 - rank(dv))) * (1 + rank(ts_sum(x["returns"], 250)))


def a040(x):
    return -rank(stddev(x["high"], 10)) * correlation(x["high"], x["volume"], 10)


def a041(x):
    return np.sqrt(x["high"] * x["low"]) - x["vwap"]


def a042(x):
    v, c = x["vwap"], x["close"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return rank(v - c) / rank(v + c)


def a043(x):
    with np.errstate(invalid="ignore", divide="ignore"):
        return ts_rank(x["amt"] / adv(x, 20), 20) * ts_rank(-delta(x["close"], 7), 8)


def a044(x):
    return -correlation(x["high"], rank(x["volume"]), 5)


def a045(x):
    c = x["close"]
    return -(rank(ts_sum(delay(c, 5), 20) / 20) * correlation(c, x["volume"], 2) * rank(correlation(ts_sum(c, 5), ts_sum(c, 20), 2)))


def _slope_gap(c):
    return (delay(c, 20) - delay(c, 10)) / 10 - (delay(c, 10) - c) / 10


def a046(x):
    c = x["close"]
    g = _slope_gap(c)
    return where(g > 0.25, -1.0, where(g < 0, 1.0, -(c - delay(c, 1))))


def a047(x):
    c, h, v = x["close"], x["high"], x["vwap"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rank(1 / c) * x["volume"] / adv(x, 20)) * (h * rank(h - c) / (ts_sum(h, 5) / 5)) - rank(v - delay(v, 5))


def a049(x):
    c = x["close"]
    return where(_slope_gap(c) < -0.1, 1.0, -(c - delay(c, 1)))


def a050(x):
    return -ts_max(rank(correlation(rank(x["volume"]), rank(x["vwap"]), 5)), 5)


def a051(x):
    c = x["close"]
    return where(_slope_gap(c) < -0.05, 1.0, -(c - delay(c, 1)))


def a052(x):
    r = x["returns"]
    return (-ts_min(x["low"], 5) + delay(ts_min(x["low"], 5), 5)) * rank((ts_sum(r, 240) - ts_sum(r, 20)) / 220) * ts_rank(x["volume"], 5)


def a053(x):
    c, l, h = x["close"], x["low"], x["high"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return -delta(((c - l) - (h - c)) / (c - l), 9)


def a054(x):
    c, l, h, o = x["close"], x["low"], x["high"], x["open"]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return (-(l - c) * o ** 5) / ((l - h) * c ** 5)


def a055(x):
    c, l, h = x["close"], x["low"], x["high"]
    with np.errstate(invalid="ignore", divide="ignore"):
        k = (c - ts_min(l, 12)) / (ts_max(h, 12) - ts_min(l, 12))
    return -correlation(rank(k), rank(x["volume"]), 6)


def a057(x):
    with np.errstate(invalid="ignore", divide="ignore"):
        return -((x["close"] - x["vwap"]) / decay_linear(rank(ts_argmax(x["close"], 30)), 2))


def a060(x):
    c, l, h = x["close"], x["low"], x["high"]
    with np.errstate(invalid="ignore", divide="ignore"):
        clv = ((c - l) - (h - c)) / (h - l) * x["volume"]
    return -(2 * scale(rank(clv)) - scale(rank(ts_argmax(c, 10))))


def a101(x):
    return (x["close"] - x["open"]) / ((x["high"] - x["low"]) + 0.001)


ALPHAS = {f"A{name[1:]}": fn for name, fn in sorted(globals().items()) if name.startswith("a") and name[1:].isdigit()}


def factor_fn(key):
    fn = ALPHAS[key]
    return lambda d: fn(inputs(d))
