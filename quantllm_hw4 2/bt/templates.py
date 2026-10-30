"""平台里的策略模板（下拉菜单的选项）。每个都是一段完整、可直接修改的代码。"""

REVERSAL_5D = '''from bt.api import Strategy


class FiveDayReversal(Strategy):
    """5DR：每 5 个交易日，买入过去 5 日
    跌得最多的 10% 股票，等权持有。
    全市场运行时初始资金建议设为 1 亿元（几百只股票，每只要买得起整手）。"""

    lookback = 5          # 看过去几天的涨跌
    rebalance_every = 5   # 每几个交易日调一次仓
    pick = 0.10           # 买入比例（跌最多的前 10%）
    min_listed = 60       # 上市满多少个交易日才考虑
    liq_drop = 0.20       # 剔除 20 日均成交额最低的 20%

    def on_bar(self, ctx):
        if ctx.bar_index % self.rebalance_every != 0:
            return None          # 非调仓日：不动

        close = ctx.history("close", self.lookback + 1).ffill()
        ret = close.iloc[-1] / close.iloc[0] - 1   # 过去 5 日收益
        amt = ctx.history("amt", 20).mean()

        ok = (ctx.listed_days >= self.min_listed) & (ctx.limit_status == 0) & ~ctx.is_st
        ok &= ret.notna() & close.iloc[-1].notna()
        ok &= amt >= amt[ok].quantile(self.liq_drop)
        ret = ret[ok]
        if len(ret) < 50:
            return None

        losers = ret.nsmallest(max(1, int(len(ret) * self.pick))).index
        return ctx.equal_weight(losers)
'''

MOMENTUM_5D = REVERSAL_5D.replace("FiveDayReversal", "FiveDayMomentum") \
    .replace("5DR：每 5 个交易日，买入过去 5 日\n    跌得最多的 10% 股票，等权持有。",
             "5 日动量（5DR 的镜像）：每 5 个交易日，买入过去 5 日\n    涨得最多的 10% 股票。") \
    .replace("# 买入比例（跌最多的前 10%）", "# 买入比例（涨最多的前 10%）") \
    .replace("losers = ret.nsmallest(", "losers = ret.nlargest(")

AVOID_EXTREMES = '''from bt.api import Strategy


class AvoidExtremes(Strategy):
    """剔除过热股：持有整个股票池（等权），但剔除过去 5 日
    涨得最多的 10% 和跌得最多的 10%。每 5 个交易日调仓。
    思路来自 5DR 的十分位结果：两头的股票之后都跑输中间。
    持有上千只股票，全市场运行时初始资金请设为 1 亿元。"""

    lookback = 5
    rebalance_every = 5
    cut_top = 0.10        # 剔除涨幅最高的比例
    cut_bottom = 0.10     # 剔除跌幅最大的比例
    min_listed = 60
    liq_drop = 0.20
    trade_band = 0.0002   # 调整金额 < 组合 0.02% 不交易，只换进出名单的股票

    def on_bar(self, ctx):
        if ctx.bar_index % self.rebalance_every != 0:
            return None

        close = ctx.history("close", self.lookback + 1).ffill()
        ret = close.iloc[-1] / close.iloc[0] - 1
        amt = ctx.history("amt", 20).mean()

        ok = (ctx.listed_days >= self.min_listed) & (ctx.limit_status == 0) & ~ctx.is_st
        ok &= ret.notna() & close.iloc[-1].notna()
        ok &= amt >= amt[ok].quantile(self.liq_drop)
        ret = ret[ok]
        if len(ret) < 50:
            return None

        pct = ret.rank(pct=True)
        keep = pct[(pct > self.cut_bottom) & (pct <= 1 - self.cut_top)].index
        return ctx.equal_weight(keep)
'''

TS_MOMENTUM = '''from bt.api import Strategy


class TrendFollowing(Strategy):
    """时间序列动量：过去 L 日涨了就持有，跌了就空仓。
    股票池有多只时，等权持有所有「在涨」的股票。
    L = 20 是在同类股上选出来、再用到长鑫的参数。"""

    L = 20

    def on_bar(self, ctx):
        close = ctx.history("close", self.L + 1)
        if len(close) <= self.L:
            return {}            # 历史不够 L 天：空仓
        ret = close.iloc[-1] / close.iloc[0] - 1
        rising = sorted(ret[ret > 0].index)
        if rising == sorted(ctx.weights):   # 名单没变就不动，省手续费
            return None
        return ctx.equal_weight(rising)
'''

MA_CROSS = '''from bt.api import Strategy


class MovingAverageCross(Strategy):
    """均线交叉：短均线在长均线上方就持有，否则空仓。"""

    fast = 5
    slow = 20

    def on_bar(self, ctx):
        close = ctx.history("close", self.slow)
        if len(close) < self.slow:
            return {}
        up = close.tail(self.fast).mean() > close.mean()
        hold = sorted(up[up].index)
        if hold == sorted(ctx.weights):
            return None
        return ctx.equal_weight(hold)
'''

BUY_HOLD = '''from bt.api import Strategy


class BuyAndHold(Strategy):
    """买入持有：第一天收盘下单，之后一直不动（用来当基准）。"""

    def on_bar(self, ctx):
        if ctx.bar_index == 0:
            return ctx.equal_weight(ctx.universe)
        return None
'''

TEMPLATES = {
    "5DR 五日反转 · 每 5 日调仓": REVERSAL_5D,
    "5 日动量（5DR 镜像）· 每 5 日调仓": MOMENTUM_5D,
    "剔除过热股 · 等权去两头 10%": AVOID_EXTREMES,
    "趋势跟随 · 时间序列动量 L=20": TS_MOMENTUM,
    "均线交叉 · 5 日 / 20 日": MA_CROSS,
    "买入持有": BUY_HOLD,
}
