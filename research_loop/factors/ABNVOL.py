HYPOTHESIS = "注意力效应：近 5 日成交额相对近 60 日突然放大的股票被过度关注，之后跑输。因子 = −log(5 日均成交额 / 60 日均成交额)。全新方向，与低波动、低成交额的水平值不同。"


def factor(d):
    amt = np.where(d.traded, d.amt, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -np.log(roll_mean(amt, 5) / roll_mean(amt, 60))
