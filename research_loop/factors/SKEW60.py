HYPOTHESIS = "彩票偏好：收益分布右偏（偶尔暴涨）的股票被高估，之后跑输。因子 = −60 日日收益偏度。与低风险家族的信息不同，希望找到低相关的新方向。"


def factor(d):
    r = daily_return(d)
    m1 = roll_mean(r, 60)
    m2 = roll_mean(r * r, 60)
    m3 = roll_mean(r ** 3, 60)
    var = m2 - m1 ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / var ** 1.5
    return -skew
