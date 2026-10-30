HYPOTHESIS = "用日内振幅衡量风险：20 日平均 (最高价 − 最低价) / 前收盘，越低越好。信息来自高低价，而不是收盘价，可能与低波动互补。"


def factor(d):
    C = ffill(np.where(d.traded, d.close, np.nan))
    prev = np.full_like(C, np.nan)
    prev[1:] = C[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        rng = np.where(d.traded, (d.high - d.low) / prev, np.nan)
    return -roll_mean(rng, 20)
