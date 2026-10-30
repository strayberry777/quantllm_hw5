HYPOTHESIS = "回撤来自下跌，只算下跌日的波动（60 日下行半方差的平方根），越低越好。希望比总波动更直接地压低最大回撤。"


def factor(d):
    r = daily_return(d)
    neg = np.where(np.isfinite(r), np.minimum(r, 0.0), np.nan)
    return -np.sqrt(roll_mean(neg * neg, 60))
