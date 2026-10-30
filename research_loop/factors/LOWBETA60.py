HYPOTHESIS = "低风险的另一种定义：对市场（股票池等权）的 60 日贝塔越低越好。检验低波动的好处是否来自市场敞口低，在大跌时回撤更小。"


def factor(d):
    r = daily_return(d)
    m = np.nanmean(np.where(d.traded, r, np.nan), axis=1)[:, None]
    mm = np.where(np.isfinite(r), m, np.nan)
    cov = roll_mean(r * mm, 60) - roll_mean(r, 60) * roll_mean(mm, 60)
    var = roll_mean(mm * mm, 60) - roll_mean(mm, 60) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = cov / var
    return -beta
