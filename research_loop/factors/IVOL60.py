HYPOTHESIS = "低贝塔无效、低波动有效，推测低波动的好处来自个股特有的风险。特质波动 = 60 日内对市场回归后的残差标准差，越低越好。"


def factor(d):
    r = daily_return(d)
    m = np.nanmean(np.where(d.traded, r, np.nan), axis=1)[:, None]
    mm = np.where(np.isfinite(r), m, np.nan)
    mr, mmm = roll_mean(r, 60), roll_mean(mm, 60)
    cov = roll_mean(r * mm, 60) - mr * mmm
    var = roll_mean(mm * mm, 60) - mmm ** 2
    vr = roll_mean(r * r, 60) - mr ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = vr - cov ** 2 / var
    return -np.sqrt(np.maximum(resid, 0))
