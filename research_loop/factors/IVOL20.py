HYPOTHESIS = "特质波动 60 日两期都好。用 20 日窗口检验它对窗口是否敏感：如果 20 日明显变差，说明需要较长窗口才能估准特质风险。"


def factor(d):
    r = daily_return(d)
    m = np.nanmean(np.where(d.traded, r, np.nan), axis=1)[:, None]
    mm = np.where(np.isfinite(r), m, np.nan)
    mr, mmm = roll_mean(r, 20), roll_mean(mm, 20)
    cov = roll_mean(r * mm, 20) - mr * mmm
    var = roll_mean(mm * mm, 20) - mmm ** 2
    vr = roll_mean(r * r, 20) - mr ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = vr - cov ** 2 / var
    return -np.sqrt(np.maximum(resid, 0))
