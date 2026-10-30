HYPOTHESIS = "低波动 + 低成交额的组合两次都是训练期好、验证期掉。换成两期都稳的特质波动 60 日再与低成交额组合，检验问题是否出在低波动这一半。"


def factor(d):
    xr = lambda a: pd.DataFrame(a).rank(axis=1, pct=True).to_numpy()
    r = daily_return(d)
    m = np.nanmean(np.where(d.traded, r, np.nan), axis=1)[:, None]
    mm = np.where(np.isfinite(r), m, np.nan)
    mr, mmm = roll_mean(r, 60), roll_mean(mm, 60)
    cov = roll_mean(r * mm, 60) - mr * mmm
    var = roll_mean(mm * mm, 60) - mmm ** 2
    vr = roll_mean(r * r, 60) - mr ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        ivol = -np.sqrt(np.maximum(vr - cov ** 2 / var, 0))
    illiq = -np.log(roll_mean(np.where(d.traded, d.amt, np.nan), 20))
    return 0.5 * xr(ivol) + 0.5 * xr(illiq)
