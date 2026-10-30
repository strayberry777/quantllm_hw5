HYPOTHESIS = "低波动 60 日回撤最小，特质波动 60 日夏普最高。两者排名等权相加，正好对应「一半夏普、一半回撤」的混合目标。"


def factor(d):
    xr = lambda a: pd.DataFrame(a).rank(axis=1, pct=True).to_numpy()
    r = daily_return(d)
    lowvol = -roll_std(r, 60)
    m = np.nanmean(np.where(d.traded, r, np.nan), axis=1)[:, None]
    mm = np.where(np.isfinite(r), m, np.nan)
    mr, mmm = roll_mean(r, 60), roll_mean(mm, 60)
    cov = roll_mean(r * mm, 60) - mr * mmm
    var = roll_mean(mm * mm, 60) - mmm ** 2
    vr = roll_mean(r * r, 60) - mr ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        ivol = -np.sqrt(np.maximum(vr - cov ** 2 / var, 0))
    return 0.5 * xr(lowvol) + 0.5 * xr(ivol)
