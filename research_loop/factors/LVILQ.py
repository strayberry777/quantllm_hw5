HYPOTHESIS = "低波动与低成交额相关仅 0.37：前者控回撤，后者给超额。两者排名等权相加，希望同时提高夏普、压低回撤（冲混合目标）。"


def factor(d):
    xr = lambda a: pd.DataFrame(a).rank(axis=1, pct=True).to_numpy()
    lowvol = -roll_std(daily_return(d), 20)
    illiq = -np.log(roll_mean(np.where(d.traded, d.amt, np.nan), 20))
    return 0.5 * xr(lowvol) + 0.5 * xr(illiq)
