HYPOTHESIS = "LVILQ 用的是 20 日低波动，验证期掉得多。换成两期都稳定的 60 日低波动，再与低成交额等权组合，冲混合目标。"


def factor(d):
    xr = lambda a: pd.DataFrame(a).rank(axis=1, pct=True).to_numpy()
    lowvol = -roll_std(daily_return(d), 60)
    illiq = -np.log(roll_mean(np.where(d.traded, d.amt, np.nan), 20))
    return 0.5 * xr(lowvol) + 0.5 * xr(illiq)
