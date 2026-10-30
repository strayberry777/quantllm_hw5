HYPOTHESIS = "极端度 IC 最强但做多端弱，低波动做多端强。先用极端度惩罚两头，再偏好低波动：两者排名等权相加，希望把极端度的 IC 转成能做多的收益。"


def factor(d):
    xr = lambda a: pd.DataFrame(a).rank(axis=1, pct=True).to_numpy()
    ext = -np.abs(xr(past_return(d, 5)) - 0.5)
    lowvol = -roll_std(daily_return(d), 20)
    return 0.5 * xr(ext) + 0.5 * xr(lowvol)
