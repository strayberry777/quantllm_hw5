HYPOTHESIS = "直接用回撤当因子：个股过去 60 日从高点下来的最大回撤越小越好。检验「过去回撤小」能否延续成「组合回撤小」。"


def factor(d):
    C = pd.DataFrame(ffill(np.where(d.traded, d.adj_close, np.nan)))
    peak = C.rolling(60, min_periods=48).max()
    dd = (C / peak - 1).rolling(60, min_periods=48).min()
    return dd.to_numpy()          # 回撤为负数，越接近 0 越好，所以值越大越好
