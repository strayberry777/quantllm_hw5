HYPOTHESIS = "20 日→60 日时低波动的夏普和回撤都变好、换手下降。继续拉长到 120 日，检验这个趋势是否延续，还是开始变慢变钝。"


def factor(d):
    return -roll_std(daily_return(d), 120)
