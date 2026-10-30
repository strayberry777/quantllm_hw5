HYPOTHESIS = "低波动是回撤目标的冠军。窗口从 20 日拉长到 60 日，波动估计更稳，排名变化更慢：希望回撤更小、换手更低。"


def factor(d):
    return -roll_std(daily_return(d), 60)
