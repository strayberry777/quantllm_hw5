## 第五讲作业：5DR 因子评价 + AI 自迭代的因子优化闭环

**仓库**：https://github.com/strayberry777/quantllm_hw5
**心路历程与心得**：[docs/JOURNEY.md](https://github.com/strayberry777/quantllm_hw5/blob/main/docs/JOURNEY.md) · **完整报告**：[docs/AI_LOOP.md](https://github.com/strayberry777/quantllm_hw5/blob/main/docs/AI_LOOP.md) · [docs/FACTOR_EVAL.md](https://github.com/strayberry777/quantllm_hw5/blob/main/docs/FACTOR_EVAL.md)

在上次的 A 股回测平台上，先完成 5DR 的单因子评价，再在**不改回测系统**的前提下，搭了一个让 AI 自迭代找因子的闭环：AI 读研究日志、写反思、提出因子，固定的评价器打分，留出期（2025-01 至 2026-09）上锁、每个因子只考一次。前 3 轮由 AI 自提 21 个因子，第 4 轮接入 Alpha101 的 51 个公式，共 72 个实验，用五个优化目标（IC、夏普、最大回撤、50/50 混合、新颖度）分别选冠军。口径与作业一致：非 ST 股票池、做多最高 10%、佣金万一 + 印花税万五 + 滑点 5bp。

### 1. 从 5DR 出发：IC 为正，做多却赚不到钱

![5DR 评价总览](https://raw.githubusercontent.com/strayberry777/quantllm_hw5/main/docs/img/issue_5dr_overview.png)

5DR 的 RankIC 为 0.031（t = 5.4），但十分组呈倒 U 形：涨得最多的一组每 5 日跑输 48.9bp，5DR 买入的「跌得最多」一组也跑输 13.2bp，扣费后做多年化 −13.9%。这让我们意识到：**只用 IC 当优化目标，会找到一堆赚不到钱的因子**，于是在闭环里改用不同目标做比较。

### 2. 全部候选：成功与失败同样保留

![全部候选](https://raw.githubusercontent.com/strayberry777/quantllm_hw5/main/docs/img/loop_candidates.png)

72 个实验按时间顺序全部写进只增不改的日志，被拒绝的都写明原因（验证期不显著、训练期方向不一致、与池内因子重复）。晋升门槛随试验次数提高（Bonferroni 校正，72 个实验后为 t > 3.39）。

### 3. 冠军进化轨迹：目标不同，冠军不同

![冠军进化轨迹](https://raw.githubusercontent.com/strayberry777/quantllm_hw5/main/docs/img/loop_incumbent.png)

IC 目标选出 EXT5，夏普目标选出 IVOL60（60 日特质波动），最大回撤、50/50 混合、新颖度目标都选出 LOWVOL60（60 日低波动）。第 4 轮 51 个 Alpha101 公式虽有 18 个 IC 显著，但年换手中位数 81 倍，扣费后没有一个成为任何目标的冠军。混合目标和新颖度目标也没选出新因子：前者因为候选池太单一，后者因为惩罚项的单位比得分小得多。

### 4. 因子池效用：按 IC 看，池子变大反而变差

![因子池效用](https://raw.githubusercontent.com/strayberry777/quantllm_hw5/main/docs/img/loop_pool_utility.png)

16 个 Alpha101 因子按规则晋升，池子从 6 个扩到 22 个，但等权组合的验证期 RankIC 从 0.093 降到 0.081。组合不用回归拟合权重，避免在验证期上过拟合。

### 5. 留出期考试：只考一次

![留出期全部考试](https://raw.githubusercontent.com/strayberry777/quantllm_hw5/main/docs/img/loop_holdout_all.png)

- **优化什么，就得到什么**：夏普冠军留出期夏普 1.00、回撤 −14.8%，两项都优于基准（0.83 / −19.4%）；回撤冠军压住了回撤，但收益只有基准的四成；IC 冠军 IC 依然最强，但做多不赚钱
- **22 个因子等权组合是唯一明显跑赢基准的**：总收益 +49.5% 对 +30.5%，夏普 1.21。它的组合 IC 却比 6 个因子的池子低——IC 和收益又一次指向不同方向；代价是回撤 −24.8%、年换手 72 倍
- **验证期分数偏乐观**：冠军的 IC t 值在留出期普遍下降（IVOL60 从 5.6 降到 2.8）

### 心得

1. 目标函数是和 AI 之间的合同，它会严格按合同交付；合同没写的（收益、换手）不会替你照顾
2. 混合目标能否选出均衡的因子，取决于候选池是否多样；加约束前要先把各项的单位对齐
3. 自迭代闭环的价值一半在搜索，一半在防止自我欺骗：留出期上锁、门槛随试验次数提高、失败同样记录

**局限**：数据无市值和行业，未做中性化；ST 为推断；Alpha101 按论文附录转写，个别公式可能有细微出入；留出期只有约 21 个月且为上涨行情，共考了 7 次，最好的结果有运气成分。结论仅限样本期，不构成交易建议。
