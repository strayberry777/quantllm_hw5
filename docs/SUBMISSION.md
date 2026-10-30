## 第五讲作业：5DR 因子评价与 AI 自迭代因子优化框架

**仓库**：https://github.com/strayberry777/quantllm_hw5
**优化目标的演进与结论**：[docs/JOURNEY.md](https://github.com/strayberry777/quantllm_hw5/blob/main/docs/JOURNEY.md) · **完整报告**：[docs/AI_LOOP.md](https://github.com/strayberry777/quantllm_hw5/blob/main/docs/AI_LOOP.md) · [docs/FACTOR_EVAL.md](https://github.com/strayberry777/quantllm_hw5/blob/main/docs/FACTOR_EVAL.md)

本作业基于自建的 A 股日频回测平台，首先完成 5DR 的单因子评价；随后在不改动回测引擎与评价器的前提下，构建了 AI 自迭代的因子研究框架：
AI 依据研究日志提出因子假设，由固定评价器打分，留出期（2025-01 至 2026-09）全程锁定、每个因子仅检验一次。
前三轮由 AI 提出 21 个因子，第四轮引入 Alpha101 的 51 个公式，共 72 个实验，分别以 IC、夏普比率、最大回撤、复合目标（夏普与回撤各 50%）及新颖度五个目标选取最优因子。
评价口径与作业规格一致：非 ST 股票池，多头最高十分位，佣金万一、印花税万五、滑点 5bp。

### 1. 5DR：IC 显著，但多头端不可实现

![5DR 评价总览](https://raw.githubusercontent.com/strayberry777/quantllm_hw5/main/docs/img/issue_5dr_overview.png)

5DR 的 RankIC 为 0.031（Newey-West t = 5.4），但十分组收益呈倒 U 形：前期涨幅最大组每 5 日超额 −48.9bp，5DR 买入的前期跌幅最大组亦为 −13.2bp，
扣费后多头年化超额 −13.9%。预测力集中于空头端，在无法做空的约束下不可实现。这说明以 IC 为唯一目标会系统性地偏向不可交易的信号，因此在后续框架中引入多个目标进行对比。

### 2. 全部候选实验

![全部候选](https://raw.githubusercontent.com/strayberry777/quantllm_hw5/main/docs/img/loop_candidates.png)

72 个实验按时间顺序写入只增不改的研究日志，被拒绝的实验均标注原因（验证期不显著、训练期方向不一致、与池内因子冗余）。
入池门槛随累计实验次数提高，72 个实验后 Bonferroni 校正的 t 值门槛为 3.39。

### 3. 各目标最优因子的演进

![冠军演进轨迹](https://raw.githubusercontent.com/strayberry777/quantllm_hw5/main/docs/img/loop_incumbent.png)

IC 目标选出 EXT5，夏普目标选出 IVOL60（60 日特质波动率），最大回撤、复合与新颖度目标均选出 LOWVOL60（60 日总波动率）。
第四轮的 51 个 Alpha101 公式中有 18 个 IC 通过校正，但年化换手中位数达 81 倍，扣费后无一在任何目标下成为最优。
复合目标退化为回撤目标，原因在于候选集集中于低风险因子族；新颖度目标未生效，原因在于惩罚项与得分的量纲不一致。

### 4. 因子池效用

![因子池效用](https://raw.githubusercontent.com/strayberry777/quantllm_hw5/main/docs/img/loop_pool_utility.png)

16 个 Alpha101 因子按规则入池，因子池由 6 个扩至 22 个，等权合成信号的验证期 RankIC 由 0.093 降至 0.081。
合成采用等权而非回归拟合权重，以避免在验证期上产生过拟合。

### 5. 留出期检验

![留出期全部检验](https://raw.githubusercontent.com/strayberry777/quantllm_hw5/main/docs/img/loop_holdout_all.png)

- **目标函数决定优化结果的性质**：夏普最优因子样本外夏普 1.00、最大回撤 −14.8%，均优于基准（0.83 / −19.4%）；回撤最优因子有效控制了回撤，但收益仅为基准的约四成；IC 最优因子保持显著的预测力，但多头收益低于基准
- **合成因子的 IC 与收益背离**：22 因子等权合成组合总收益 +49.5%、夏普 1.21，是唯一显著跑赢基准（+30.5%）的组合，但其合成 IC 低于 6 因子组合。代价是最大回撤 −24.8%、年化换手 72 倍
- **验证期存在乐观偏差**：最优因子的 IC t 值在样本外普遍下降（IVOL60 由 5.6 降至 2.8）

### 主要结论

1. 目标函数决定优化结果的性质，未纳入目标的维度（收益、换手、容量）不会自动受到约束
2. 复合目标的有效性依赖候选集的分散程度；引入惩罚项前须先统一量纲
3. IC 与可实现收益的背离在研究中多次出现；自迭代框架中，防止数据窥探的机制（留出期锁定、多重检验校正、完整记录失败实验）与搜索机制同等重要

**局限**：数据缺少市值与行业分类，未做中性化处理；ST 状态为推断；Alpha101 按论文附录转写，个别公式可能存在细微差异；
留出期约 21 个月且处于上涨行情，共进行 7 次检验，最优结果存在选择偏差。结论仅限样本期，不构成投资建议。
