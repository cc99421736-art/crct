# DCSA 项目状态

更新时间：2026-10-02。

## 当前目标

建立独立于 `ml+te` 的 DCSA 项目，用 StarryData2、ESTM 和 sysTEm 数据研究论文内少量支持曲线驱动的 ZT–T 外推。

## 已完成

- 已确认上级目录包含 `StarryData2/`、`ESTM/`、`sysTEm/` 原始数据目录。
- 已从 `OneDrive/ml+te/data/processed/` 复制既有清洗数据到本项目 `data/processed/`，不重新清洗。
- 已确认 `ml+te` 原文件保持不变；DCSA 后续代码、测试和结果只写本项目。
- 两个副本与来源文件 SHA-256 一致：曲线 `e70b386c...24a00a2`，样品 `b6b8254c...be93945b`；完整记录见 `results/data_provenance.json`。
- 已复制 DCSA 当前需要的组成解析和元素特征依赖到 `src/features/`。
- 支持曲线审计完成：输入 ZT 点 289,252；有效 DOI–组成曲线 14,176；有效 DOI 3,882；`D_1/D_2/D_3/D_5 = 2,601/2,249/1,891/796`。
- 严格 DOI 留出基线完成，结果在 `results/support_curve_baseline*.csv`；1-shot MAE：zero-shot 0.1412、paper-offset 0.1272、temperature-Ridge 0.1211。
- 已完成按 `seed:DOI` 聚类的 95% bootstrap 汇总：`results/support_curve_baseline_bootstrap.csv`。
- 已在本项目 `DCSA/.deps` 安装 CPU PyTorch 2.14.1；`ml+te` 未修改。
- DCSA 双通道模型前向接口已实现并通过置换不变性测试；当前模型尚未宣称完成训练或达到最终性能。
- 研究方案已改为“5 条已测曲线 support → 同论文未测组成 query”，详见 `docs/research_protocol.md`。
- 历史 5-shot 遮蔽验证已完成：445 个 seed:DOI episode；temperature-Ridge 的 random / central / ordered support MAE 分别为 0.1006 / 0.1133 / 0.1172。
- 支持选择会影响外推误差；上述结果是基线诊断，不代表 DCSA 神经网络已经优于基线。
- DCSA 5-shot CPU 训练与三项消融已完成（20 epochs，3 seeds，445 episodes/seed）；完整模型原始 ZT MAE 0.1815，paper-only 0.1814，base-only 0.2742，chemical-only 0.2797。当前 DCSA 尚未超过 temperature-Ridge 5-shot random-support 基线 0.1006，下一步需改进曲线级 support 编码和训练目标。
- 已将 support 从均值摘要改为 16 点温度–ZT 序列编码，并加入 LightGBM zero-shot 先验与 support 残差输入；3 seeds、20 epochs 后 DCSA full MAE 0.1202，paper-only 0.1219，base-only 0.1449，chemical-only 0.1561。
- 新模型相对旧版明显改善，但仍未超过 temperature-Ridge random-support MAE 0.1006；当前不能宣称神经 DCSA 已取得最终方法优势。
- 主线已强化为三层残差分解：跨论文先验 + 论文级温度 Ridge 锚点 + support 条件化化学残差；DCSA full 将与严格同口径的 Ridge anchor 做 seed:DOI 配对比较。
- CRCT（化学条件残差曲线传输）已完成正式验证：总体 MAE 0.0884，Ridge 0.1006；平均 paired improvement +0.0122，DOI-bootstrap 95% CI [0.0086, 0.0137]，episode 胜率 80.0%。远距离四分位组 CRCT/Ridge MAE 为 0.1039/0.1249。
- CRCT 结果已增加 query 到最近 support 的组成距离分层和 support 温区覆盖率字段；当前 far/near 阈值按测试结果分位数作描述性分组，正式论文需改为训练内预注册阈值。
- 当前全套测试为 19 passed；`ml+te` 仍未修改。
- 主动 diverse support 已完成：按组成特征 k-center/farthest-point 选择 5 条支持曲线，不读取 ZT；CRCT MAE 0.0733，同策略 temperature-Ridge 0.0882，paired improvement +0.0149，DOI-bootstrap 95% CI [0.0115, 0.0175]，胜率 81.8%。远距离组 CRCT/Ridge 为 0.0778/0.1067。
- CRCT 已接入峰值 ZT、最佳温度、Pearson/Spearman 曲线相关性和 top-k 排序指标；每个 episode 同时记录 support 温区覆盖率。
- CRCT 参数敏感性已完成：18 组（2 种 support × 3 tau × 3 blend），共 8,010 条 episode 记录；diverse support 的 tau=.25/.5/1.0、blend=1 MAE 为 .0708/.0733/.0765，blend=.5 均更差，blend=0 等于 Ridge。
- leave-element/family-out 已完成：未见元素组 CRCT/Ridge MAE 为 .0687/.0730，点估计改善 .0043，但 DOI-bootstrap CI 为 [-.0074, .0148]；未见元素族组仅 4 DOI、5 episodes，CRCT/Ridge 为 .1882/.1377，当前属于明确外推边界，不能作为成功主结论。
- 当前全套测试为 32 passed；`ml+te` 仍未修改。
- 内层参数选择已完成：3 个 outer seed 均在 outer-train 的 inner DOI validation 选择 `tau=.25, blend=1`；outer test 445 episodes 的 CRCT/Ridge MAE 为 .070763/.088226。
- 距离度量敏感性已完成：标准化欧氏距离在 diverse/tau=.25 下远距离 MAE .0707；cosine 对照远距离 MAE .0941，故保留欧氏距离为主模型。
- 严格结果分析包已写入 `results/analysis-report.md`、`results/stats-appendix.md`、`results/figure-catalog.md`。
- nested 外层结果已补充正式 DOI-cluster paired bootstrap：445 个 episodes、382 个 DOI，CRCT/Ridge MAE 为 .070763/.088226，平均改善 .017463，95% CI [.013525, .020416]，episode 胜率 80.2%；结果见 `results/crct_nested_paired_improvement_summary.csv`。
- 已实现自适应局部邻居传输：新增 `neighbor_k` 参数，inner DOI validation 在三个 outer seed 中选择局部 2-neighbor；nested outer MAE 降至 .062309，Ridge 为 .088226，平均改善 .025917，95% CI [.019903, .030286]，胜率 76.2%；结果见 `results/crct_nested_paired_improvement_summary.csv` 和 `results/crct_nested_outer_validation.csv`。
- 自适应邻居的距离分层已完成：near/mid/far CRCT MAE 为 .056584/.062951/.067430，Ridge 为 .082114/.081848/.100757；far 组平均改善 .033327，DOI bootstrap 95% CI [.021285, .039183]。分层摘要见 `results/crct_nested_paired_improvement_summary.csv`。
- `neighbor_k=2, tau=.25` 的未见元素压力测试已完成：CRCT/Ridge MAE 为 .067349/.073006，平均改善 .005657，24 DOI、32 episodes，95% CI [-.018783, .020849]，只能作为支持性证据；元素族外推仍失败（.199296 vs .137710，4 DOI、5 episodes）。
- 距离门控已纳入 nested 参数选择：两个 outer seed 选择 `gate_tau=1.0`，一个选择不门控；总体 MAE .062248，Ridge .088226，平均改善 .025978，95% CI [.020070, .030407]；far 组改善 .033506，95% CI [.021838, .039533]。门控保留远距离外推优势，但不是所有 outer split 都需要。
- 已新增实验候选排序模块 `src/dcsa/active_design.py`：使用“预测峰值 ZT × 门控置信度 ÷ (1 + 化学距离)”生成透明优先级；该分数用于候选筛选，不解释为实验成功概率。下一步需在历史遮蔽数据上验证 top-k 命中率。
- 实验选点历史回测已完成：联合 screening score top-1 命中率 .735、regret .0392；只按预测峰值排序命中率 .742、regret .0357。联合分数当前未优于简单峰值排序，因此主动选点暂不作为独立成功贡献；实验指导建议采用峰值排序并附带距离/门控风险标签。
- screening 指数探索显示弱距离惩罚（distance exponent=.1）在 outer 结果上约为 .746 命中率、.0345 regret，但这是探索性、存在 outer 选择偏差；已加入 `rank_candidates` 参数化接口，下一步需在 inner split 学习指数后重新评估。
- 正式实验选点评价已补齐：预测峰值排序 top-1 命中率 .7416（95% CI [.6933, .7805]），随机 .6427；top-3 命中率 .8438（160 episodes），随机 .6750；top-5 .9012（81 episodes），随机 .7901。对应 regret 逐级下降至 .0357/.0219/.0076。结果见 `results/active_design_evaluation.csv`。
- 时间前瞻验证已完成：按样品 `created_at` 年份做最早 80% DOI 训练、最晚 20% DOI 测试（3102/776 DOI，203 episodes）；CRCT MAE .065167，Ridge .094570，下降约 31.1%。结果见 `results/temporal_validation.csv`。时间字段是数据库记录时间，不等同于正式发表年份。
- 时间前瞻 DOI bootstrap 已补齐：平均改善 .029403，95% CI [.022027, .036844]，203 个 DOI 中 CRCT 胜率 79.3%；2024 年子组仅 6 DOI、CI 跨 0，需谨慎解释。结果见 `results/temporal_validation_stats.csv`。
- 已生成统一论文结果表 `results/unified_results_table.csv`，整合随机 DOI、时间前瞻、距离分层和未见元素/元素族压力测试，并保留元素族失败边界。

## 下一步

1. 将距离度量、远距离阈值和端点外推规则纳入 outer-train 内层选择。
2. 扩大元素族外推样本，或将其明确作为失败边界并研究自适应距离/不确定性拒答。
3. 进行时间前瞻验证，再决定是否继续神经 DCSA 扩展。

## 约束

- 不覆盖原始数据或 `ml+te` 原文件。
- `ml+te/data/processed` 是复制来源；本项目后续只读取自己的 `data/processed` 副本。
- 所有训练必须 DOI 外层隔离，支持组成与查询组成等价键互斥。
- 不把基线改善直接写成新算法成功。
