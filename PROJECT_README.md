# DCSA 独立项目

DCSA（Dual-Channel Support-Curve Adaptation）是一个独立的热电材料曲线外推项目。当前主算法为 CRCT（Chemistry-conditioned Residual Curve Transport），DCSA 神经网络作为其可学习扩展。

## 任务

给定某篇训练中未见论文内 1–5 个已测材料的 ZT–T 曲线，预测该论文其他组成材料的 ZT–T 曲线。

## 数据边界

DCSA 使用从 `C:\Users\26037\OneDrive\ml+te\data\processed\` 复制到本项目的既有清洗数据：

- `data/processed/starrydata2_curves_long_clean.parquet`
- `data/processed/starrydata2_samples_clean.csv`

本项目不重新清洗这些数据。复制来源只作为溯源记录，DCSA 后续只读取本项目 `data/processed/` 中的副本；`ml+te` 原文件不修改。

## 目录

- `src/`：DCSA 模型和数据处理
- `tests/`：单元测试
- `results/`：审计、基线和模型结果
- `docs/`：协议与研究记录

## 当前主线

对一篇论文中 5 条已测曲线，先用跨论文组成–温度先验得到 zero-shot 预测，再计算 support 的完整残差曲线；CRCT 按组成化学距离将残差传输到未测组成的温度网格。历史 DOI 遮蔽验证中，random support 的 CRCT MAE 为 0.0884，temperature-Ridge 为 0.1006；改用不读取 ZT 的 diverse support 后，CRCT MAE 降至 0.0733，同策略 Ridge 为 0.0882。

## 当前阶段

清洗数据副本已经就位。下一步在本项目内完成 DOI–组成–曲线支持审计、基线和 PyTorch 双通道模型训练；每次运行记录输入文件哈希。
