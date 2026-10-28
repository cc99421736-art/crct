# CRCT Nature-Style Figure Generation

科学期刊级热电性能曲线预测论文图表自动生成脚本

## 快速开始（3 步）

### 方法 1：一键运行（推荐）

双击 `run_figures.bat`，脚本会自动：
1. 检查 Python 环境
2. 安装必要的库（matplotlib, seaborn, pandas, numpy, scipy）
3. 生成全部 6 张 Nature 风格图表

**预计耗时：30-60 秒**

### 方法 2：手动运行

如果 `.bat` 文件不工作，在命令行执行：

```bash
# 1. 进入项目目录
cd C:\Users\26037\Desktop\dataset_bundle_2026-09-29\crct

# 2. 安装依赖（仅需一次）
pip install -r requirements.txt

# 3. 运行生成脚本
python gen_figures.py
```

### 方法 3：在 VS Code 中运行

1. 用 VS Code 打开项目文件夹
2. 打开 `gen_figures.py`
3. 点击右上角"Run"或按 `Ctrl+F5`
4. 查看输出

---

## 生成的图表

| 图号 | 文件名 | 描述 |
|-----|--------|-----|
| **Figure 2** | `Figure_2_Main_Results.*` | CRCT vs 强基线的主结果对比（3 面板） |
| **Figure 3** | `Figure_3_Ablation.*` | 组件消融与参数敏感性（3 面板） |
| **Figure 4** | `Figure_4_Distance_Boundaries.*` | 组成距离分层与外推边界（3 面板） |
| **Figure 5** | `Figure_5_Temporal_Ranking.*` | 时间前瞻验证与候选排序（3 面板） |
| **Figure 6** | `Figure_6_Risk_Layer.*` | 风险层与选择性预测（3 面板） |
| **Figure 7** | `Figure_7_MultiSystem.*` | 多体系候选筛选（4 面板） |

每张图生成为三种格式：
- `.svg` — 可编辑的矢量图（用于 Illustrator/Inkscape 微调）
- `.pdf` — 出版物高质量版本（推荐投稿用）
- `.tiff` — 高分辨率栅格图（600 dpi，备选格式）

---

## 输出位置

所有图表保存到：
```
C:\Users\26037\Desktop\dataset_bundle_2026-09-29\crct\output\
```

每个文件名形如：
```
Figure_2_Main_Results.svg
Figure_2_Main_Results.pdf
Figure_2_Main_Results.tiff
```

---

## 技术规范（Nature 风格）

### 色彩方案
- **CRCT**：深蓝 `#0047AB`
- **基线**：灰色 `#888888`
- **失败边界**：暗红 `#8B3A3A`

### 排版
- **字体**：Arial 7pt（正文）、8pt（标题）
- **线宽**：0.8pt
- **背景**：白色，无阴影或渐变
- **图例**：无边框，12pt 字体

### 统计
- **误差条**：DOI 聚类 bootstrap 95% 置信区间
- **相关性**：Spearman ρ（图 6a）
- **显著性**：p 值注记

---

## 常见问题

### Q1: 运行后没有生成任何图
**A:**
1. 检查 `output/` 目录是否存在
2. 查看控制台错误信息
3. 确认所有数据文件在 `figures/` 子文件夹中

### Q2: 图表颜色/布局不满意
**A:**
编辑 `gen_figures.py` 中的以下部分：
- **COLOR_CRCT**、**COLOR_BASELINE** 等常量（第 30-35 行）：修改颜色
- **figsize** 参数（各 `plt.subplots()` 调用）：调整大小
- **fontsize** 参数：改变字体大小

然后重新运行。

### Q3: 如何在论文中嵌入这些图？

**对于 LaTeX：**
```latex
\begin{figure}
\centering
\includegraphics[width=0.9\textwidth]{Figure_2_Main_Results.pdf}
\caption{CRCT vs strong baselines...}
\label{fig:main-results}
\end{figure}
```

**对于 Word：**
直接插入 `.pdf` 或 `.png`（从 `.tiff` 转换）

---

## 数据依赖

脚本期望以下 CSV 文件存在于 `figures/` 子目录中：

```
figures/
├── 图2_主结果与强基线比较/
│   ├── nested_strong_baselines_outer.csv
│   ├── nested_strong_baselines_paired_stats.csv
│   └── crct_nested_paired_improvement_summary.csv
├── 图3_组件消融与支持集敏感性/
│   ├── matched_baselines_ablation_summary.csv
│   ├── crct_support_selection_comparison_summary.csv
│   └── crct_parameter_sensitivity_summary.csv
├── 图4_组成距离与外推能力/
│   ├── crct_masked_validation.csv
│   ├── crct_masked_validation_summary.csv
│   └── element_family_holdout_paired.csv
├── 图5_时间前瞻与候选排序回测/
│   ├── temporal_validation.csv
│   └── active_design_backtest.csv
├── 图6_风险层与选择性预测/
│   ├── crct_uncertainty_risk_scores.csv
│   ├── crct_selective_prediction_summary.csv
│   └── multisystem_crct_robustness_summary.csv
└── 图7_多体系候选筛选与风险排序/
    ├── multisystem_crct_candidate_screening.csv
    └── multisystem_crct_robustness_summary.csv
```

---

## 自定义调整

### 改变图表大小

编辑对应 `gen_figure*()` 函数中的 `figsize` 参数。例如：

```python
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0))  # 改为 (10, 3)
```

### 改变颜色

在脚本顶部修改色彩常量（第 30-35 行）：

```python
COLOR_CRCT = "#0047AB"        # 改为您想要的十六进制颜色
COLOR_DARKRED = "#FF0000"     # 例如
```

### 改变字体大小

修改 `mpl.rcParams` 或特定 `ax.text()` 调用中的 `fontsize` 参数。

---

## 输出示例

运行后，您将在 `output/` 目录看到：

```
output/
├── Figure_2_Main_Results.svg
├── Figure_2_Main_Results.pdf
├── Figure_2_Main_Results.tiff
├── Figure_3_Ablation.svg
├── Figure_3_Ablation.pdf
├── Figure_3_Ablation.tiff
├── ... (6 个图，每个 3 种格式 = 18 个文件)
└── Figure_7_MultiSystem.tiff
```

---

## 许可

这些脚本和生成的图表遵循原论文的许可和发布政策。

---

## 支持

如有问题或反馈，请查看：
- 脚本中的 `# ============================================================================` 注释
- 各 `gen_figure*()` 函数的文档字符串
- Nature 期刊图表指南：https://www.nature.com/documents/nr-reporting-summary.pdf

---

**Generated with CRCT Python Figure Generation Kit**
