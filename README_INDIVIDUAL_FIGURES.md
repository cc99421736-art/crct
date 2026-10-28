# 单独图表脚本运行说明

每张图都有独立的 Python 脚本，放在对应的 `figures/` 子文件夹中。

## 快速开始

### 方法 1：一键运行全部图表

在项目目录打开命令行，运行：

```bash
python RUN_ALL_FIGURES.py
```

这会逐个运行图 2-7，生成所有 18 个文件（6 张图 × 3 种格式）。

### 方法 2：单独运行某一张图

进入对应的子文件夹，运行该图的脚本。例如，只运行图 2：

```bash
cd figures/图2_主结果与强基线比较
python gen_fig2.py
```

输出文件会保存到 `../../output/` 目录。

---

## 文件结构

```
crct/
├── figures/
│   ├── 图2_主结果与强基线比较/
│   │   ├── gen_fig2.py                          ← 脚本
│   │   ├── nested_strong_baselines_outer.csv
│   │   ├── nested_strong_baselines_paired_stats.csv
│   │   └── crct_nested_paired_improvement_summary.csv
│   ├── 图3_组件消融与支持集敏感性/
│   │   ├── gen_fig3.py
│   │   ├── matched_baselines_ablation_summary.csv
│   │   ├── crct_support_selection_comparison_summary.csv
│   │   └── crct_parameter_sensitivity_summary.csv
│   ├── 图4_组成距离与外推能力/
│   │   ├── gen_fig4.py
│   │   ├── crct_masked_validation.csv
│   │   ├── crct_masked_validation_summary.csv
│   │   └── element_family_holdout_paired.csv
│   ├── 图5_时间前瞻与候选排序回测/
│   │   ├── gen_fig5.py
│   │   ├── temporal_validation.csv
│   │   └── active_design_backtest.csv
│   ├── 图6_风险层与选择性预测/
│   │   ├── gen_fig6.py
│   │   ├── crct_uncertainty_risk_scores.csv
│   │   ├── crct_selective_prediction_summary.csv
│   │   └── multisystem_crct_robustness_summary.csv
│   └── 图7_多体系候选筛选与风险排序/
│       ├── gen_fig7.py
│       ├── multisystem_crct_candidate_screening.csv
│       └── multisystem_crct_robustness_summary.csv
├── output/                                        ← 输出目录
│   ├── Figure_2_Main_Results.svg/.pdf/.tiff
│   ├── Figure_3_Ablation.svg/.pdf/.tiff
│   ├── Figure_4_Distance_Boundaries.svg/.pdf/.tiff
│   ├── Figure_5_Temporal_Ranking.svg/.pdf/.tiff
│   ├── Figure_6_Risk_Layer.svg/.pdf/.tiff
│   └── Figure_7_MultiSystem.svg/.pdf/.tiff
├── RUN_ALL_FIGURES.py                           ← 一键运行脚本
└── README_INDIVIDUAL_FIGURES.md
```

---

## 运行结果

每张图生成 3 种格式：

- **`.svg`** — 矢量格式，可在 Illustrator/Inkscape 中编辑
- **`.pdf`** — 出版物高质量格式，**推荐投稿用**
- **`.tiff`** — 栅格格式（600 dpi），备选选项

---

## 常见问题

### Q: 运行后没有生成图？

**检查清单：**
1. `output/` 目录是否存在？（不存在会自动创建）
2. 对应的数据 CSV 文件是否都在？
3. Python 库是否已安装？

```bash
pip install matplotlib seaborn pandas numpy scipy
```

### Q: 只运行某一张图怎么做？

进入对应的子文件夹后运行脚本：

```bash
cd figures/图2_主结果与强基线比较
python gen_fig2.py
```

### Q: 如何修改颜色或样式？

编辑对应的 `gen_fig*.py` 脚本。配色在脚本顶部：

```python
COLOR_CRCT = "#0047AB"        # 深蓝 → 改为其他颜色
COLOR_RIDGE = "#888888"       # 灰色
COLOR_DARKRED = "#8B3A3A"     # 暗红
```

### Q: 运行脚本时看到错误如何排查？

直接运行脚本会打印详细错误信息，例如：

```
✗ Data loaded
  - nested_strong_baselines_outer.csv not found
```

这表示数据文件缺失或路径错误。检查文件是否都在对应的子文件夹中。

---

## Nature 风格规范

所有脚本都遵循以下规范：

- **字体**：Arial 7pt，细线 0.8pt
- **背景**：白色，无阴影、无渐变
- **色彩**：
  - 深蓝 `#0047AB` — CRCT 主方法
  - 灰色 `#888888` — 基线对照
  - 暗红 `#8B3A3A` — 失败边界/警告
- **统计**：DOI 聚类 bootstrap 95% 置信区间
- **输出**：SVG（可编辑）、PDF（投稿）、TIFF（高分辨率）

---

## 依赖包

```
matplotlib>=3.5.0
seaborn>=0.12.0
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
```

安装全部：

```bash
pip install matplotlib seaborn pandas numpy scipy
```

---

**Generated with CRCT Python Figure Kit**
