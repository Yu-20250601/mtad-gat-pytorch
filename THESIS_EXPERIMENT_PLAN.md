# 论文正式实验计划

## 1. 文档定位

本文件用于约束毕设论文阶段的正式实验流程，目标是：

- 保证实验设置可复现、可解释、可直接写入论文；
- 区分“正式论文实验”和“快速调试实验”；
- 统一对比实验、稳定性实验、消融实验和定性可视化实验的执行口径；
- 暂不预设唯一主指标，等结果汇总成表后再确定论文主指标与辅指标。

本计划默认面向当前仓库中的统一实验脚本 `benchmark_models.py`。

## 2. 正式实验总原则

- 正式论文实验默认使用全量训练集和全量测试集。
- 正式论文实验命令中不使用 `--max_train_size` 和 `--max_test_size`。
- `--max_train_size` / `--max_test_size` 只允许出现在快速调试、小样本预跑、代码通路验证中，不进入论文正式结果。
- 所有正式对比实验和消融实验都在同一批 SMD 机器上运行，保持数据范围一致。
- 模型比较时，除被研究变量外，其余参数尽量保持不变。
- 多随机种子实验用于验证结论稳定性，不用于替代正式主对比结果。
- 主指标、辅指标、补充指标的最终归属，等全部结果汇总后再定。

## 3. 数据选择与论文表述

### 3.1 正式实验数据

论文主实验固定使用 SMD 的 4 组机器：

- `1-2`
- `1-3`
- `2-1`
- `3-6`

### 3.2 选择理由

- 覆盖不同的 SMD machine group：`1-*`、`2-*`、`3-*`；
- 兼顾异常模式差异，避免只在单一类型机器上得出结论；
- 在保证代表性的前提下控制实验规模，便于完成多模型、多随机种子和消融实验；
- 与当前论文写作进度匹配，适合作为正式实验主表的数据子集。

### 3.3 可直接写入论文的方法描述

> 本文从 SMD 数据集中选取 `1-2`、`1-3`、`2-1` 和 `3-6` 四组具有代表性的机器进行实验。所选机器覆盖不同 machine group，并尽量避免异常过于稀疏或过于简单的极端情况，以在保证实验代表性的同时控制实验规模，便于进行模型对比、稳定性验证与消融分析。

## 4. 实验口径统一说明

### 4.1 统一脚本

正式实验统一使用：

```bash
python benchmark_models.py
```

脚本默认会输出：

- `benchmark_results.csv`
- `best_per_entity.csv`
- `run_config.json`
- 各模型目录下的 `metrics.json`

### 4.2 正式实验固定项

除特别说明外，正式实验统一采用：

- 数据集：`SMD`
- 机器：`1-2,1-3,2-1,3-6`
- `lookback=100`
- `batch_size=256`
- `normalize=true`
- `use_gatv2=true` 作为默认运行设置

说明：

- `use_gatv2=true` 在正式实验中作为默认设置，仅表示先统一固定该开关；
- 是否保留 `use_gatv2=true` 作为最终论文默认配置，应由后续 GATv2 消融实验结果决定；
- 正式计划阶段不提前写死“GATv2 一定更优”这一结论。

### 4.3 关于阈值指标

- 论文最终主指标暂不在本计划中写死；
- 统一保留 `epsilon_result`、`pot_result`、`bf_result` 三套结果；
- 等正式结果汇总成表后，再根据文献口径、结果稳定性和论文表达需要确定主指标。

## 5. 正式实验一：主模型对比实验

### 5.1 目的

- 比较主要 MTAD-GAT 变体在相同正式实验设置下的表现；
- 为论文主模型选择提供正式依据；
- 为后续结构消融分析提供基础结果。

### 5.2 参与模型

- `mtad_gat_gru`
- `mtad_gat_vae`
- `mtad_gat_sparsemax_vae`
- `mtad_gat_sparsemax_vae_node`

说明：

- 这 4 个模型足以覆盖当前论文最关心的结构差异：`GRU / VAE / Sparsemax / Node Embedding`；
- 若后续需要补传统基线，可单独增加一轮基线对比，不与结构消融主线混在一起。

### 5.3 正式运行命令

```bash
python benchmark_models.py \
  --datasets SMD \
  --smd_groups 1-2,1-3,2-1,3-6 \
  --models mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node \
  --mtad_epochs 15 \
  --batch_size 256 \
  --use_gatv2 true \
  --seed 42
```

### 5.4 需要保留的结果

- 每台机器、每个模型、每种阈值方法下的 `Precision / Recall / F1`
- `benchmark_results.csv`
- `best_per_entity.csv`
- `run_config.json`
- 各模型目录下的 `metrics.json`

### 5.5 需要形成的论文材料

- 主模型比较总表
- 每个模型在 4 台机器上的平均表现
- 一段关于主模型候选排序的分析文字

## 6. 正式实验二：多随机种子稳定性实验

### 6.1 目的

- 验证模型排序不是由单次随机初始化偶然造成；
- 检查不同模型在正式实验设置下的稳定性；
- 为后续论文中的“结果可信性”提供支撑。

### 6.2 参与模型

- `mtad_gat_gru`
- `mtad_gat_vae`
- `mtad_gat_sparsemax_vae`
- `mtad_gat_sparsemax_vae_node`

### 6.3 推荐随机种子

- `42`
- `52`
- `62`

如时间允许，可扩展到 5 个随机种子；若时间紧张，先完成以上 3 个。

### 6.4 正式运行命令

```bash
python benchmark_models.py \
  --datasets SMD \
  --smd_groups 1-2,1-3,2-1,3-6 \
  --models mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node \
  --mtad_epochs 15 \
  --batch_size 256 \
  --use_gatv2 true \
  --seed 42
```

```bash
python benchmark_models.py \
  --datasets SMD \
  --smd_groups 1-2,1-3,2-1,3-6 \
  --models mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node \
  --mtad_epochs 15 \
  --batch_size 256 \
  --use_gatv2 true \
  --seed 52
```

```bash
python benchmark_models.py \
  --datasets SMD \
  --smd_groups 1-2,1-3,2-1,3-6 \
  --models mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node \
  --mtad_epochs 15 \
  --batch_size 256 \
  --use_gatv2 true \
  --seed 62
```

### 6.5 需要形成的论文材料

- 各模型在不同阈值方法下的 `mean +- std`
- 关于稳定性与波动性的分析文字
- 主模型排序是否跨 seed 保持一致的结论

## 7. 正式实验三：结构消融实验

### 7.1 目的

- 解释最终模型为什么优于基础版本；
- 量化 `VAE`、`Sparsemax`、`Node Embedding` 的贡献；
- 为论文“方法分析”章节提供结构性证据。

### 7.2 消融链条

建议按以下逻辑解释：

- `mtad_gat_gru -> mtad_gat_vae`
- `mtad_gat_vae -> mtad_gat_sparsemax_vae`
- `mtad_gat_sparsemax_vae -> mtad_gat_sparsemax_vae_node`

### 7.3 是否需要单独重跑

- 若“多随机种子稳定性实验”已经覆盖以上 4 个模型，则结构消融通常无需额外重跑；
- 结构消融表可直接由正式对比实验与多 seed 结果整理得到。

### 7.4 需要形成的论文材料

- 一张结构消融表
- 一段模块贡献分析文字
- 对各组件收益是否稳定的简要结论

## 8. 正式实验四：GATv2 消融实验

### 8.1 目的

- 验证 `use_gatv2=true` 是否在当前正式实验设置下具有稳定收益；
- 为论文中是否保留 GATv2 相关表述提供依据。

### 8.2 建议模型

- 优先只在一个主模型候选上做；
- 当前建议使用 `mtad_gat_vae`；
- 若主模型比较结果显示 `mtad_gat_sparsemax_vae` 更适合作为最终主模型，可追加在该模型上复核一轮。

### 8.3 正式运行命令

```bash
python benchmark_models.py \
  --datasets SMD \
  --smd_groups 1-2,1-3,2-1,3-6 \
  --models mtad_gat_vae \
  --mtad_epochs 15 \
  --batch_size 256 \
  --use_gatv2 true \
  --seed 42
```

```bash
python benchmark_models.py \
  --datasets SMD \
  --smd_groups 1-2,1-3,2-1,3-6 \
  --models mtad_gat_vae \
  --mtad_epochs 15 \
  --batch_size 256 \
  --use_gatv2 false \
  --seed 42
```

### 8.4 需要形成的论文材料

- `use_gatv2=true` 与 `use_gatv2=false` 的结果对比表
- 各机器差异与平均差异
- 一段关于是否保留 GATv2 默认设置的结论

## 9. 正式实验五：定性案例与可视化实验

### 9.1 目的

- 为论文正文提供直观案例；
- 展示模型对异常片段的定位能力；
- 支撑定量表格之外的可解释性分析。

### 9.2 推荐机器

- `1-3`：适合展示模型差异；
- `2-1` 或 `3-6`：适合展示相对稳定、较清晰的检测效果。

### 9.3 推荐模型

- `mtad_gat_gru`
- `mtad_gat_vae`

如后续主模型确认是 `mtad_gat_sparsemax_vae`，则可替换其中一个模型。

### 9.4 需要形成的论文材料

- 异常分数曲线
- 真实异常区间
- 阈值线
- 2 到 4 张最终论文插图
- 一段案例分析文字

## 10. 可选补充：传统基线实验

如果论文需要体现“与传统方法比较”，可补充以下模型：

- `pca`
- `isolation_forest`

建议单独执行，不与结构消融主线混在同一张表里，避免表格叙事混乱。

### 10.1 建议命令

```bash
python benchmark_models.py \
  --datasets SMD \
  --smd_groups 1-2,1-3,2-1,3-6 \
  --models pca,isolation_forest,mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae \
  --mtad_epochs 15 \
  --batch_size 256 \
  --use_gatv2 true \
  --seed 42
```

### 10.2 使用建议

- 若论文篇幅有限，可将传统基线作为补充表或附录表；
- 若需要突出改进方法优于传统方法，可单独做一张“传统基线 vs 主模型”表。

## 11. 快速调试实验与正式实验的边界

### 11.1 快速调试允许使用

- `--max_train_size`
- `--max_test_size`
- 较少的 `mtad_epochs`
- 较少的机器数量

### 11.2 正式实验禁止使用

- 在最终论文结果对应命令中加入 `--max_train_size`
- 在最终论文结果对应命令中加入 `--max_test_size`
- 把调试运行结果混入正式实验结论

### 11.3 调试命令示例

```bash
python benchmark_models.py \
  --datasets SMD \
  --smd_groups 1-2 \
  --models mtad_gat_gru,mtad_gat_vae \
  --max_train_size 2000 \
  --max_test_size 2000 \
  --mtad_epochs 3 \
  --batch_size 256 \
  --seed 42
```

该命令仅用于检查脚本是否跑通，不得进入论文正式结果。

## 12. 每次正式实验必须保存的文件

每次正式运行后，至少保留：

- `run_config.json`
- `benchmark_results.csv`
- `best_per_entity.csv`
- 各模型目录下的 `metrics.json`

如有定性分析或图像输出，额外保留：

- 可视化图片
- 中间分数文件
- 手工筛选案例记录

## 13. 最终需要汇总出的论文结果

全部实验完成后，至少需要整理出：

- 主模型比较总表
- 多 seed 稳定性汇总表
- 结构消融汇总表
- GATv2 消融汇总表
- 最终案例可视化图

若补传统基线，还需整理：

- 传统基线与主模型比较表

## 14. 论文表格建议

### 表 1：主模型比较表

- 行：`mtad_gat_gru`、`mtad_gat_vae`、`mtad_gat_sparsemax_vae`、`mtad_gat_sparsemax_vae_node`
- 列：`1-2`、`1-3`、`2-1`、`3-6`、`Average`
- 指标：待结果汇总后确定主指标对应列，同时可附辅指标结果

### 表 2：多随机种子稳定性表

- 行：4 个 MTAD-GAT 变体
- 列：各阈值方法下的 `mean +- std`

### 表 3：结构消融表

- 行：`GRU`、`VAE`、`VAE + Sparsemax`、`VAE + Sparsemax + Node Embedding`
- 内容：各设置在正式实验数据上的结果汇总

### 表 4：GATv2 消融表

- 行：`use_gatv2=false`、`use_gatv2=true`
- 内容：同一模型在两种设置下的结果比较

### 可选表 5：传统基线比较表

- 行：`pca`、`isolation_forest`、主模型候选
- 作用：展示改进模型与传统方法的差异

## 15. 论文图片建议

### 图 1：主模型异常分数图

- 机器：`1-3`
- 内容：`anomaly score + threshold + 真实异常区间`

### 图 2：模型对比例图

- 同一台机器上比较 `mtad_gat_gru` 与主模型候选

### 图 3：GATv2 消融图

- 展示 `use_gatv2=true` 与 `use_gatv2=false` 的差异

### 图 4：平均性能图

- 展示多个模型在 4 台机器上的平均表现

## 16. 推荐执行顺序

1. 跑正式主模型对比实验；
2. 跑多随机种子稳定性实验；
3. 基于已有结果整理结构消融表；
4. 跑 GATv2 消融实验；
5. 视论文需要补传统基线实验；
6. 生成定性案例图并整理正文图表。

## 17. 最小可交付版本

若时间紧张，论文正式实验的最小可交付版本包括：

- 一组正式主模型比较实验：`1-2,1-3,2-1,3-6`
- 三个 seed 的稳定性验证
- 一组 GATv2 消融实验
- 两张定性案例图
- 四张核心表格

## 18. 当前阶段注意事项

- 当前阶段先不要在计划文件中写死唯一主模型；
- 当前阶段先不要在计划文件中写死唯一主指标；
- 当前阶段先保证正式实验设置统一、命令可复现、输出文件完整；
- 等所有正式结果齐全后，再确定论文最终的主模型口径和指标口径。
