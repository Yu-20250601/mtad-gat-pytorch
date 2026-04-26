# 论文实验计划

## 当前结论

- 当前主实验数据固定为 SMD 的 4 组机器：`1-2`、`1-3`、`2-1`、`3-6`
- 当前主模型方向：`mtad_gat_vae`
- 后续实验默认设置：`--use_gatv2 true`
- 论文中的指标使用建议：
  - 主指标：`bf_result` 的 F1
  - 辅指标：`epsilon_result` 的 F1
  - 补充分析：`pot_result` 的 F1

## 数据选择依据

论文主实验使用 SMD 的 `1-2`、`1-3`、`2-1`、`3-6` 四组机器。

选择标准：

- 覆盖不同 SMD 大组：`1-*`、`2-*`、`3-*`
- 避开异常过于稀疏或过于容易的极端机器
- 保证异常比例和异常片段数量处于相对适中的范围
- 在保证代表性的同时，控制实验规模，方便后续稳定复现实验

可直接写入论文的方法描述：

> 本文从 SMD 数据集中选取 `1-2`、`1-3`、`2-1` 和 `3-6` 四组具有代表性的机器进行实验。所选机器覆盖不同 machine group，并尽量避免异常过于稀疏或过于简单的极端情况，以保证实验的代表性与稳定性。

## 待做事项

### 1. 主模型比较实验

目的：

- 在选定的 4 组 SMD 机器上比较主要 MTAD-GAT 变体
- 确认后续论文主模型是否定为 `mtad_gat_vae`

运行命令：

```bash
python benchmark_models.py \
  --datasets SMD \
  --smd_groups 1-2,1-3,2-1,3-6 \
  --models mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node \
  --max_train_size 10000 \
  --max_test_size 20000 \
  --mtad_epochs 15 \
  --batch_size 256 \
  --use_gatv2 true \
  --seed 42
```

需要得到的结果：

- 每台机器上每个模型的 `F1 / Precision / Recall`
- 每个模型对应的 `bf_result / epsilon_result / pot_result`
- 4 台机器上的平均表现
- 最终主模型结论

本实验需要保留：

- `benchmark_results.csv`
- `best_per_entity.csv`
- 各模型目录下的 `metrics.json`

### 2. 多随机种子稳定性验证

目的：

- 验证模型排序不是单次随机结果造成的
- 判断 `mtad_gat_vae` 是否仍然是整体最均衡的选择

运行命令：

```bash
python benchmark_models.py \
  --datasets SMD \
  --smd_groups 1-2,1-3,2-1,3-6 \
  --models mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node \
  --max_train_size 10000 \
  --max_test_size 20000 \
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
  --max_train_size 10000 \
  --max_test_size 20000 \
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
  --max_train_size 10000 \
  --max_test_size 20000 \
  --mtad_epochs 15 \
  --batch_size 256 \
  --use_gatv2 true \
  --seed 62
```

需要得到的结果：

- 每个模型 `bf_result` F1 的 `mean +- std`
- 每个模型 `epsilon_result` F1 的 `mean +- std`
- 各机器上的稳定性观察
- 主模型结论是否跨 seed 保持一致

本实验需要保留：

- 三次运行的 `benchmark_results.csv`
- 三次运行的 `run_config.json`
- 最终手工汇总或脚本汇总后的均值标准差表

### 3. 结构消融实验

目的：

- 解释为什么最终选择该主模型
- 量化 VAE 重构头、Sparsemax 注意力、Node Embedding 三部分的作用

参与对比的模型：

- `mtad_gat_gru`
- `mtad_gat_vae`
- `mtad_gat_sparsemax_vae`
- `mtad_gat_sparsemax_vae_node`

说明：

- 本实验可以直接复用“多随机种子稳定性验证”的结果
- 如果第 2 步做完，这一步通常不需要额外重新跑

需要得到的结果：

- `GRU -> VAE` 的效果变化
- `VAE -> Sparsemax VAE` 的效果变化
- `Sparsemax VAE -> Sparsemax VAE + Node Embedding` 的效果变化
- 每个组件作用的一段简洁文字结论

本实验需要产出：

- 一张消融表
- 一段模块贡献分析文字

### 4. GATv2 消融实验

目的：

- 验证 `use_gatv2=true` 是否确实优于 `use_gatv2=false`

建议设置：

- 只在主模型 `mtad_gat_vae` 上做
- 其他设置完全保持不变

运行命令：

```bash
python benchmark_models.py \
  --datasets SMD \
  --smd_groups 1-2,1-3,2-1,3-6 \
  --models mtad_gat_vae \
  --max_train_size 10000 \
  --max_test_size 20000 \
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
  --max_train_size 10000 \
  --max_test_size 20000 \
  --mtad_epochs 15 \
  --batch_size 256 \
  --use_gatv2 false \
  --seed 42
```

需要得到的结果：

- `use_gatv2=true` 与 `use_gatv2=false` 的 `bf_result` F1 对比
- `use_gatv2=true` 与 `use_gatv2=false` 的 `epsilon_result` F1 对比
- 各机器及平均表现差异

本实验需要产出：

- 一张 GATv2 消融表
- 一段关于是否保留 `use_gatv2=true` 的结论

### 5. 定性案例与可视化实验

目的：

- 为论文提供直观图示
- 展示模型如何定位异常片段

建议选取的机器：

- `1-3`：适合展示模型差异
- `2-1` 或 `3-6`：适合展示相对稳定、清晰的检测结果

建议对比的模型：

- `mtad_gat_gru`
- `mtad_gat_vae`

需要得到的结果：

- 异常分数随时间变化曲线
- 真实异常区间
- 阈值线
- 2 到 4 个有代表性的案例图

本实验需要产出：

- 2 到 4 张最终论文插图
- 一段案例分析文字

## 每次实验必须保存的结果文件

每次实验完成后，建议至少保留：

- `run_config.json`
- `benchmark_results.csv`
- `best_per_entity.csv`
- 各模型目录下的 `metrics.json`
- 若有可视化结果，也保存对应图片或中间分数文件

## 最终需要汇总出的结果

全部实验完成后，需要整理出以下内容：

- 主模型比较总表
- 多 seed 稳定性汇总表
- 结构消融汇总表
- GATv2 消融汇总表
- 最终案例可视化图

## 论文中建议放入的表格

### 表 1：主模型比较表

建议内容：

- 行：`mtad_gat_gru`、`mtad_gat_vae`、`mtad_gat_sparsemax_vae`、`mtad_gat_sparsemax_vae_node`
- 列：`1-2`、`1-3`、`2-1`、`3-6`、`Average`
- 指标：以 `bf_result` F1 为主

作用：

- 展示所选 4 组机器上的主对比结果

### 表 2：稳定性表

建议内容：

- 行：4 个 MTAD-GAT 变体
- 列：`bf_result mean +- std`、`epsilon_result mean +- std`

作用：

- 展示不同模型跨 seed 的稳定性

### 表 3：结构消融表

建议内容：

- `mtad_gat_gru`
- `mtad_gat_vae`
- `mtad_gat_sparsemax_vae`
- `mtad_gat_sparsemax_vae_node`

作用：

- 解释各结构组件的贡献

### 表 4：GATv2 消融表

建议内容：

- `mtad_gat_vae + use_gatv2=false`
- `mtad_gat_vae + use_gatv2=true`

作用：

- 说明保留 `use_gatv2=true` 的依据

## 论文中建议放入的图片

### 图 1：主模型异常分数图

建议内容：

- 机器：`1-3`
- 模型：`mtad_gat_vae`
- 展示 anomaly score、threshold、真实异常区间

作用：

- 直观展示主模型对异常片段的检测效果

### 图 2：模型对比例图

建议内容：

- 同一台机器上，对比 `mtad_gat_gru` 与 `mtad_gat_vae`

作用：

- 展示主模型相较基础模型的优势

### 图 3：GATv2 消融柱状图

建议内容：

- 对比 `use_gatv2=true` 与 `use_gatv2=false`

作用：

- 直观说明 GATv2 的作用

### 图 4：平均性能柱状图

建议内容：

- 4 个模型在 4 台机器上的平均 F1

作用：

- 给出整体性能排序的直观展示

## 推荐执行顺序

1. 完成 3 个 seed 的主实验比较
2. 直接复用结果整理结构消融表
3. 跑 `mtad_gat_vae` 的 GATv2 消融实验
4. 生成 `1-3` 与 `2-1` 或 `3-6` 的案例图
5. 将表格和图整理进论文正文

## 最小可交付版本

如果时间比较紧，最小可交付实验包应包括：

- 一组主模型比较实验：`1-2,1-3,2-1,3-6`
- 三个 seed 的稳定性验证
- 一组 `mtad_gat_vae` 的 GATv2 消融实验
- 两张定性案例图
- 四张核心表格
