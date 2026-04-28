# 论文实验计划

## 1. 目标

本计划用于当前毕业论文的正式实验阶段，优先目标是：

- 先完成一套可以直接写入论文的主实验结果；
- 数据集固定为 `SMD`；
- 主评价指标固定为 `POT`，即以 `pot_result` 作为论文主表汇报指标；
- 先只选择 3 台机器，保证实验量可控、结果可解释、命令可一次性跑完。

## 2. 当前确定的论文口径

### 2.1 数据集

- 数据集：`SMD`
- 机器数量：`3` 台
- 最终选用机器：`1-2`、`2-1`、`3-2`

### 2.2 主评价指标

- 主指标：`POT` 对应的 `pot_result`
- 辅助指标：`epsilon_result`、`bf_result`
- 论文正文主表优先汇报：`F1 / Precision / Recall`（基于 `pot_result`）

## 3. 为什么选这 3 台机器

本轮选择基于已有筛选实验结果，而不是仅依据文献习惯。

### 3.1 选择依据

在当前结果中，按 `pot_result` 的最佳 F1 看：

- `2-1`：`0.9213`
- `3-2`：`0.9123`
- `1-2`：`0.9077`
- `3-6`：`0.7778`
- `1-1`：`0.7096`
- `1-3`：`0.1851`
- `3-7`：`0.1073`

因此，`1-2`、`2-1`、`3-2` 是当前最适合在 POT 口径下作为论文主实验机器的三台。

### 3.2 未选机器的原因

- `3-6`：在 `POT` 下 F1 明显低于前三台，且 precision 偏低，不作为当前论文主结果机；
- `1-1`：虽然是常见机器，但在本次 `POT` 结果下不如前三台；
- `1-3`：`POT` 下结果过低，不适合进入主表；
- `3-7`：`POT` 下整体表现过差，不适合作为论文主实验机器。

### 3.3 可直接写入论文的表述

> 本文在 SMD 数据集上选择 `machine-1-2`、`machine-2-1` 和 `machine-3-2` 作为正式实验对象。该选择基于预实验中不同机器在 POT 阈值评估口径下的表现：所选三台机器均取得较高且相对稳定的 F1 分数，能够在控制实验规模的同时保证结果具有代表性与可解释性。

## 4. 主实验模型

当前论文主实验固定比较以下 4 个 MTAD-GAT 变体：

- `mtad_gat_gru`
- `mtad_gat_vae`
- `mtad_gat_sparsemax_vae`
- `mtad_gat_sparsemax_vae_node`

说明：

- 这 4 个模型足以覆盖当前论文最关心的结构差异：`GRU / VAE / Sparsemax / Node Embedding`；
- 先完成这 4 个模型的正式对比，再决定是否补充传统基线。

## 5. 正式实验固定设置

除非后续单独说明，当前论文正式实验统一使用：

- `dataset=SMD`
- `smd_groups=1-2,2-1,3-2`
- `models=mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node`
- `lookback=100`
- `batch_size=256`
- `mtad_epochs=15`
- `use_gatv2=true`
- `seed=42`
- 不使用 `--max_train_size`
- 不使用 `--max_test_size`

## 6. 一次性正式运行命令

### 6.1 前台运行命令

```powershell
python benchmark_models.py --datasets SMD --smd_groups 1-2,2-1,3-2 --models mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node --mtad_epochs 15 --batch_size 256 --use_gatv2 true --seed 42 --output_root thesis_runs_smd_pot_3machines
```

### 6.2 后台运行命令（推荐）

```powershell
Start-Process -FilePath python -ArgumentList 'benchmark_models.py','--datasets','SMD','--smd_groups','1-2,2-1,3-2','--models','mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node','--mtad_epochs','15','--batch_size','256','--use_gatv2','true','--seed','42','--output_root','thesis_runs_smd_pot_3machines' -WorkingDirectory 'C:\Users\86199\Desktop\mtad-gat-pytorch-1' -RedirectStandardOutput 'C:\Users\86199\Desktop\mtad-gat-pytorch-1\thesis_runs_smd_pot_3machines.log' -RedirectStandardError 'C:\Users\86199\Desktop\mtad-gat-pytorch-1\thesis_runs_smd_pot_3machines.err.log' -WindowStyle Hidden
```

说明：

- 这条命令会在后台启动实验；
- 标准输出写入 `thesis_runs_smd_pot_3machines.log`；
- 错误输出写入 `thesis_runs_smd_pot_3machines.err.log`；
- 结果目录写入 `thesis_runs_smd_pot_3machines`。

## 7. 远程服务器一次性跑完论文实验与绘图

### 7.1 本轮远程任务范围

远程服务器端不只运行主对比实验，还要一次性完成：

- 主实验：4 个 MTAD-GAT 变体在 `1-2`、`2-1`、`3-2` 上的正式 benchmark；
- 稳定性实验：`seed=42,52,62`；
- `GATv2` 消融：在论文主模型上分别跑 `use_gatv2=true/false`；
- 详细输出实验：生成 `test_output.pkl`、`summary.txt`、`rca_attention` 等文件；
- 论文绘图：生成主表汇总图、按机器分组柱状图、案例图、异常检测图、稀疏注意力热力图。

### 7.2 统一入口脚本

仓库中新增统一流水线脚本：

- `run_thesis_pipeline.py`
- `aggregate_thesis_results.py`
- `bash_scripts/run_thesis_pipeline.sh`

其中：

- `run_thesis_pipeline.py` 负责依次调用 benchmark、详细训练、结果分析和案例绘图；
- `aggregate_thesis_results.py` 负责把结果汇总成论文可直接使用的表格与图；
- `bash_scripts/run_thesis_pipeline.sh` 用于在 Linux 服务器上一条命令启动整个流程。

### 7.3 GitHub 上传后在远程服务器执行的推荐命令

先在本地提交并推送到 GitHub，然后在远程服务器执行：

```bash
git clone <你的仓库地址>
cd mtad-gat-pytorch-1
pip install -r requirements_fixed.txt
nohup bash bash_scripts/run_thesis_pipeline.sh > thesis_pipeline.nohup.log 2>&1 &
```

如果服务器环境里默认 Python 不是 `python`，可以显式指定：

```bash
nohup env PYTHON_BIN=python3 bash bash_scripts/run_thesis_pipeline.sh > thesis_pipeline.nohup.log 2>&1 &
```

### 7.4 流水线默认输出

整套流程默认输出到：

- `thesis_artifacts/`

重点关注：

- `thesis_artifacts/manifest.json`
- `thesis_artifacts/logs/`
- `thesis_artifacts/paper_assets/tables/`
- `thesis_artifacts/paper_assets/figures/`

### 7.5 论文写作时优先使用的输出

- 主表：`thesis_artifacts/paper_assets/tables/main_pot_f1.csv`
- 主表补充指标：`thesis_artifacts/paper_assets/tables/main_pot_precision.csv`
- 主表补充指标：`thesis_artifacts/paper_assets/tables/main_pot_recall.csv`
- 阈值法辅助表：`thesis_artifacts/paper_assets/tables/aux_threshold_averages.csv`
- 稳定性表：`thesis_artifacts/paper_assets/tables/seed_stability_pot.csv`
- GATv2 消融表：`thesis_artifacts/paper_assets/tables/gatv2_ablation_pot.csv`
- 平均性能图：`thesis_artifacts/paper_assets/figures/pot_average_f1.png`
- 机器分组图：`thesis_artifacts/paper_assets/figures/pot_f1_by_machine.png`
- 案例图：`thesis_artifacts/paper_assets/figures/case_compare_1-2_pot.png`
- 案例图：`thesis_artifacts/paper_assets/figures/case_compare_3-2_pot.png`

## 8. 本轮实验需要保留的文件

每次正式运行后，至少保留：

- `run_config.json`
- `benchmark_results.csv`
- `best_per_entity.csv`
- 各模型目录下的 `metrics.json`
- 后台日志文件：
  - `thesis_runs_smd_pot_3machines.log`
  - `thesis_runs_smd_pot_3machines.err.log`

## 9. 论文中需要整理出的表格

### 表 1：主模型比较表

- 行：4 个 MTAD-GAT 变体
- 列：`1-2`、`2-1`、`3-2`、`Average`
- 主指标：`pot_result` 下的 `F1`
- 可附：`Precision`、`Recall`

### 表 2：辅助指标表

- 同样的模型与机器设置；
- 补充汇报 `epsilon_result` 与 `bf_result`；
- 作用是说明结论不完全依赖单一阈值法。

### 表 3：结构消融说明表

- `mtad_gat_gru`
- `mtad_gat_vae`
- `mtad_gat_sparsemax_vae`
- `mtad_gat_sparsemax_vae_node`

该表可直接由主实验结果整理得到，无需额外重复跑一套实验。

## 10. 当前阶段执行顺序

1. 先跑 3 台机器的正式主实验；
2. 汇总 `pot_result`，形成论文主表；
3. 再补充 `epsilon_result` 与 `bf_result` 作为辅助结果；
4. 根据主表结果决定是否追加第 4 台或更多机器。

## 11. 当前最小可交付版本

如果当前目标是先把论文写出来，那么最小可交付版本包括：

- `SMD` 上 `1-2`、`2-1`、`3-2` 三台机器；
- 4 个 MTAD-GAT 变体；
- 主指标固定为 `POT`；
- 一张主模型比较表；
- 一张辅助指标表；
- 一段关于机器选择依据的文字说明。
