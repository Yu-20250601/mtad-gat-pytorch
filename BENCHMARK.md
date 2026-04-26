# 对比实验说明

仓库中新增了 `benchmark_models.py`，用于统一运行以下异常检测模型的对比实验：

- `pca`
- `isolation_forest`
- `mtad_gat_gru`
- `mtad_gat_gru_node`
- `mtad_gat_sparsemax_gru`
- `mtad_gat_sparsemax_gru_node`
- `mtad_gat_vae`
- `mtad_gat_vae_node`
- `mtad_gat_sparsemax_vae`
- `mtad_gat_sparsemax_vae_node`
- `gdn`
- `mtad_gat_custom`

当前支持的数据集：

- `SMD`
- `MSL`
- `SMAP`

## 数据准备

当前工作区里的 `SMD` 预处理文件已经存在，可以直接运行。

`MSL` 和 `SMAP` 还需要先放置原始遥测数据，再执行预处理。请把原始数据放到 `datasets/data/` 下，然后运行：

```bash
python preprocess.py --dataset MSL
python preprocess.py --dataset SMAP
```

运行对比实验前，脚本会检查下列文件是否存在：

- `datasets/data/processed/MSL_train.pkl`
- `datasets/data/processed/MSL_test.pkl`
- `datasets/data/processed/MSL_test_label.pkl`
- `datasets/data/processed/SMAP_train.pkl`
- `datasets/data/processed/SMAP_test.pkl`
- `datasets/data/processed/SMAP_test_label.pkl`

## 常用命令

先做一个小规模快速验证，只跑 SMD 的 3 台机器：

```bash
python benchmark_models.py --datasets SMD --smd_groups 1-1,1-2,1-3 --max_train_size 2000 --max_test_size 2000 --gdn_epochs 3 --mtad_epochs 3
```

在 SMD 全部机器上跑完整对比：

```bash
python benchmark_models.py --datasets SMD --smd_groups all
```

说明：为了兼顾时间成本，脚本默认模型集合是：

- `pca`
- `isolation_forest`
- `mtad_gat_gru`
- `mtad_gat_vae`
- `mtad_gat_sparsemax_vae`

也就是说，默认不会自动跑 `gdn` 和 `node embedding` 版本，适合先做论文主实验。

在完成预处理后，跑 MSL 和 SMAP：

```bash
python benchmark_models.py --datasets MSL,SMAP
```

一次性跑 SMD、MSL、SMAP 三个数据集：

```bash
python benchmark_models.py --datasets SMD,MSL,SMAP --smd_groups all
```

如果你想只比较指定模型，也可以这样写：

```bash
python benchmark_models.py --datasets SMD --smd_groups all --models pca,isolation_forest,gdn,mtad_gat_gru,mtad_gat_sparsemax_gru,mtad_gat_vae,mtad_gat_sparsemax_vae
```

只跑默认 GRU 版 MTAD-GAT 与 sparsemax 版：

```bash
python benchmark_models.py --datasets SMD --smd_groups all --models mtad_gat_gru,mtad_gat_sparsemax_gru
```

只跑更接近论文的 VAE 版 MTAD-GAT 与 sparsemax 版：

```bash
python benchmark_models.py --datasets SMD --smd_groups all --models mtad_gat_vae,mtad_gat_sparsemax_vae
```

只跑 node embedding 消融：

```bash
python benchmark_models.py --datasets SMD --smd_groups all --models mtad_gat_vae,mtad_gat_vae_node,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node
```

如果你想用“自定义开关”的方式跑 MTAD-GAT，也可以使用 `mtad_gat_custom`：

```bash
python benchmark_models.py --datasets SMD --smd_groups all --models mtad_gat_custom --mtad_custom_recon_model vae --mtad_use_sparsemax true --mtad_use_node_embedding true --mtad_node_embed_dim 16
```

如果你想保留旧名称，也仍然可以使用：

- `mtad_gat` 等价于 `mtad_gat_gru`
- `mtad_gat_sparsemax` 等价于 `mtad_gat_sparsemax_gru`

## 输出结果

每次运行都会生成一个时间戳目录：

```text
benchmark_output/<时间戳>/
```

其中最重要的文件有：

- `benchmark_results.csv`：保存所有数据集、所有实体、所有模型、所有阈值方法的完整指标
- `best_per_entity.csv`：按数据实体汇总每种阈值方法下 F1 最好的模型
- `run_config.json`：本次运行使用的参数配置
- `.../<model>/metrics.json`：某个模型在某个数据集实体上的详细结果

## 结果字段说明

脚本会输出三套评估结果，和原仓库评测风格保持一致：

- `epsilon_result`：基于 TelemAnom 的 epsilon 阈值方法
- `pot_result`：基于 POT 的阈值方法
- `bf_result`：暴力搜索得到最佳 F1 的阈值方法

常见指标含义如下：

- `f1`：F1 分数
- `precision`：精确率
- `recall`：召回率
- `TP / TN / FP / FN`：混淆矩阵统计量
- `threshold`：对应阈值
- `latency`：异常检测延迟

## 说明

- `gdn` 是在本仓库里实现的一个“受 GDN 启发的轻量图预测基线”，这样可以直接纳入统一脚本中跑对比。
- `mtad_gat_gru` 是当前仓库默认的 MTAD-GAT 版本，重构头使用 GRU。
- `mtad_gat_gru_node` 是默认 GRU 重构头版本，并启用可学习节点嵌入。
- `mtad_gat_sparsemax_gru` 是默认 GRU 重构头版本，同时把特征注意力中的 softmax 替换为 sparsemax。
- `mtad_gat_sparsemax_gru_node` 是 GRU 重构头 + sparsemax + 可学习节点嵌入。
- `mtad_gat_vae` 是更接近论文设定的 MTAD-GAT 版本，重构头使用 VAE。
- `mtad_gat_vae_node` 是 VAE 重构头版本，并启用可学习节点嵌入。
- `mtad_gat_sparsemax_vae` 是 VAE 重构头版本，同时把特征注意力中的 softmax 替换为 sparsemax。
- `mtad_gat_sparsemax_vae_node` 是 VAE 重构头 + sparsemax + 可学习节点嵌入。
- `mtad_gat_custom` 会读取命令行参数 `--mtad_custom_recon_model`、`--mtad_use_sparsemax`、`--mtad_use_node_embedding` 生成一个自定义 MTAD-GAT 变体。
- `--mtad_node_embed_dim` 用于控制节点嵌入维度。
- 可以通过 `--mtad_vae_latent_dim` 调整 VAE 潜变量维度，通过 `--mtad_kl_beta` 调整 KL 损失权重。
- 如果只想快速检查流程是否可跑通，建议先加 `--max_train_size` 和 `--max_test_size` 做小样本验证；正式实验再去掉这两个参数。
