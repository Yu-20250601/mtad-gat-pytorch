# MTAD-GAT 对比试验与消融实验计划

## 1. 实验范围

本轮实验仅使用 `SMD` 数据集，主实验机器固定为：

- `1-2`
- `1-3`
- `2-1`
- `3-6`

这样可以兼顾：

- 不同 machine group 的代表性；
- 总实验量可控；
- 后续表格整理更清晰。

## 2. 对比实验与消融实验的关系

你前面问的“消融实验是不是已经包含了对比实验内容”，答案是：

- 是，包含了很大一部分；
- 但最好仍然保留一个明确的“基线 vs 最终方案”主对比。

建议这样理解：

- 对比实验：回答“最终方案整体是否优于基线”；
- 消融实验：回答“最终方案中到底是哪一部分起作用”。

因此本计划把两者合并组织，但保留一个主对比组。

## 3. 本次只做的 4 类核心因素

### 3.1 GRU vs VAE

对应参数：

- `--recon_model gru`
- `--recon_model vae`
- `--vae_latent_dim`
- `--kl_beta`

对应代码：

- `C:\Users\86199\Desktop\mtad-gat-pytorch-1\args.py`
- `C:\Users\86199\Desktop\mtad-gat-pytorch-1\mtad_gat.py`
- `C:\Users\86199\Desktop\mtad-gat-pytorch-1\modules.py`

### 3.2 GATv2 vs GAT

对应参数：

- `--use_gatv2 True`
- `--use_gatv2 False`

对应代码：

- `C:\Users\86199\Desktop\mtad-gat-pytorch-1\args.py`
- `C:\Users\86199\Desktop\mtad-gat-pytorch-1\modules.py`

### 3.3 sparsemax vs softmax

当前实现中：

- `FeatureAttentionLayer` 使用 `softmax`
- `AdaptiveSparseGAT` 使用 `Sparsemax`

因此可直接通过下面两个配置完成对比：

- `--use_adaptive_sparse_feat_gat False` -> softmax
- `--use_adaptive_sparse_feat_gat True` -> sparsemax

对应代码：

- `C:\Users\86199\Desktop\mtad-gat-pytorch-1\modules.py`

### 3.4 节点嵌入有 / 无

我已补充显式开关：

- `--use_node_embedding True`
- `--use_node_embedding False`

对应代码：

- `C:\Users\86199\Desktop\mtad-gat-pytorch-1\args.py`
- `C:\Users\86199\Desktop\mtad-gat-pytorch-1\modules.py`
- `C:\Users\86199\Desktop\mtad-gat-pytorch-1\mtad_gat.py`
- `C:\Users\86199\Desktop\mtad-gat-pytorch-1\train.py`

## 4. 基线与最终方案

### 4.1 基线 E1

定义为：

- Dense feature attention
- softmax
- GRU reconstruction
- GATv2

对应含义：

- `--use_adaptive_sparse_feat_gat False`
- `--recon_model gru`
- `--use_gatv2 True`

这组作为统一基线。

### 4.2 最终方案 E2

定义为：

- Sparse feature attention
- sparsemax
- node embedding
- VAE reconstruction
- GATv2

对应含义：

- `--use_adaptive_sparse_feat_gat True`
- `--use_node_embedding True`
- `--node_embed_dim 16`
- `--recon_model vae`
- `--vae_latent_dim 64`
- `--kl_beta 0.001`
- `--use_gatv2 True`
- `--export_sparse_attention True`

这组既是最终方案，也是后续消融母体。

## 5. 推荐实验矩阵

| ID | 特征注意力 | 重构头 | GAT 类型 | 节点嵌入 | 作用 |
|---|---|---|---|---|---|
| E1 | Dense / softmax | GRU | GATv2 | 无 | 基线 |
| E2 | Sparse / sparsemax | VAE | GATv2 | 有 | 最终方案 |
| E3 | Sparse / sparsemax | GRU | GATv2 | 有 | 消融：GRU vs VAE |
| E4 | Sparse / sparsemax | VAE | GAT | 有 | 消融：GATv2 vs GAT |
| E5 | Dense / softmax | VAE | GATv2 | 无 | 消融：sparsemax vs softmax |
| E6 | Sparse / sparsemax | VAE | GATv2 | 无 | 消融：节点嵌入有/无 |

## 6. 运行建议

### 6.1 主对比实验

`E1` 和 `E2` 在以下 4 台机器全跑：

- `1-2`
- `1-3`
- `2-1`
- `3-6`

### 6.2 消融实验

`E3` 到 `E6` 先只在以下 2 台机器上跑：

- `1-2`
- `3-6`

如果趋势稳定，再补：

- `1-3`
- `2-1`

这样更符合“消融实验不用这么多”的要求。

## 7. 评价指标

建议主表统一汇报 `epsilon_result`：

- F1
- Precision
- Recall
- TP / FP / FN
- Latency

附录可补充：

- `pot_result`
- `bf_result`

## 8. RCA 实验安排

RCA 建议只在下面两组上做：

- `E2`：Sparse + NodeEmbed + VAE + GATv2
- `E6`：Sparse + no NodeEmbed + VAE + GATv2

原因：

- 两组都能导出稀疏注意力；
- 正好回答“节点嵌入是否提升根因定位能力”。

RCA 指标：

- Hit@1
- Hit@3
- Hit@5

## 9. 结果表建议

### 9.1 主结果表

| Machine | Baseline F1 | Proposed F1 | Baseline Recall | Proposed Recall |
|---|---:|---:|---:|---:|

### 9.2 消融表

| ID | Sparsemax | Node Embedding | Recon | GATv2 | F1 | Precision | Recall |
|---|---:|---:|---|---:|---:|---:|---:|

### 9.3 RCA 表

| Machine | Method | Hit@1 | Hit@3 | Hit@5 |
|---|---|---:|---:|---:|

## 10. 写作顺序建议

最后整理实验章节时，建议按下面顺序写：

1. `E1 vs E2`：证明最终方案整体有效；
2. `E3 ~ E6`：解释提升来自哪些模块；
3. `E2 vs E6` 的 RCA：说明节点嵌入不仅提升检测，也提升定位。
