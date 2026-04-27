# 异常检测结果汇总

## 当前结论

- 异常检测模块的核心实验结果已经基本齐全。
- 当前不缺必须补跑的新实验。
- 若仅从异常检测部分出发，接下来应重点进入论文表格整理、结果分析和图注撰写阶段。
- 唯一的可选补充项是：如果希望传统方法也有案例图，可额外给 `pca` 或 `isolation_forest` 生成 1 张可视化图，但这不是必须。

## 已有结果来源

- 主模型与传统基线对比：`C:\Users\86199\Desktop\20260427_093756`
- 多随机种子稳定性：`C:\Users\86199\Desktop\多随机种子验证稳定性`
- GATv2 消融：
  - `C:\Users\86199\Desktop\GATv2消融实验`
  - `C:\Users\86199\Desktop\20260427_101354`
  - `C:\Users\86199\Desktop\20260427_101614`
- 论文案例图：`C:\Users\86199\Desktop\case_figures`

## 表 1：主模型对比表（`bf_result` F1）

| Model | 1-2 | 1-3 | 2-1 | 3-6 | Average |
|---|---:|---:|---:|---:|---:|
| `mtad_gat_sparsemax_vae` | 0.9770 | 0.9763 | 0.9182 | 0.9956 | 0.9668 |
| `mtad_gat_gru` | 0.9723 | 0.9347 | 0.9457 | 0.9951 | 0.9619 |
| `mtad_gat_vae` | 0.9770 | 0.9472 | 0.9053 | 0.9962 | 0.9564 |
| `pca` | 0.8720 | 0.9743 | 0.9538 | 0.9789 | 0.9447 |
| `isolation_forest` | 0.4930 | 0.4162 | 0.8286 | 0.8279 | 0.6414 |

简要结论：

- 若主指标以 `bf_result` F1 为准，`mtad_gat_sparsemax_vae` 的平均表现最好。
- `mtad_gat_gru` 与 `mtad_gat_vae` 也非常接近，三者差距不大。
- `pca` 在部分机器上表现非常强，是不能忽略的传统基线。

## 表 2：多随机种子稳定性表

| Model | BF mean | BF std | Epsilon mean | Epsilon std |
|---|---:|---:|---:|---:|
| `mtad_gat_gru` | 0.9578 | 0.0289 | 0.8429 | 0.0912 |
| `mtad_gat_vae` | 0.9616 | 0.0337 | 0.7934 | 0.1845 |
| `mtad_gat_sparsemax_vae` | 0.9597 | 0.0349 | 0.8619 | 0.0959 |
| `mtad_gat_sparsemax_vae_node` | 0.9562 | 0.0351 | 0.8116 | 0.1958 |

简要结论：

- `bf_result` 下，4 个 MTAD 变体都很接近，整体差距不大。
- `epsilon_result` 下，`mtad_gat_sparsemax_vae` 最稳，`mtad_gat_gru` 次之。
- `mtad_gat_vae` 在 `bf_result` 上均值较高，但 `epsilon_result` 波动更大。

## 表 3：结构消融表

| Setting | BF mean | BF std | Epsilon mean | Epsilon std |
|---|---:|---:|---:|---:|
| `GRU` | 0.9578 | 0.0289 | 0.8429 | 0.0912 |
| `VAE` | 0.9616 | 0.0337 | 0.7934 | 0.1845 |
| `VAE + Sparsemax` | 0.9597 | 0.0349 | 0.8619 | 0.0959 |
| `VAE + Sparsemax + Node Embedding` | 0.9562 | 0.0351 | 0.8116 | 0.1958 |

简要结论：

- `GRU -> VAE` 在 `bf_result` 上略有收益，但在 `epsilon_result` 上不占优。
- 在 VAE 基础上加入 `Sparsemax` 后，`epsilon_result` 明显改善。
- `Node Embedding` 没有带来稳定提升，更适合作为消融结果而非主模型默认配置。

## 表 4：GATv2 消融表（`mtad_gat_vae`）

| GATv2 | 1-2 | 1-3 | 2-1 | 3-6 | BF mean | BF std | Epsilon mean | Epsilon std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `False` | 0.9766 | 0.9621 | 0.9192 | 0.9960 | 0.9635 | 0.0305 | 0.8637 | 0.0789 |
| `True` | 0.9774 | 0.9630 | 0.9098 | 0.9960 | 0.9616 | 0.0337 | 0.7934 | 0.1845 |

简要结论：

- 在当前选定的 SMD 子集上，`use_gatv2=true` 并没有稳定优于 `use_gatv2=false`。
- 两者在 `bf_result` 上总体接近，但 `use_gatv2=false` 的均值略高，波动也略小。
- 因此，论文中对 GATv2 的表述建议保持客观，不宜写成“显著提升”。

## 论文正文图片建议

建议正文使用以下两张图：

- `C:\Users\86199\Desktop\case_figures\2-1_mtad_gat_vae_paper.png`
- `C:\Users\86199\Desktop\case_figures\1-3_gru_vs_vae_paper.png`

建议附录或补充材料使用：

- `C:\Users\86199\Desktop\case_figures\1-3_mtad_gat_vae_paper.png`

## 现在还缺什么

从异常检测模块来看：

- 不缺必须补跑的实验结果。
- 接下来主要缺的是：
  - 论文用最终表格排版
  - 每张图的图注
  - 异常检测结果分析文字

## 可选补充（非必须）

如需进一步丰富异常检测章节，可选补充：

- 为 `pca` 或 `isolation_forest` 增加 1 张案例图
- 但这不是必须项，因为传统基线的数值对比已经齐全
