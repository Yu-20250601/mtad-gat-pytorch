# 本科论文实验清单建议

适用题目：

`基于深度学习的多元时间序列异常检测与根因分析`

本文档的目标不是“把所有模型都跑一遍”，而是帮助你在时间有限的情况下，完成一套：

- 能支撑论文主线
- 实现成本低
- 对答辩友好
- 能按时收尾

的实验方案。

## 一、最终建议跑哪些模型

如果你的时间比较紧，我建议主对比实验先跑下面 5 个模型：

- `pca`
- `isolation_forest`
- `mtad_gat_gru`
- `mtad_gat_vae`
- `mtad_gat_sparsemax_vae`

这 5 个模型已经足够形成一条完整的论文叙事：

- `pca`：传统线性方法基线
- `isolation_forest`：传统无监督机器学习基线
- `mtad_gat_gru`：原仓库默认实现基线
- `mtad_gat_vae`：更接近论文设定的 MTAD-GAT 基线
- `mtad_gat_sparsemax_vae`：你的改进模型候选

## 二、什么时候再加 node embedding

如果你先做一轮小规模实验后，发现 `node embedding` 有稳定提升，再把它加入正式主实验。

建议比较的版本：

- `mtad_gat_vae`
- `mtad_gat_vae_node`
- `mtad_gat_sparsemax_vae`
- `mtad_gat_sparsemax_vae_node`

如果 `node embedding` 的提升不明显，或者只在单个数据集上有效，那么：

- 主对比实验里可以不放它
- 只在消融实验里报告即可

这在本科论文中是完全合理的。

## 三、GDN 要不要跑

如果你还有时间，并且想补一个“图模型外部基线”，再加：

- `gdn`

但如果时间明显不够，`gdn` 不是必须项。

优先级上，我建议：

1. 先确保 `pca / isolation_forest / mtad_gat_gru / mtad_gat_vae / mtad_gat_sparsemax_vae` 跑完
2. 再考虑是否补 `gdn`

## 四、推荐的数据集优先级

如果时间很紧，优先级建议如下：

1. `SMD`
2. `MSL`
3. `SMAP`

原因：

- `SMD` 是工业多变量场景里最常用的数据集之一，论文说服力强
- `MSL` 和 `SMAP` 常被一起使用，适合作为跨数据集补充验证

如果你只能做一个数据集：

- 首选 `SMD`

如果能做两个：

- `SMD + MSL`

如果时间足够再做完整版本：

- `SMD + MSL + SMAP`

## 五、推荐的实验顺序

### 第一步：快速筛选最终模型

先不要跑全量数据，先做小样本验证：

```bash
python benchmark_models.py --datasets SMD --smd_groups 1-1 --models mtad_gat_vae,mtad_gat_vae_node,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node --max_train_size 4000 --max_test_size 4000 --mtad_epochs 5
```

目的：

- 看 `sparsemax` 是否有效
- 看 `node embedding` 是否有效
- 决定“你的最终模型”到底是哪一个

### 第二步：跑正式主对比实验

如果 `node embedding` 没明显提升，推荐正式实验直接跑：

```bash
python benchmark_models.py --datasets SMD,MSL,SMAP --smd_groups all --models pca,isolation_forest,mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae
```

如果 `node embedding` 提升稳定，则改成：

```bash
python benchmark_models.py --datasets SMD,MSL,SMAP --smd_groups all --models pca,isolation_forest,mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae_node
```

### 第三步：补消融实验

推荐至少做以下 3 组消融：

1. `GRU vs VAE`
2. `softmax vs sparsemax`
3. `node embedding off/on`

对应模型可以直接这样跑：

```bash
python benchmark_models.py --datasets SMD --smd_groups all --models mtad_gat_gru,mtad_gat_vae,mtad_gat_sparsemax_vae,mtad_gat_sparsemax_vae_node
```

## 六、根因分析建议

如果时间不多，根因分析部分可以只做你自己的最终模型，不和其他模型做对比。

建议写法：

- “本文对所提出模型的根因定位能力进行案例分析”

不建议写法：

- “本文证明所提模型的根因分析能力优于所有对比模型”

因为 `SMD / MSL / SMAP` 更适合做异常检测评测，而不是严格的根因分析量化比较。

### 根因分析部分建议展示的内容

选择 3 到 5 个异常片段，展示：

- 全局异常分数曲线
- 异常片段前后的原始多变量曲线
- top-k 可疑变量排名
- sparse attention 或 feature attention 热图
- 如果启用了 node embedding，可补充变量间关联解释

## 七、论文里可以怎么定义 baseline

推荐这样写：

- `mtad_gat_gru`：原仓库默认实现基线
- `mtad_gat_vae`：更接近原论文设定的 MTAD-GAT 基线

这样最稳妥，也最容易回答老师的问题：

- 为什么同时有两个 MTAD-GAT baseline？

答案是：

- 一个对应当前工程实现
- 一个对应更接近原论文的方法设定

## 八、最推荐的最终版本

如果我替你做最终定稿选择，我会推荐：

### 主对比实验

- `pca`
- `isolation_forest`
- `mtad_gat_gru`
- `mtad_gat_vae`
- `mtad_gat_sparsemax_vae`

### 可选补充

- `mtad_gat_sparsemax_vae_node`
- `gdn`

### 消融实验

- `mtad_gat_gru`
- `mtad_gat_vae`
- `mtad_gat_sparsemax_vae`
- `mtad_gat_sparsemax_vae_node`

### 根因分析

- 只分析你最终选定的模型

## 九、一句话版建议

如果你现在时间真的不多，就照这个最小可行方案做：

- 主实验跑 `pca + isolation_forest + mtad_gat_gru + mtad_gat_vae + mtad_gat_sparsemax_vae`
- 先用小样本判断 `node embedding` 要不要保留
- 根因分析只做你自己的最终模型

