# MTAD-GAT 实验命令清单

以下命令都在仓库根目录执行：

- `C:\Users\86199\Desktop\mtad-gat-pytorch-1`

## 1. 主对比实验

主对比实验在 4 台机器上全部运行：

- `1-2`
- `1-3`
- `2-1`
- `3-6`

### E1 基线：Dense + GRU + GATv2

```powershell
python train.py --dataset SMD --group 1-2 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat False --recon_model gru --use_gatv2 True
python train.py --dataset SMD --group 1-3 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat False --recon_model gru --use_gatv2 True
python train.py --dataset SMD --group 2-1 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat False --recon_model gru --use_gatv2 True
python train.py --dataset SMD --group 3-6 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat False --recon_model gru --use_gatv2 True
```

### E2 最终方案：Sparse + NodeEmbed + VAE + GATv2

```powershell
python train.py --dataset SMD --group 1-2 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding True --node_embed_dim 16 --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 True --export_sparse_attention True
python train.py --dataset SMD --group 1-3 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding True --node_embed_dim 16 --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 True --export_sparse_attention True
python train.py --dataset SMD --group 2-1 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding True --node_embed_dim 16 --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 True --export_sparse_attention True
python train.py --dataset SMD --group 3-6 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding True --node_embed_dim 16 --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 True --export_sparse_attention True
```

## 2. 消融实验

建议先只在 `1-2` 和 `3-6` 上跑。

### E3：GRU vs VAE

```powershell
python train.py --dataset SMD --group 1-2 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding True --node_embed_dim 16 --recon_model gru --use_gatv2 True --export_sparse_attention True
python train.py --dataset SMD --group 3-6 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding True --node_embed_dim 16 --recon_model gru --use_gatv2 True --export_sparse_attention True
```

与 `E2` 对比即可完成 `GRU vs VAE`。

### E4：GATv2 vs GAT

```powershell
python train.py --dataset SMD --group 1-2 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding True --node_embed_dim 16 --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 False --export_sparse_attention True
python train.py --dataset SMD --group 3-6 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding True --node_embed_dim 16 --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 False --export_sparse_attention True
```

与 `E2` 对比即可完成 `GATv2 vs GAT`。

### E5：sparsemax vs softmax

```powershell
python train.py --dataset SMD --group 1-2 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat False --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 True
python train.py --dataset SMD --group 3-6 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat False --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 True
```

与 `E2` 对比即可完成 `sparsemax vs softmax`。

### E6：节点嵌入有 / 无

```powershell
python train.py --dataset SMD --group 1-2 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding False --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 True --export_sparse_attention True
python train.py --dataset SMD --group 3-6 --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding False --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 True --export_sparse_attention True
```

与 `E2` 对比即可完成“节点嵌入有 / 无”。

## 3. PowerShell 批量运行写法

### 3.1 主对比实验

```powershell
$groups = @("1-2", "1-3", "2-1", "3-6")

foreach ($g in $groups) {
  python train.py --dataset SMD --group $g --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat False --recon_model gru --use_gatv2 True
}

foreach ($g in $groups) {
  python train.py --dataset SMD --group $g --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding True --node_embed_dim 16 --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 True --export_sparse_attention True
}
```

### 3.2 消融实验

```powershell
$ablation_groups = @("1-2", "3-6")

# E3: GRU vs VAE
foreach ($g in $ablation_groups) {
  python train.py --dataset SMD --group $g --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding True --node_embed_dim 16 --recon_model gru --use_gatv2 True --export_sparse_attention True
}

# E4: GATv2 vs GAT
foreach ($g in $ablation_groups) {
  python train.py --dataset SMD --group $g --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding True --node_embed_dim 16 --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 False --export_sparse_attention True
}

# E5: sparsemax vs softmax
foreach ($g in $ablation_groups) {
  python train.py --dataset SMD --group $g --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat False --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 True
}

# E6: 节点嵌入有 / 无
foreach ($g in $ablation_groups) {
  python train.py --dataset SMD --group $g --epochs 30 --lookback 100 --bs 256 --init_lr 1e-3 --dropout 0.3 --val_split 0.1 --use_adaptive_sparse_feat_gat True --use_node_embedding False --recon_model vae --vae_latent_dim 64 --kl_beta 0.001 --use_gatv2 True --export_sparse_attention True
}
```

## 4. RCA 分析命令

### 4.1 分析某个 group 的最新一次输出

```powershell
python analyze_results.py --dataset SMD --group 1-2
python analyze_results.py --dataset SMD --group 3-6
```

### 4.2 指定某次输出目录

```powershell
python analyze_results.py --dataset SMD --group 1-2 --target_dir output/SMD/1-2/28042026_120000
```

### 4.3 让脚本自动从 target_dir 推断 group

```powershell
python analyze_results.py --dataset SMD --target_dir output/SMD/1-2/28042026_120000
```

### 4.4 手动指定标签文件

```powershell
python analyze_results.py --dataset SMD --group 1-2 --target_dir output/SMD/1-2/28042026_120000 --label_path datasets/ServerMachineDataset/interpretation_label/machine-1-2.txt
```

## 5. 建议统计顺序

1. 先汇总 `E1` 和 `E2` 在 4 台机器上的结果，形成主对比表；
2. 再汇总 `E3 ~ E6` 在 `1-2` 和 `3-6` 上的结果，形成消融表；
3. 最后对 `E2` 和 `E6` 跑 `analyze_results.py`，形成 RCA 表。
