import os
import subprocess
import datetime
import json

# 配置参数
datasets = ["SMD"]  # 可以添加 "MSL", "SMAP"
smd_groups = ["1-1", "1-2", "1-3"]  # SMD 数据集的不同机器

# 要测试的配置组合
configs = [
    # (use_gatv2, recon_model, use_sparsemax, use_node_embedding)
    (True, "gru", False, False),  # GATv2 + GRU + softmax
    (True, "gru", True, False),   # GATv2 + GRU + sparsemax
    (True, "gru", False, True),   # GATv2 + GRU + softmax + embedding
    (True, "gru", True, True),    # GATv2 + GRU + sparsemax + embedding
    (True, "vae", False, False),  # GATv2 + VAE + softmax
    (True, "vae", True, False),   # GATv2 + VAE + sparsemax
    (True, "vae", False, True),   # GATv2 + VAE + softmax + embedding
    (True, "vae", True, True),    # GATv2 + VAE + sparsemax + embedding
    (False, "gru", False, False), # GAT + GRU + softmax
    (False, "gru", True, False),  # GAT + GRU + sparsemax
    (False, "gru", False, True),  # GAT + GRU + softmax + embedding
    (False, "gru", True, True),   # GAT + GRU + sparsemax + embedding
    (False, "vae", False, False), # GAT + VAE + softmax
    (False, "vae", True, False),  # GAT + VAE + sparsemax
    (False, "vae", False, True),  # GAT + VAE + softmax + embedding
    (False, "vae", True, True),   # GAT + VAE + sparsemax + embedding
]

# 创建输出目录
output_dir = "anomaly_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 运行实验
for dataset in datasets:
    if dataset == "SMD":
        for group in smd_groups:
            for config in configs:
                use_gatv2, recon_model, use_sparsemax, use_node_embedding = config
                
                # 生成实验名称
                experiment_name = f"{dataset}_{group}_gatv2_{use_gatv2}_recon_{recon_model}_sparsemax_{use_sparsemax}_embed_{use_node_embedding}"
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(output_dir, f"{dataset}_{timestamp}")
                
                # 构建命令
                cmd = [
                    "python", "train.py",
                    "--dataset", dataset,
                    "--group", group,
                    "--use_gatv2", str(use_gatv2),
                    "--recon_model", recon_model,
                    "--use_sparsemax", str(use_sparsemax),
                    "--use_node_embedding", str(use_node_embedding),
                    "--comment", experiment_name
                ]
                
                print(f"Running experiment: {experiment_name}")
                print(f"Command: {' '.join(cmd)}")
                
                # 运行命令
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
                    
                    # 保存输出
                    output_file = os.path.join(save_path, "output.txt")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    
                    with open(output_file, "w") as f:
                        f.write(f"Command: {' '.join(cmd)}\n\n")
                        f.write("STDOUT:\n")
                        f.write(result.stdout)
                        f.write("\nSTDERR:\n")
                        f.write(result.stderr)
                    
                    print(f"Experiment completed. Output saved to: {output_file}")
                except Exception as e:
                    print(f"Error running experiment: {e}")
    else:
        # 对于 MSL 和 SMAP 数据集
        for config in configs:
            use_gatv2, recon_model, use_sparsemax, use_node_embedding = config
            
            # 生成实验名称
            experiment_name = f"{dataset}_gatv2_{use_gatv2}_recon_{recon_model}_sparsemax_{use_sparsemax}_embed_{use_node_embedding}"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f"{dataset}_{timestamp}")
            
            # 构建命令
            cmd = [
                "python", "train.py",
                "--dataset", dataset,
                "--use_gatv2", str(use_gatv2),
                "--recon_model", recon_model,
                "--use_sparsemax", str(use_sparsemax),
                "--use_node_embedding", str(use_node_embedding),
                "--comment", experiment_name
            ]
            
            print(f"Running experiment: {experiment_name}")
            print(f"Command: {' '.join(cmd)}")
            
            # 运行命令
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
                
                # 保存输出
                output_file = os.path.join(save_path, "output.txt")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                with open(output_file, "w") as f:
                    f.write(f"Command: {' '.join(cmd)}\n\n")
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                    f.write("\nSTDERR:\n")
                    f.write(result.stderr)
                
                print(f"Experiment completed. Output saved to: {output_file}")
            except Exception as e:
                print(f"Error running experiment: {e}")

print("All experiments completed!")
