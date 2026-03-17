import torch
import os
from train import train
from evaluate import evaluate


def main():

    params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_path": "../data/user_item_interact.h5",
        "ckpt": "sasrec.pt",
        "max_len": 20,  # n=50 (序列最大长度)
        "d": 16,        # d=64 (嵌入维度)
        "num_blocks": 2,# 2层Self-Attention堆叠
        "num_heads": 1, # 单头Attention
        "dropout": 0.2, # Dropout率
        "lr": 0.0001,    # 学习率
        "batch_size": 128,     # 训练批大小
        "eval_batch_size": 128,# 评估批大小
        "epochs": 5,          # 训练轮数
        
        # 从model.py中提取的参数
        "mlp_layer": 64,              # MLP隐藏层维度
        "layernorm_eps": 1e-8,         # LayerNorm epsilon值
        "num_neg_samples": 1,          # 负采样数量
        
        # 从train.py中提取的参数
        "num_workers": 2,              # DataLoader工作进程数
        "adam_betas": (0.9, 0.98),     # Adam优化器beta参数
        "loss_eps": 1e-24,             # BCE Loss epsilon值
        
        # 从evaluate.py中提取的参数
        "top_k": 10,                   # 评估使用的Top-K
        
        # 从data_vision.py中提取的参数
        "min_seq_len": 3,              # 最小序列长度过滤
    }

    print("\n>>> 训练")
    train(params)

    # 运行评估
    print("\n>>> 评估")
    evaluate(params)


if __name__ == "__main__":
    main()