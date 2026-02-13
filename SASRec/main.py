import torch
import os
from train import train
from evaluate import evaluate


def main():

    params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_path": "recommendation_data.h5",
        "ckpt": "sasrec.pt",

        "max_len": 50,  # n=50 (序列最大长度)
        "d": 64,        # d=64 (嵌入维度)
        "num_blocks": 2,# 2层Self-Attention堆叠
        "num_heads": 1, # 单头Attention
        "dropout": 0.2, # Dropout率
        "lr": 0.001,    # 学习率
        "batch_size": 128,     # 训练批大小
        "eval_batch_size": 256,# 评估批大小
        "epochs": 50,          # 训练轮数
    }

    print("\n>>> 训练")
    train(params)

    # 运行评估
    print("\n>>> 评估")
    evaluate(params)


if __name__ == "__main__":
    main()