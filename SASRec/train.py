import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model import SASRec
from data_vision import SASRecDataset
import sys

def get_neg_samples(seq, item_num, num_neg=1):
    """
    负采样：
    对每个序列，生成不在用户历史S^u中的负样本j∉S^u
    """
    neg_samples = []
    for s in seq.cpu().numpy():
        # 过滤padding和用户已交互物品
        valid_neg = np.setdiff1d(np.arange(1, item_num + 1), s[s != 0])
        # 随机选1个负样本
        neg = np.random.choice(valid_neg, num_neg, replace=False)
        neg_samples.append(neg)

    # 先转成单一numpy数组，再转tensor
    neg_samples_np = np.array(neg_samples)
    return torch.tensor(neg_samples_np, device=seq.device)


def train(params):
    device = torch.device(params['device'])
    dataset = SASRecDataset(params["data_path"], max_len=params["max_len"], mode='train')
    num_workers = 0 if device.type == 'cpu' and 'win' in sys.platform.lower() else 4
    dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, num_workers=num_workers)

    item_num = dataset.item_num
    model = SASRec(item_num, params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], betas=(0.9, 0.98))

    model.train()
    for epoch in range(1, params["epochs"] + 1):
        total_loss = 0.0
        total_valid_t = 0.0  # 统计非padding的时间步数量

        for step, (input_seqs, target_o_t) in enumerate(dataloader):
            """
            input_seqs: s = {s_1, s_2, ..., s_n} (固定长度n)
            target_o_t: o_t (每个时间步t的目标输出)
            """
            input_seqs = input_seqs.to(device)  # [B, n]
            target_o_t = target_o_t.to(device)  # [B, n]

            # 1. 前向传播获取所有时间步的特征 F_t^{(b)}
            seq_features = model.forward(input_seqs)  # [B, n, d]

            # 2. 计算所有时间步t的物品得分 r_{j,t} = h_t · M_j
            # 物品嵌入矩阵M: [item_num+1, d]
            item_emb_weight = model.item_emb.weight  # M矩阵
            # 得分矩阵: [B, n, item_num+1] = [B, n, d] @ [d, item_num+1]
            score_matrix = torch.matmul(seq_features, item_emb_weight.t())

            # 3. 生成负样本（每个时间步t随机选1个j∉S^u）
            neg_samples = get_neg_samples(input_seqs, item_num, num_neg=1)  # [B, 1]

            # 4. 计算BCE Loss
            # 生成mask，过滤padding位置（o_t=0的位置）
            mask = (target_o_t != 0).float()  # [B, n]
            batch_size, max_len = input_seqs.shape

            # 扩展负样本维度匹配score_matrix: [B, n]
            neg_samples_expanded = neg_samples.repeat(1, max_len)  # [B, n]

            # 提取正样本得分: [B, n]
            pos_scores = torch.gather(score_matrix, dim=2, index=target_o_t.unsqueeze(-1)).squeeze(-1)
            # 提取负样本得分: [B, n]
            neg_scores = torch.gather(score_matrix, dim=2, index=neg_samples_expanded.unsqueeze(-1)).squeeze(-1)

            # 计算正负样本的BCE Loss
            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-24) * mask
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-24) * mask

            # 计算batch总loss和有效时间步数
            batch_loss = (pos_loss + neg_loss).sum()
            batch_valid_t = mask.sum().item()

            # 5. 归一化Loss
            if batch_valid_t > 0:
                loss = batch_loss / batch_valid_t
            else:
                loss = torch.tensor(0.0, device=device)

            # 6. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计Loss和有效时间步
            total_loss += batch_loss.item()
            total_valid_t += batch_valid_t

        # 打印Epoch Loss
        if total_valid_t > 0:
            avg_loss = total_loss / total_valid_t
        else:
            avg_loss = 0.0
        if epoch % 10 == 0:
            print(f"训练轮数：{epoch} | 平均损失:{avg_loss:.4f}")

    # 保存模型权重
    torch.save(model.state_dict(), params["ckpt"])
    print(f"训练完成，权重保存至 {params['ckpt']}")
