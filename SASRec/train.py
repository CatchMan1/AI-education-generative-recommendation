import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model import SASRec
from data_vision import SASRecDataset
import sys
import logging
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    return torch.tensor(neg_samples_np, device=seq.device, dtype=torch.long)


def evaluate(model, test_loader, params, device):
    model.eval()
    topk_list = params['topk_list']
    hits  = {k: [] for k in topk_list}
    ndcgs = {k: [] for k in topk_list}
    with torch.no_grad():
        for input_ids, target_item in test_loader:
            input_ids = input_ids.to(device)
            target_item = target_item.to(device)
            valid_mask = target_item != 0
            if not valid_mask.any():
                continue
            logits = model.predict(input_ids[valid_mask])
            logits[:, 0] = -1e9
            target_scores = logits.gather(1, target_item[valid_mask].unsqueeze(1))
            ranks = (logits > target_scores).sum(dim=1) + 1
            for r in ranks.cpu().numpy():
                for k in topk_list:
                    hits[k].append(1 if r <= k else 0)
                    ndcgs[k].append(1 / np.log2(r + 1) if r <= k else 0)
    model.train()
    avg_hits  = {k: float(np.mean(v)) if v else 0.0 for k, v in hits.items()}
    avg_ndcgs = {k: float(np.mean(v)) if v else 0.0 for k, v in ndcgs.items()}
    return avg_hits, avg_ndcgs


def eval_loss(model, test_loader, item_num, params, device):
    model.eval()
    total_loss, total_valid = 0.0, 0
    eps = params['loss_eps']
    with torch.no_grad():
        for input_ids, target_item in test_loader:
            input_ids  = input_ids.to(device)
            target_item = target_item.to(device)
            valid_mask  = target_item != 0
            if not valid_mask.any():
                continue
            inp = input_ids[valid_mask]
            tgt = target_item[valid_mask]
            h   = model.forward(inp)[:, -1, :]          # [B', d]
            pos_score = (h * model.item_emb(tgt)).sum(-1)   # [B']
            neg = get_neg_samples(inp, item_num, num_neg=1).squeeze(-1)
            neg_score = (h * model.item_emb(neg)).sum(-1)   # [B']
            loss = (-torch.log(torch.sigmoid(pos_score) + eps)
                    - torch.log(1 - torch.sigmoid(neg_score) + eps)).sum()
            total_loss  += loss.item()
            total_valid += valid_mask.sum().item()
    model.train()
    return total_loss / total_valid if total_valid > 0 else 0.0


def train(params):
    log_path = os.path.abspath(params['log_path'])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Configuration: {params}")
    device = torch.device(params['device'])
    dataset = SASRecDataset(params["data_path"], max_len=params["max_len"], mode='train', params=params)
    test_dataset = SASRecDataset(params["data_path"], max_len=params["max_len"], mode='test', params=params)
    num_workers = 0 if device.type == 'cpu' and 'win' in sys.platform.lower() else params.get('num_workers', 4)
    dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=params["eval_batch_size"], shuffle=False, num_workers=num_workers)

    item_num = dataset.item_num
    model = SASRec(item_num, params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], betas=params.get('adam_betas', (0.9, 0.98)))

    topk_list = params['topk_list']
    top_k_main = topk_list[-1]
    best_ndcg = -1.0
    no_improve = 0
    patience = params['early_stop']
    train_losses = []
    val_losses   = []
    model.train()
    for epoch in range(1, params["epochs"] + 1):
        total_loss = 0.0
        total_valid_t = 0.0  # 统计非padding的时间步数量

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{params['epochs']}", leave=True)
        for step, (input_seqs, target_o_t) in enumerate(pbar):
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

            # 3. 生成负样本: [B, num_neg]
            neg_samples = get_neg_samples(input_seqs, item_num, num_neg=params['num_neg_samples'])

            # 4. 计算BCE Loss
            # 生成mask，过滤padding位置（o_t=0的位置）
            mask = (target_o_t != 0).float()  # [B, seq_len]
            seq_len = score_matrix.shape[1]

            # neg_samples: [B, num_neg] → [B, seq_len, num_neg]
            neg_samples_expanded = neg_samples.unsqueeze(1).expand(-1, seq_len, -1)

            # 提取正样本得分: [B, seq_len]
            pos_scores = torch.gather(score_matrix, dim=2, index=target_o_t.unsqueeze(-1)).squeeze(-1)
            # 提取负样本得分: [B, seq_len, num_neg]
            neg_scores = torch.gather(score_matrix, dim=2, index=neg_samples_expanded)

            # 计算正负样本的BCE Loss
            eps = params['loss_eps']
            pos_loss = -torch.log(torch.sigmoid(pos_scores) + eps) * mask                         # [B, seq_len]
            neg_loss = (-torch.log(1 - torch.sigmoid(neg_scores) + eps) * mask.unsqueeze(-1)).sum(dim=-1)  # [B, seq_len]

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
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 打印Epoch Loss
        if total_valid_t > 0:
            avg_loss = total_loss / total_valid_t
        else:
            avg_loss = 0.0
        val_loss = eval_loss(model, test_loader, item_num, params, device)
        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        avg_hits, avg_ndcgs = evaluate(model, test_loader, params, device)
        cur_ndcg = avg_ndcgs[top_k_main]
        improved = " *" if cur_ndcg > best_ndcg else ""
        print(f"Epoch {epoch:>3} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}{improved}")
        for k in topk_list:
            print(f"  Hit@{k}: {avg_hits[k]:.4f}  NDCG@{k}: {avg_ndcgs[k]:.4f}")
        logging.info(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Hits: {avg_hits} | NDCGs: {avg_ndcgs}")
        if cur_ndcg > best_ndcg:
            best_ndcg = cur_ndcg
            no_improve = 0
            torch.save(model.state_dict(), params["ckpt"])
            logging.info(f"Best model saved (NDCG@{top_k_main}={best_ndcg:.4f}) to {params['ckpt']}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"早停：Epoch {epoch}，NDCG@{top_k_main} 连续 {patience} 次未提升")
                logging.info(f"Early stopping at epoch {epoch}.")
                break

    print(f"训练完成，最优 NDCG@{top_k_main}: {best_ndcg:.4f}，权重已保存至 {params['ckpt']}")
    logging.info(f"Training complete. Best NDCG@{top_k_main}={best_ndcg:.4f}")

    plot_path = params['loss_plot_path']
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1),   val_losses,   marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logging.info(f"Loss plot saved to {plot_path}")
    print(f"Loss plot saved to: {plot_path}")
