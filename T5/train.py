import torch
from typing import Dict, Any, List
import numpy as np
import random
from tqdm import tqdm
from model import TIGER
import logging
import torch.optim as optim
from data_vision import EmbDataset
from torch.utils.data import DataLoader
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F

def collate_emb_batch(batch: List[Dict[str, Any]]):
    seq_lens = [b['seq_len'] for b in batch] # 取出每个样本
    max_len = max(seq_lens)
    emb_dim = batch[0]['seq_embs'].shape[-1]
    seq_embs = torch.zeros((len(batch), max_len, emb_dim), dtype=torch.float32)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    target_emb = torch.stack([b['target_emb'] for b in batch], dim=0)
    target_id = torch.tensor([b['target_id'] for b in batch], dtype=torch.long)
    user_id = torch.tensor([b['user_id'] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        cur_len = b['seq_len']
        seq_embs[i, :cur_len, :] = b['seq_embs']
        attention_mask[i, :cur_len] = 1

    return {
        'seq_embs': seq_embs,
        'attention_mask': attention_mask,
        'target_emb': target_emb,
        'target_id': target_id,
        'user_id': user_id,
    }

def train_one_epoch(model, train_loader, optimizer, device, params, epoch=None):
    model.train()
    total_loss = 0.0
    desc = f"Epoch {epoch}/{params['num_epochs']}" if epoch is not None else "Training"
    for batch in tqdm(train_loader, desc=desc, leave=True):
        seq_embs = batch['seq_embs'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_emb = batch['target_emb'].to(device)

        optimizer.zero_grad()
        loss, _ = model(seq_embs=seq_embs, attention_mask=attention_mask, target_emb=target_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def eval_loss(model, eval_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="ValLoss"):
            seq_embs = batch['seq_embs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_emb = batch['target_emb'].to(device)
            loss, _ = model(seq_embs=seq_embs, attention_mask=attention_mask, target_emb=target_emb)
            total_loss += loss.item()
    return total_loss / len(eval_loader)

def evaluate(model, test_loader, item_embs_np, params, device):
    model.eval()
    item_embs_t = torch.tensor(item_embs_np, dtype=torch.float32, device=device)
    item_embs_t = torch.nn.functional.normalize(item_embs_t, p=2, dim=1)
    topk_list = params.get('topk_list', [5, 10, 20])
    recalls = {f'Recall@{k}': [] for k in topk_list}
    ndcgs   = {f'NDCG@{k}':   [] for k in topk_list}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", leave=False):
            seq_embs = batch['seq_embs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_id = batch['target_id'].to(device)
            _, pred_emb = model(seq_embs=seq_embs, attention_mask=attention_mask, target_emb=None)
            pred_emb = torch.nn.functional.normalize(pred_emb, p=2, dim=1)
            scores = pred_emb @ item_embs_t.T
            max_k  = max(topk_list)
            topk_idx = torch.topk(scores, k=max_k, dim=1).indices
            for k in topk_list:
                cur_topk = topk_idx[:, :k]
                hit = (cur_topk == target_id.unsqueeze(1)).any(dim=1).float()
                recalls[f'Recall@{k}'].append(hit.mean().item())
                ranks = torch.arange(1, k + 1, device=device).unsqueeze(0)
                match = (cur_topk == target_id.unsqueeze(1)).float()
                ndcg  = (match / torch.log2(ranks + 1)).sum(dim=1)
                ndcgs[f'NDCG@{k}'].append(ndcg.mean().item())
    model.train()
    avg_recalls = {k: sum(v) / len(v) if v else 0.0 for k, v in recalls.items()}
    avg_ndcgs   = {k: sum(v) / len(v) if v else 0.0 for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_splits_and_loaders(params):
    shared_kwargs = dict(
        rec_path=params['rec_path'],
        course_path=params['course_path'],
        course_id_map=params['course_id_map_path'],
        item_emb_h5_path=params.get('item_emb_h5_path'),
        user_emb_h5_path=params.get('user_emb_h5_path'),
    )
    train_dataset = EmbDataset(**shared_kwargs, mode='train')
    test_dataset  = EmbDataset(**shared_kwargs, mode='test')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=collate_emb_batch,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=params['infer_size'],
        shuffle=False,
        collate_fn=collate_emb_batch,
    )

    return train_dataset, train_dataloader, test_dataloader

def train(params):
    log_path = os.path.abspath(params['log_path'])
    save_path = os.path.abspath(params['save_path'])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Set up logging
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Configuration: {params}")
    # Initialize model
    model = TIGER(params)
    print(model.n_parameters)
    logging.info(model.n_parameters)

    # Set random seed for reproducibility
    set_seed(params['seed'])
    # Check if the device is available
    device = torch.device(params['device'] if torch.cuda.is_available() else 'cpu')

    train_dataset, train_dataloader, test_dataloader = build_splits_and_loaders(params)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    # Train the model
    model.to(device)
    best_recall = -1.0
    no_improve = 0
    patience = params.get('early_stop', 10)
    train_losses = []
    val_losses   = []
    
    for epoch in range(params['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{params['num_epochs']}")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, params, epoch=epoch+1)
        val_loss   = eval_loss(model, test_dataloader, device)
        logging.info(f"Training loss: {train_loss} | Val loss: {val_loss}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, train_dataset.item_embs, params, device)
        topk_list = params['topk_list']
        top_k = topk_list[-1]
        cur_recall = avg_recalls.get(f'Recall@{top_k}', 0)
        print(f"Epoch {epoch + 1}/{params['num_epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        for k in topk_list:
            print(f"  Recall@{k}: {avg_recalls.get(f'Recall@{k}', 0):.4f}  NDCG@{k}: {avg_ndcgs.get(f'NDCG@{k}', 0):.4f}")
        logging.info(f"Test: {avg_recalls} | {avg_ndcgs}")
        if cur_recall > best_recall:
            best_recall = cur_recall
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            logging.info(f"Best model saved (Recall@{top_k}={best_recall:.4f}) to {save_path}")
            print(f"  >> Best model saved (Recall@{top_k}={best_recall:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"早停：Epoch {epoch + 1}，Recall@{top_k} 连续 {patience} 次未提升")
                logging.info(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f"训练完成，最优模型已保存至 {save_path}")
    logging.info(f"Training complete. Best Recall@{top_k}={best_recall:.4f}")

    # 绘制并保存 loss 曲线
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses,   marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # 保存路径优先使用 params 中的配置，否则基于 log_path
    os.makedirs('loss_picture', exist_ok=True)
    plot_path = params['loss_plot_path']
    plt.savefig(plot_path, dpi=200)
    logging.info(f"Loss plot saved to {plot_path}")
    print(f"Loss plot saved to: {plot_path}")