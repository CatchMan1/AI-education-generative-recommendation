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

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
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

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_splits_and_loaders(params):
    full_dataset = EmbDataset(params['rec_path'], params['course_path'], params['course_id_map_path'])
    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)
    n_valid = int(n_total * 0.1)
    n_test = n_total - n_train - n_valid

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_valid, n_test],
        generator=torch.Generator().manual_seed(params['seed'])
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=collate_emb_batch,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=params['infer_size'],
        shuffle=False,
        collate_fn=collate_emb_batch,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=params['infer_size'],
        shuffle=False,
        collate_fn=collate_emb_batch,
    )

    return full_dataset, train_dataloader, validation_dataloader, test_dataloader

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

    _, train_dataloader, validation_dataloader, _ = build_splits_and_loaders(params)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    # Train the model
    model.to(device)
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(params['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{params['num_epochs']}")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        logging.info(f"Training loss: {train_loss}")
        train_losses.append(train_loss)
        val_loss = eval_loss(model, validation_dataloader, device)
        logging.info(f"Validation loss: {val_loss}")
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0  # Reset early stop counter
            torch.save(model.state_dict(), save_path)
            logging.info(f"Best model saved to {save_path}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in val_loss. Early stop counter: {early_stop_counter}")
            if early_stop_counter >= params['early_stop']:
                logging.info("Early stopping triggered.")
                break

    # 绘制并保存 loss 曲线
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Val Loss')
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