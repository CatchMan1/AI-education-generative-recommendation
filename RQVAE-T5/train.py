
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
import logging
from data_vision import GenRecDataset, GenRecDataLoader
from utils import evaluate_model, plot_training_curves
from model import TIGER

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(params):
    # Set up logging
    logging.basicConfig(
        filename=params['log_path'],
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
    
    train_dataset = GenRecDataset(
        dataset_path=params['train_dataset_path'],
        code_path=params['code_path'],
        max_len=params['max_len'],
        codebook_size = params['codebook_size']
    )
    validation_dataset = GenRecDataset(
        dataset_path=params['val_dataset_path'],
        code_path=params['code_path'],
        max_len=params['max_len'],
        codebook_size = params['codebook_size']
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=params['infer_size'], shuffle=False)
    
    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Validation dataset size: {len(validation_dataset)}")
    # for batch in train_dataloader:
    #     print(f"Batch size: {len(batch['history'])}")
    #     print(f"the first batch history:{batch['history'][0]}")
    #     print(f"the first batch target:{batch['target'][0]}")
    #     print(f"the first batch attention mask:{batch['attention_mask'][0]}")
    #     break

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    # Train the model
    model.to(device)
    best_ndcg = 0.0
    early_stop_counter = 0
    
    # 记录训练损失和验证指标
    train_losses = []
    val_recalls = {f'Recall@{k}': [] for k in params['topk_list']}
    val_ndcgs = {f'NDCG@{k}': [] for k in params['topk_list']}
    
    for epoch in range(params['num_epochs']):
        print(f"\n=== Epoch {epoch + 1}/{params['num_epochs']} ===")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
        train_losses.append(train_loss)
        print(f"Training loss: {train_loss:.4f}")
        logging.info(f"Epoch {epoch + 1}/{params['num_epochs']}, Training loss: {train_loss:.4f}")
        
        # Evaluate the model
        avg_recalls, avg_ndcgs = evaluate_model(model, validation_dataloader, params['topk_list'], params['beam_size'], device)
        
        # 打印验证指标
        print("Validation Results:")
        for k in params['topk_list']:
            recall_key = f'Recall@{k}'
            ndcg_key = f'NDCG@{k}'
            if recall_key in avg_recalls and ndcg_key in avg_ndcgs:
                print(f"  {recall_key}: {avg_recalls[recall_key]:.4f}")
                print(f"  {ndcg_key}: {avg_ndcgs[ndcg_key]:.4f}")
        
        logging.info(f"Validation Dataset: {avg_recalls}")
        logging.info(f"Validation Dataset: {avg_ndcgs}")
        
        # 记录验证指标
        for k in params['topk_list']:
            recall_key = f'Recall@{k}'
            ndcg_key = f'NDCG@{k}'
            if recall_key in avg_recalls and ndcg_key in avg_ndcgs:
                val_recalls[recall_key].append(avg_recalls[recall_key])
                val_ndcgs[ndcg_key].append(avg_ndcgs[ndcg_key])
        
        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            early_stop_counter = 0  # Reset early stop counter
            print(f"🎉 New Best NDCG@20: {best_ndcg:.4f}")
            logging.info(f"Best NDCG@20: {best_ndcg}")
            # Save the best model
            torch.save(model.state_dict(), params['save_path'])
            logging.info(f"Best model saved to {params['save_path']}")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop counter: {early_stop_counter}/{params['early_stop']}")
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}")
            if early_stop_counter >= params['early_stop']:
                print("🛑 Early stopping triggered!")
                logging.info("Early stopping triggered.")
                break
    
    # 训练结束后绘制曲线
    logging.info("训练完成，正在绘制训练曲线...")
    
    # 合并验证指标
    val_metrics = {}
    val_metrics.update(val_recalls)
    val_metrics.update(val_ndcgs)
    
    # 绘制训练曲线
    plot_training_curves(
        train_losses=train_losses,
        val_metrics=val_metrics,
        save_path=params.get('loss_plot_path', './training_curves.png'),
        show_plot=False  # 设置为False避免在服务器环境下显示
    )
    
    logging.info("训练曲线图已保存")
        
