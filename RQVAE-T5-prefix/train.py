
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

def train_one_epoch(model, train_loader, optimizer, device, epoch=None, num_epochs=None):
    model.train()
    total_loss = 0.0
    batch_count = 0
    desc = f"Epoch {epoch}/{num_epochs}" if epoch is not None else "Training"
    for batch in tqdm(train_loader, desc=desc, leave=True):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        prof_lvl1 = batch['prof_lvl1']
        prof_lvl2 = batch['prof_lvl2']
        prof_lvl3 = batch['prof_lvl3']
        
        prof_lvl1 = prof_lvl1.to(device)
        prof_lvl2 = prof_lvl2.to(device)
        prof_lvl3 = prof_lvl3.to(device)

        optimizer.zero_grad()
        loss, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            prof_lvl1=prof_lvl1,
            prof_lvl2=prof_lvl2,
            prof_lvl3=prof_lvl3,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def eval_loss(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['target'].to(device)
            prof_lvl1 = batch['prof_lvl1'].to(device)
            prof_lvl2 = batch['prof_lvl2'].to(device)
            prof_lvl3 = batch['prof_lvl3'].to(device)
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                prof_lvl1=prof_lvl1,
                prof_lvl2=prof_lvl2,
                prof_lvl3=prof_lvl3,
            )
            total_loss += loss.item()
    model.train()
    return total_loss / len(test_loader)


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
    device = torch.device(params['device'] if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    print(f"使用设备: {device}")
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 如果指定了cuda:1但只有一个GPU，则使用cuda:0
    if params['device'] == 'cuda:1' and torch.cuda.device_count() == 1:
        print("警告: 只有一个GPU可用，将使用cuda:0")
        device = torch.device('cuda:0')
    
    train_dataset = GenRecDataset(
        dataset_path=params['train_dataset_path'],
        code_path=params['code_path'],
        max_len=params['max_len'],
        codebook_size=params['codebook_size'],
        prof_h5_paths=params.get('prof_h5_paths', None),
    )
    test_dataset = GenRecDataset(
        dataset_path=params['test_dataset_path'],
        code_path=params['code_path'],
        max_len=params['max_len'],
        codebook_size=params['codebook_size'],
        prof_h5_paths=params.get('prof_h5_paths', None),
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=params['infer_size'], shuffle=False)
    
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
    best_val_loss = float('inf')
    no_improve = 0
    patience = params.get('early_stop', 10)
    train_losses = []
    val_losses   = []
    
    for epoch in range(params['num_epochs']):
        print(f"\n=== Epoch {epoch + 1}/{params['num_epochs']} ===")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch=epoch+1, num_epochs=params['num_epochs'])
        val_loss   = eval_loss(model, test_dataloader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Training loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        logging.info(f"Epoch {epoch + 1}/{params['num_epochs']}, Training loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), params['save_path'])
            logging.info(f"Best model saved (val_loss={best_val_loss:.4f}) to {params['save_path']}")
            print(f"  >> Best model saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"早停：Epoch {epoch + 1}，val_loss 连续 {patience} 次未降低")
                logging.info(f"Early stopping at epoch {epoch + 1}.")
                break

    print(f"训练完成，最优模型已保存至 {params['save_path']}")
    logging.info(f"Training complete. Best val_loss={best_val_loss:.4f}")

    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        val_metrics={},
        save_path=params.get('loss_plot_path', './training_curves.png'),
        show_plot=False
    )
    logging.info("训练曲线图已保存")
        
