import torch
from transformers import T5Config, T5EncoderModel
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
import logging
from data_vision import EmbDataset

class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
        num_layers=config['num_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        d_kv=config['d_kv'],
        dropout_rate=config['dropout_rate'],
        vocab_size=config['vocab_size'],
        pad_token_id=config['pad_token_id'],
        eos_token_id=config['eos_token_id'],
        decoder_start_token_id=config['pad_token_id'],
        feed_forward_proj=config['feed_forward_proj'],
    )
        self.model = T5EncoderModel(t5config)
        self.input_proj = nn.Linear(config['input_emb_dim'], config['d_model'])
        self.output_proj = nn.Linear(config['d_model'], config['target_emb_dim'])
        self.loss_fn = nn.MSELoss()
    
    @property
    def n_parameters(self):
      num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
      total_params = num_params(self.parameters())
      emb_params = num_params(self.model.get_input_embeddings().parameters())
      return (
          f'#Embedding parameters: {emb_params}\n'
          f'#Non-embedding parameters: {total_params - emb_params}\n'
          f'#Total trainable parameters: {total_params}\n'
      )

    def forward(self, seq_embs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, target_emb: Optional[torch.Tensor] = None):

      inputs_embeds = self.input_proj(seq_embs)
      outputs = self.model(
          inputs_embeds=inputs_embeds,
          attention_mask=attention_mask,
      )
      hidden = outputs.last_hidden_state
      pooled = hidden[:, 0, :]
      pred_emb = self.output_proj(pooled)
      loss = None
      if target_emb is not None:
          loss = self.loss_fn(pred_emb, target_emb)
      return loss, pred_emb
    
    def generate(self, seq_embs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        _, pred_emb = self.forward(seq_embs=seq_embs, attention_mask=attention_mask, target_emb=None)
        return pred_emb


def collate_emb_batch(batch: List[Dict[str, Any]]):
    seq_lens = [b['seq_len'] for b in batch]
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

def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert (
        preds.shape[1] == maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {maxk}'

    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
      cur_label = labels[i].tolist()
      for j in range(maxk):
        cur_pred = preds[i, j].tolist()
        if cur_pred == cur_label:
          pos_index[i, j] = True
          break
    return pos_index

def recall_at_k(pos_index, k):
  return pos_index[:, :k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index, k):
  # Assume only one ground truth item per example
  ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
  dcg = 1.0 / torch.log2(ranks + 1)
  dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
  return dcg[:, :k].sum(dim=1).cpu().float()

def train(model, train_loader, optimizer, device):
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

def evaluate(model, eval_loader, topk_list, item_embs, device):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}

    item_embs_t = torch.tensor(item_embs, dtype=torch.float32, device=device)
    item_embs_t = F.normalize(item_embs_t, p=2, dim=1)
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            seq_embs = batch['seq_embs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_id = batch['target_id'].to(device)

            _, pred_emb = model(seq_embs=seq_embs, attention_mask=attention_mask, target_emb=None)
            # 余弦相似度召回
            pred_emb = F.normalize(pred_emb, p=2, dim=1)
            scores = pred_emb @ item_embs_t.T
            max_k = max(topk_list)
            topk_idx = torch.topk(scores, k=max_k, dim=1).indices

            for k in topk_list:
                cur_topk = topk_idx[:, :k]
                hit = (cur_topk == target_id.unsqueeze(1)).any(dim=1).float()
                recalls['Recall@' + str(k)].append(hit.mean().item())

                ranks = torch.arange(1, k + 1, device=device).unsqueeze(0)
                match = (cur_topk == target_id.unsqueeze(1)).float()
                denom = torch.log2(ranks + 1)
                ndcg = (match / denom).sum(dim=1)
                ndcgs['NDCG@' + str(k)].append(ndcg.mean().item())
    # Calculate average recalls and ndcgs
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIGER configuration")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--infer_size', type=int, default=96, help='Inference size for generating recommendations')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers in the model')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--d_ff', type=int, default=1024, help='Dimension of the feed-forward layer')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--d_kv', type=int, default=64, help='Dimension of key and value vectors')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=1025, help='Vocabulary size')
    parser.add_argument('--pad_token_id', type=int, default=0, help='Padding token ID')
    parser.add_argument('--eos_token_id', type=int, default=0, help='End of sequence token ID')
    parser.add_argument('--feed_forward_proj', type=str, default='relu', help='Feed forward projection type')
    parser.add_argument('--max_len', type=int, default=20, help='Maximum length for padding or truncation')
    parser.add_argument('--input_emb_dim', type=int, default=768, help='Input embedding dimension')
    parser.add_argument('--target_emb_dim', type=int, default=768, help='Target embedding dimension')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'], help='Mode of operation')
    parser.add_argument('--log_path', type=str, default='./logs/tiger.log', help='Path to the log file')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./ckpt/tiger.pth', help='Path to save the trained model')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--topk_list', type=list, default=[5,10,20], help='List of top-k values for evaluation metrics')
    parser.add_argument('--beam_size', type=int, default=30, help='Beam size for generation')
    config = vars(parser.parse_args())
    log_path = os.path.abspath(config['log_path'])
    save_path = os.path.abspath(config['save_path'])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Set up logging
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration: {config}")
    
    # Initialize model
    model = TIGER(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)

    # Set random seed for reproducibility
    set_seed(config['seed'])
    # Check if the device is available
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    full_dataset = EmbDataset()
    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)
    n_valid = int(n_total * 0.1)
    n_test = n_total - n_train - n_valid
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_valid, n_test],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_emb_batch)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False, collate_fn=collate_emb_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False, collate_fn=collate_emb_batch)
    
    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Validation dataset size: {len(validation_dataset)}")
    # print(f"Test dataset size: {len(test_dataset)}")
    # for batch in train_dataloader:
    #     print(f"Batch size: {len(batch['history'])}")
    #     print(f"the first batch history:{batch['history'][0]}")
    #     print(f"the first batch target:{batch['target'][0]}")
    #     print(f"the first batch attention mask:{batch['attention_mask'][0]}")
    #     break

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Train the model
    model.to(device)
    best_ndcg = 0.0
    early_stop_counter = 0
    
    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss = train(model, train_dataloader, optimizer, device)
        logging.info(f"Training loss: {train_loss}")
        # Evaluate the model
        avg_recalls, avg_ndcgs = evaluate(model, validation_dataloader, config['topk_list'], full_dataset.item_embs, device)
        logging.info(f"Validation Dataset: {avg_recalls}")
        logging.info(f"Validation Dataset: {avg_ndcgs}")
        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            early_stop_counter = 0  # Reset early stop counter
            test_avg_recalls, test_avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], full_dataset.item_embs, device)
            logging.info(f"Best NDCG@20: {best_ndcg}")
            logging.info(f"Test Dataset: {test_avg_recalls}")
            logging.info(f"Test Dataset: {test_avg_ndcgs}")
            # Save the best model
            torch.save(model.state_dict(), save_path)
            logging.info(f"Best model saved to {save_path}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}")
            if early_stop_counter >= config['early_stop']:
                logging.info("Early stopping triggered.")
                break
        
