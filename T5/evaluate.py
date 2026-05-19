
import torch
from tqdm import tqdm
from model import TIGER
from train import build_splits_and_loaders, eval_loss
import logging
import os
import csv

def infer(params):
    log_path = os.path.abspath(params['log_path'])
    save_path = os.path.abspath(params['save_path'])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    device = torch.device(params['device'] if torch.cuda.is_available() else 'cpu')
    model = TIGER(params).to(device)

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Checkpoint not found: {save_path}")

    state = torch.load(save_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    full_dataset, _, _, test_dataloader = build_splits_and_loaders(params)

    test_loss = eval_loss(model, test_dataloader, device)
    logging.info(f"Test loss: {test_loss}")
    print(f"Test loss: {test_loss}")

    item_embs_t = torch.tensor(full_dataset.item_embs, dtype=torch.float32, device=device)
    item_embs_t = torch.nn.functional.normalize(item_embs_t, p=2, dim=1)

    topk_list = params.get('topk_list', [5, 10, 20])
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Infer"):
            seq_embs = batch['seq_embs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_id = batch['target_id'].to(device)

            _, pred_emb = model(seq_embs=seq_embs, attention_mask=attention_mask, target_emb=None)
            pred_emb = torch.nn.functional.normalize(pred_emb, p=2, dim=1)
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

    avg_recalls = {k: sum(v) / len(v) if len(v) > 0 else 0.0 for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) if len(v) > 0 else 0.0 for k, v in ndcgs.items()}

    logging.info(f"Test Recall: {avg_recalls}")
    logging.info(f"Test NDCG: {avg_ndcgs}")
    print(f"Test Recall: {avg_recalls}")
    print(f"Test NDCG: {avg_ndcgs}")
    
    save_results_to_csv(params, avg_recalls, avg_ndcgs)


def save_results_to_csv(params, recalls, ndcgs):
    csv_path = './result.csv'
    
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    data = {}
    
    # 添加task_id
    data['task_id'] = params['task_id']
    
    hyper_params = [
        'd_model', 'd_ff', 'd_kv', 'num_heads', 'num_layers', 
        'dropout_rate', 'batch_size', 'num_epochs', 'lr',
        'input_emb_dim', 'target_emb_dim', 'temperature'
    ]
    
    for param_name in hyper_params:
        if param_name in params:
            data[param_name] = params[param_name]
    
    topk_list = params.get('topk_list', [5, 10, 20])
    for k in topk_list:
        recall_key = f'Recall@{k}'
        ndcg_key = f'NDCG@{k}'
        if recall_key in recalls:
            data[recall_key] = f"{recalls[recall_key]:.6f}"
        if ndcg_key in ndcgs:
            data[ndcg_key] = f"{ndcgs[ndcg_key]:.6f}"
    
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(data.keys())
        
        writer.writerow(data.values())
    
    logging.info(f"超参数和评估结果已保存到 {csv_path}")
    print(f"超参数和评估结果已保存到 {csv_path}")