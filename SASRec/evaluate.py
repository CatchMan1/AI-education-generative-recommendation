import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import csv
import os
from data_vision import SASRecDataset
from model import SASRec

def evaluate(params):
    device = torch.device(params['device'])
    dataset = SASRecDataset(params["data_path"], max_len=params["max_len"], mode='test', params=params)
    dataloader = DataLoader(dataset, batch_size=params["eval_batch_size"], shuffle=False)

    model = SASRec(dataset.item_num, params).to(device)
    model.load_state_dict(torch.load(params["ckpt"], map_location=device))
    model.eval()
    HT, NDCG = [], []

    with torch.no_grad():
        for input_ids, target_item in tqdm(dataloader, desc="评估中"):
            input_ids = input_ids.to(device)
            target_item = target_item.to(device)

            # 计算所有物品得分 r_{i,t} = h_t·M_i
            logits = model.predict(input_ids)
            logits[:, 0] = -1e9  # 屏蔽padding项

            # 计算目标物品的Rank
            target_scores = logits.gather(1, target_item.unsqueeze(1))  # 目标物品得分
            # 严格排序：ranks = 比目标得分高的物品数 + 1
            ranks = (logits > target_scores).sum(dim=1) + 1

            # 计算Hit@top_k和NDCG@top_k
            top_k = params.get('top_k', 10)
            for r in ranks.cpu().numpy():
                if r <= top_k:
                    HT.append(1)
                    NDCG.append(1 / np.log2(r + 1))  # NDCG公式
                else:
                    HT.append(0)
                    NDCG.append(0)

    # 打印评估结果
    top_k = params['top_k']
    hit_at_k = np.mean(HT)
    ndcg_at_k = np.mean(NDCG)
    print(f"\n>>> 评估:")
    print(f"Hit@{top_k} = {hit_at_k:.4f}, NDCG@{top_k} = {ndcg_at_k:.4f}")
    
    results = {f"Hit@{top_k}": hit_at_k, f"NDCG@{top_k}": ndcg_at_k}
    save_results_to_csv(params, results)
    
    return results


def save_results_to_csv(params, results):
    csv_path = params['params_path']
    
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    data = {}
    
    # 添加task_id
    data['task_id'] = params['task_id']
    
    hyper_params = [
        'd', 'num_blocks', 'num_heads', 'dropout', 'lr',
        'batch_size', 'epochs', 'mlp_layer', 'max_len', 'top_k'
    ]
    
    for param_name in hyper_params:
        if param_name in params:
            data[param_name] = params[param_name]
    
    for key, value in results.items():
        data[key] = f"{value:.6f}"
    
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(data.keys())
        
        writer.writerow(data.values())
    
    print(f"超参数和评估结果已保存到 {csv_path}")