import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from data_vision import SASRecDataset
from model import SASRec

def evaluate(params):
    device = torch.device(params['device'])
    dataset = SASRecDataset(params["data_path"], max_len=params["max_len"], mode='test')
    dataloader = DataLoader(dataset, batch_size=params["eval_batch_size"], shuffle=False)

    model = SASRec(dataset.item_num, params).to(device)

    model.eval()
    HT, NDCG = [], []

    with torch.no_grad():
        for input_ids, target_item in tqdm(dataloader, desc="评估中"):
            input_ids = input_ids.to(device)
            target_item = target_item.to(device)

            # 计算所有物品得分 r_{i,t} = h_t·M_i
            logits = model.predict(input_ids)
            logits[:, 0] = -1e9  # 屏蔽padding项 (M_0=0向量)

            # 计算目标物品的Rank
            target_scores = logits.gather(1, target_item.unsqueeze(1))  # 目标物品得分
            # 严格排序：ranks = 比目标得分高的物品数 + 1
            ranks = (logits > target_scores).sum(dim=1) + 1

            # 计算Hit@10和NDCG@10
            for r in ranks.cpu().numpy():
                if r <= 10:
                    HT.append(1)
                    NDCG.append(1 / np.log2(r + 1))  # NDCG公式
                else:
                    HT.append(0)
                    NDCG.append(0)

    # 打印评估结果
    hit_at_10 = np.mean(HT)
    ndcg_at_10 = np.mean(NDCG)
    print(f"\n>>> 评估:")
    print(f"Hit@10 = {hit_at_10:.4f}, NDCG@10 = {ndcg_at_10:.4f}")
    return {"Hit@10": hit_at_10, "NDCG@10": ndcg_at_10}