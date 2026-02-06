
import torch
from tqdm import tqdm
from model import TIGER
from train import build_splits_and_loaders, eval_loss
import logging
import os

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