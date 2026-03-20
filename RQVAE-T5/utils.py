import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def calculate_pos_index(preds, labels, maxk=20):
    """Calculate the position index of the ground truth items.

    Args:
      preds: The predicted token sequences, of shape
        (batch_size, maxk, seq_len).
      labels: The ground truth token sequences, of shape (batch_size, seq_len).

    Returns:
      A boolean tensor of shape (batch_size, maxk) indicating whether the
      prediction at each position is correct.
    """
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

def evaluate_model(model, eval_loader, topk_list, beam_size, device):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        eval_loader: 评估数据加载器
        topk_list: top-k值列表
        beam_size: beam search大小
        device: 设备
        
    Returns:
        avg_recalls, avg_ndcgs: 平均recall和NDCG指标
    """
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds[:, 1:]  # Exclude the start token
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)  # Reshape to (batch_size, beam_size, seq_len)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)
    
    # Calculate average recalls and ndcgs
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs

def plot_training_curves(train_losses, val_metrics=None, save_path=None, show_plot=True):
    """
    绘制训练曲线图
    
    Args:
        train_losses (list): 训练损失列表
        val_metrics (dict, optional): 验证指标字典，格式为 {metric_name: [values]}
        save_path (str, optional): 图片保存路径
        show_plot (bool): 是否显示图片，默认True
    """
    plt.figure(figsize=(12, 8))
    
    # 创建子图
    if val_metrics:
        # 如果有验证指标，创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training and Validation Curves', fontsize=16)
        
        # 训练损失
        axes[0, 0].plot(train_losses, 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 验证指标 - Recall@20
        if 'Recall@20' in val_metrics:
            axes[0, 1].plot(val_metrics['Recall@20'], 'g-', linewidth=2, label='Recall@20')
            axes[0, 1].set_title('Validation Recall@20')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Recall@20')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # 验证指标 - NDCG@20
        if 'NDCG@20' in val_metrics:
            axes[1, 0].plot(val_metrics['NDCG@20'], 'r-', linewidth=2, label='NDCG@20')
            axes[1, 0].set_title('Validation NDCG@20')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('NDCG@20')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # 隐藏最后一个子图
        axes[1, 1].axis('off')
        
    else:
        # 只有训练损失
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线图已保存到: {save_path}")
    
    # 显示图片
    if show_plot:
        plt.show()
    else:
        plt.close()