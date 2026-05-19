import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import csv
import os
from model import TIGER
from data_vision import GenRecDataset, GenRecDataLoader
from utils import evaluate_model

def infer(params):
    """
    加载最优模型并在测试集上进行评估
    
    Args:
        params: 参数字典
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("开始加载最优模型进行测试集评估...")
    
    # Set device
    device = torch.device(params['device'])
    logging.info(f"使用设备: {device}")
    print(f"使用设备: {device}")
    
    # Load test dataset
    logging.info("加载测试数据集...")
    test_dataset = GenRecDataset(
        dataset_path=params['test_dataset_path'],
        max_len=params['max_len'],
    )
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=params['infer_size'], shuffle=False)
    logging.info(f"测试集大小: {len(test_dataset)}")
    
    # Initialize model
    logging.info("初始化模型...")
    model = TIGER(params)
    model.to(device)
    
    # Load best model weights
    logging.info(f"加载最优模型权重: {params['save_path']}")
    try:
        checkpoint = torch.load(params['save_path'], map_location=device)
        model.load_state_dict(checkpoint)
        logging.info("模型权重加载成功")
    except FileNotFoundError:
        logging.error(f"模型文件未找到: {params['save_path']}")
        return
    except Exception as e:
        logging.error(f"加载模型权重时出错: {e}")
        return
    
    # Evaluate on test set
    logging.info("开始在测试集上评估...")
    avg_recalls, avg_ndcgs = evaluate_model(
        model, test_dataloader, params['topk_list'], params['beam_size'], device
    )
    
    # Print results
    logging.info("=== 测试集评估结果 ===")
    print("\n=== 测试集评估结果 ===")
    for k in params['topk_list']:
        recall_key = f'Recall@{k}'
        ndcg_key = f'NDCG@{k}'
        if recall_key in avg_recalls and ndcg_key in avg_ndcgs:
            print(f"{recall_key}: {avg_recalls[recall_key]:.4f}")
            print(f"{ndcg_key}: {avg_ndcgs[ndcg_key]:.4f}")
            logging.info(f"{recall_key}: {avg_recalls[recall_key]:.4f}")
            logging.info(f"{ndcg_key}: {avg_ndcgs[ndcg_key]:.4f}")
    
    print("\n测试集评估完成！")
    logging.info("测试集评估完成！")
    
    save_results_to_csv(params, avg_recalls, avg_ndcgs)
    
    return avg_recalls, avg_ndcgs


def save_results_to_csv(params, recalls, ndcgs):
    csv_path = params['params_path']
    
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    data = {}
    
    # 添加task_id
    data['task_id'] = params['task_id']
    
    hyper_params = [
        'd_model', 'd_ff', 'd_kv', 'num_heads', 'num_layers', 
        'num_decoder_layers', 'dropout_rate', 'batch_size', 
        'num_epochs', 'lr', 'beam_size', 'vocab_size', 
        'codebook_size', 'max_len'
    ]
    
    for param_name in hyper_params:
        if param_name in params:
            data[param_name] = params[param_name]
    
    for k in params['topk_list']:
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