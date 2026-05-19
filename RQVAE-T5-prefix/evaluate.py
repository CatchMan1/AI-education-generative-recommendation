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
    
    # Load test dataset
    logging.info("加载测试数据集...")
    test_dataset = GenRecDataset(
        dataset_path=params['test_dataset_path'],
        max_len=params['max_len'],
        prof_h5_paths=params.get('prof_h5_paths', None),
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
    
    # 保存超参数和评估结果到CSV文件
    save_results_to_csv(params, avg_recalls, avg_ndcgs)
    
    return avg_recalls, avg_ndcgs


def save_results_to_csv(params, recalls, ndcgs):
    """
    将超参数和评估结果追加保存到result.csv文件
    
    Args:
        params: 参数字典
        recalls: Recall字典
        ndcgs: NDCG字典
    """
    # 固定文件名：result.csv
    csv_path = params['params_path']
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # 准备要保存的数据
    data = {}
    
    # 添加task_id
    data['task_id'] = params['task_id']
    
    # 添加超参数
    hyper_params = [
        'd_model', 'd_ff', 'd_kv', 'num_heads', 'num_layers', 
        'num_decoder_layers', 'dropout_rate', 'batch_size', 
        'num_epochs', 'lr', 'beam_size', 'vocab_size', 
        'codebook_size', 'max_len', 'bert_dim'
    ]
    
    for param_name in hyper_params:
        if param_name in params:
            data[param_name] = params[param_name]
    
    # 添加评估结果
    for k in params['topk_list']:
        recall_key = f'Recall@{k}'
        ndcg_key = f'NDCG@{k}'
        if recall_key in recalls:
            data[recall_key] = f"{recalls[recall_key]:.6f}"
        if ndcg_key in ndcgs:
            data[ndcg_key] = f"{ndcgs[ndcg_key]:.6f}"
    
    # 检查文件是否存在
    file_exists = os.path.exists(csv_path)
    
    # 写入CSV文件
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(data.keys())
        
        # 写入数据行
        writer.writerow(data.values())
    
    logging.info(f"超参数和评估结果已保存到 {csv_path}")
    print(f"超参数和评估结果已保存到 {csv_path}")