import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
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
    device = torch.device(params['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # Load test dataset
    logging.info("加载测试数据集...")
    test_dataset = GenRecDataset(
        dataset_path=params['test_dataset_path'],
        code_path=params['code_path'],
        max_len=params['max_len'],
        codebook_size = params['codebook_size']
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
    
    return avg_recalls, avg_ndcgs