import numpy as np
import csv
import torch
from sklearn.metrics.pairwise import cosine_similarity
from Rec import load_plm, generate_item_embedding
import pickle
import json
import os
import random
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from data_process import load_data_from_h5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 导入 direct_rec 中的核心函数
from direct_rec import (
    find_history_interactions_from_db,
    find_user_profiles_from_db,
    find_items_info_from_db,
    f_mat,
    f_sim,
    get_user_history_labels,
    generate_llm_recommendations,
    match_generated_content_to_items
)
def get_all_users_from_db():
    """从HDF5数据文件获取所有有交互记录的用户ID"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h5_path = os.path.join(script_dir, '..', 'data', 'recommendation_data.h5')
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"数据文件不存在: {h5_path}\n"
            f"请先运行 data_process.py 生成数据文件"
        )
    
    data = load_data_from_h5(h5_path)
    user_ids = data['students']['user_ids']
    
    print(f"从数据文件获取到 {len(user_ids)} 个用户")
    return user_ids


def leave_one_out_split(user_history):
    if len(user_history) < 2:
        return user_history, None
    
    # 留一法
    train_history = user_history[:-1]
    test_item = user_history[-1]
    
    return train_history, test_item


def recommend_top_k_for_evaluation(user_history, item_pool, item_names, item_keywords_pos, 
                                   item_keywords_neg, item_content, item_embeddings, 
                                   user_profiles, user_id, params):
    # 提取用户已交互过的项目ID
    interacted_items = user_history
    # 过滤掉已交互的项目
    candidate_items = [item for item in item_pool if item not in interacted_items]
    if len(candidate_items) == 0:
        return [], []
    # 构建用户历史标签（正负样本）
    user_history_labels = get_user_history_labels(user_history, candidate_items)
    
    # 关键词匹配分数
    mat_scores = []
    for item in candidate_items:
        mat_score = f_mat(user_history_labels, item, item_keywords_pos, item_keywords_neg)
        mat_scores.append(mat_score)
    
    # 嵌入相似度分数
    sim_scores = []
    for item in candidate_items:
        sim_score = f_sim(user_history_labels, item, item_embeddings)
        sim_scores.append(sim_score)
    
    # LLM生成推荐分数
    if params['use_llm']:
        print(f"为用户 {user_id} 生成LLM推荐...")
        generated_content = generate_llm_recommendations(user_history, item_names, user_profiles, user_id, params['k'])
        llm_similarities = match_generated_content_to_items(generated_content, candidate_items, item_names)
        llm_score_dict = dict(llm_similarities)
        llm_scores = [llm_score_dict.get(item, 0.0) for item in candidate_items]
    else:
        llm_scores = [0.0] * len(candidate_items)
    
    # 归一化所有分数
    def normalize_scores(scores):
        if not scores or max(scores) == min(scores):
            return [0.0] * len(scores)
        min_score, max_score = min(scores), max(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    mat_scores_norm = normalize_scores(mat_scores)
    sim_scores_norm = normalize_scores(sim_scores)
    llm_scores_norm = normalize_scores(llm_scores) if params['use_llm'] else [0.0] * len(candidate_items)
    
    # 加权融合
    if params['use_llm']:
        total_scores = [
            params['alpha'] * m + params['beta'] * s + params['gamma'] * l 
            for m, s, l in zip(mat_scores_norm, sim_scores_norm, llm_scores_norm)
        ]
    else:
        # 不使用LLM时，重新分配权重
        alpha_new = 0.5
        beta_new = 0.5
        total_scores = [
            alpha_new * m + beta_new * s 
            for m, s in zip(mat_scores_norm, sim_scores_norm)
        ]
    
    # 6. 排序并返回前k个
    scores_with_items = list(zip(candidate_items, total_scores))
    scores_with_items.sort(key=lambda x: x[1], reverse=True)
    top_k_items = [item for item, score in scores_with_items[:params['k']]]
    top_k_scores = [score for item, score in scores_with_items[:params['k']]]
    
    return top_k_items, top_k_scores


def calculate_metrics(recommended_items, test_item, k=10):
    metrics = {}
    
    # Precision@K
    if test_item in recommended_items[:k]:
        metrics['precision'] = 1.0 / k
        metrics['recall'] = 1.0
        metrics['hit'] = 1.0
        
        # NDCG@K
        position = recommended_items.index(test_item) + 1
        metrics['ndcg'] = 1.0 / np.log2(position + 1)
    else:
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['hit'] = 0.0
        metrics['ndcg'] = 0.0
    
    return metrics


def evaluate_leave_one_out(params):
    set_seed(params['seed'])
    # 获取所有用户
    user_ids = get_all_users_from_db()
    user_ids = user_ids[:params['max_users']]
    print(f"限制评估用户数量: {params['max_users']}")
    
    # 获取课程信息
    item_pool, item_names, item_keywords_pos, item_keywords_neg, item_content, item_url = find_items_info_from_db()
    
    # 加载PLM模型和生成item embeddings
    print("\n加载预训练模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plm_tokenizer, plm_model = load_plm('bert-base-uncased')
    plm_model = plm_model.to(device)
    plm_model.eval()
    
    item_text_dic = {
        item_id: " ".join(item_keywords_pos.get(item_id, set()) | item_keywords_neg.get(item_id, set()))
        for item_id in item_pool
    }
    item_text_dic[0] = ""
    
    print("生成课程嵌入向量...")
    with torch.no_grad():
        item_embeddings = generate_item_embedding(item_text_dic, plm_tokenizer, plm_model, word_drop_ratio=-1)
    item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32).to(device)
    
    # 获取用户画像
    user_profiles = {}
    for user_id in user_ids:
        profile = find_user_profiles_from_db(user_id)
        user_profile_text = f"专业: {profile['major']}, 兴趣: {profile['interest_long_profile']}"
        user_profiles[user_id] = user_profile_text
    
    # 对每个用户进行留一法评估
    all_metrics = defaultdict(list)
    valid_users = 0
    
    print(f"\n开始评估 {len(user_ids)} 个用户...")
    print("-" * 60)
    
    for idx, user_id in enumerate(user_ids, 1):
        # 获取用户历史交互
        user_history = find_history_interactions_from_db(user_id)
        
        # 至少需要2个交互才能进行留一法评估
        if len(user_history) < 2:
            print(f"[{idx}/{len(user_ids)}] 用户 {user_id}: 交互记录不足，跳过")
            continue
        
        # 留一法划分
        train_history, test_item = leave_one_out_split(user_history)
        
        # 生成推荐
        recommended_items, scores = recommend_top_k_for_evaluation(
            train_history, item_pool, item_names, item_keywords_pos, 
            item_keywords_neg, item_content, item_embeddings, 
            user_profiles, user_id, params
        )
        
        if not recommended_items:
            print(f"[{idx}/{len(user_ids)}] 用户 {user_id}: 无推荐结果，跳过")
            continue
        
        # 计算指标
        metrics = calculate_metrics(recommended_items, test_item, params['k'])
        
        for metric_name, value in metrics.items():
            all_metrics[metric_name].append(value)
        
        valid_users += 1
            
    # 计算平均指标
    print("评估结果汇总")
    results = {}
    for metric_name, values in all_metrics.items():
        avg_value = np.mean(values)
        results[metric_name] = avg_value
        print(f"{metric_name.upper()}@{params['k']}: {avg_value:.4f}")
    
    print("="*60)
    
    return results, all_metrics

if __name__ == "__main__":
    print("\n推荐系统留一法评估")
    print("基于 direct_rec.py 的混合推荐模型")
    params = {
        'k' : 10,
        'alpha' : 0.1,
        'beta' : 0.3,
        'gamma' : 0.6,
        'use_llm' : True,
        'max_users' : 14,
        'seed' : 42
    }
    results_fast, metrics_fast = evaluate_leave_one_out(params)
    
