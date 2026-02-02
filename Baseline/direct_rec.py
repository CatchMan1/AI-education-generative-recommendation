from tokenize import String

import numpy as np
import csv
import torch
from sklearn.metrics.pairwise import cosine_similarity
from Rec import load_plm, generate_item_embedding
# from render import Render
import pickle
import json
import os
import random
from sentence_transformers import SentenceTransformer
import pandas as pd
import requests
from data_process import load_data_from_h5

# 全局数据缓存
_DATA_CACHE = None

def get_data():
    """获取数据，使用缓存机制"""
    global _DATA_CACHE
    if _DATA_CACHE is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        h5_path = os.path.join(script_dir, '..', 'data', 'recommendation_data.h5')
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"数据文件不存在: {h5_path}\n"
                f"请先运行 data_process.py 生成数据文件"
            )
        _DATA_CACHE = load_data_from_h5(h5_path)
    return _DATA_CACHE

def load_system_prompt(prompt_type="regular"):
    # 使用绝对路径，基于当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if prompt_type == "regular":
        file_path = os.path.join(script_dir, "prompts", "system_prompt_regular_user.txt")
    elif prompt_type == "cold_start":
        file_path = os.path.join(script_dir, "prompts", "system_prompt_cold_start.txt")
    else:
        raise ValueError(f"未知的提示词类型: {prompt_type}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    print(f"成功加载{prompt_type}系统提示词")
    return content
        
def find_history_interactions_from_db(userid):
    """从HDF5数据文件获取用户交互历史"""
    data = get_data()
    user_histories = data['interactions']['user_histories']
    
    print(f"=== 正在查找用户 {userid} 的交互记录 ===")
    
    if userid not in user_histories:
        print(f"未找到用户 {userid} 的交互记录")
        return []
    
    course_ids = user_histories[userid]
    print(f"找到 {len(course_ids)} 条交互记录")
    
    return course_ids
        
def find_user_profiles_from_db(userid):
    """从HDF5数据文件获取用户资料"""
    data = get_data()
    profiles = data['students']['profiles']
    
    print(f"=== 正在查找用户 {userid} 的资料信息 ===")
    
    if userid not in profiles:
        print(f"未找到用户 {userid} 的资料信息")
        return {'major': None, 'interest_long_profile': None}
    
    return profiles[userid]

def find_items_info_from_db():
    """
    从HDF5数据文件获取课程信息
    
    Returns:
        tuple: 包含以下元素的元组
            - item_pool (list): 所有课程ID的列表
            - item_names (dict): 课程ID到课程名称的映射
            - item_keywords_pos (dict): 课程ID到正面关键词集合的映射
            - item_keywords_neg (dict): 课程ID到负面关键词集合的映射  
            - item_content (dict): 课程ID到内容的映射
            - item_url (dict): 课程ID到URL的映射
    """
    data = get_data()
    classes = data['classes']
    
    print("=== 正在从数据文件读取课程信息 ===")
    
    item_pool = classes['item_pool']
    item_names = classes['item_names']
    item_keywords_pos = classes['item_keywords_pos']
    item_keywords_neg = classes['item_keywords_neg']
    item_content = classes['item_content']
    item_url = classes['item_url']
    
    return item_pool, item_names, item_keywords_pos, item_keywords_neg, item_content, item_url

def recommender(userid, topk=10):
    history = find_history_interactions_from_db(userid)
    
    profile = find_user_profiles_from_db(userid)
    
    item_pool, item_names, item_keywords_pos, item_keywords_neg, item_content, item_url = find_items_info_from_db()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plm_tokenizer, plm_model = load_plm('bert-base-uncased')
    plm_model = plm_model.to(device)


    item_text_dic = {
        item_id: " ".join(item_keywords_pos.get(item_id, set()) | item_keywords_neg.get(item_id, set()))
        for item_id in item_pool
    }

    item_text_dic[0] = ""  # 添加 padding 项目，用空字符串（作为嵌入的index=0的位置），保证课程编号从1开始正确匹配
    # 批量生成item_embeddings 维度为（81，768）的tensor, 内容包含了课程的积极和消极关键词文本的嵌入
    item_embeddings = generate_item_embedding(item_text_dic, plm_tokenizer, plm_model, word_drop_ratio=-1)
    item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32).to(device)

    if not item_pool:
        print("无法获取课程信息，推荐终止")
        return []
    
    print(f"\n=== 开始为用户 {userid} 生成推荐 ===")
    print(f"用户交互历史: {len(history) if history else 0} 条记录")
    print(f"用户资料: {'已获取' if profile['major'] or profile['interest_long_profile'] else '未获取'}")
    print(f"可用课程池: {len(item_pool)} 门课程")
    

    if history:    # 如果该用户有交互记录，即非冷启动用户
        print("执行基于历史记录的推荐...")

        top_k_names, top_k_urls = recommend_top_k(
            history, item_pool, item_names, item_keywords_pos, item_keywords_neg, 
            item_content, item_url, item_embeddings, profile, user_id = userid, k=topk
        )

        recommendations = [
            {"name": name, "url": url} for name, url in zip(top_k_names, top_k_urls)
        ]
        return recommendations
    
    else:
        print("执行冷启动推荐...")
        # 基于用户专业和兴趣进行冷启动推荐
        if profile['major'] or profile['interest_long_profile']:
            generated_content = generate_llm_recommendations_cold_start(
                profile['major'], profile['interest_long_profile']
            )
        
            candidate_items = [item for item in item_pool if item not in history]

            # 使用语义匹配将生成内容与课程匹配
            llm_similarities = match_generated_content_to_items(
                generated_content, candidate_items, item_names
            )
            
            llm_similarities.sort(key=lambda x: x[1], reverse=True)
            top_courses = llm_similarities[:topk]
            
            # 构建推荐结果
            recommendations = []
            for course_id, similarity in top_courses:
                course_name = item_names.get(course_id, f"课程{course_id}")
                course_url = item_url.get(course_id, "")
                recommendations.append({"name": course_name, "url": course_url})
            
            print(f"基于用户画像生成了 {len(recommendations)} 个冷启动推荐")
            return recommendations
                
        else:
            recommended_courses = item_pool[:topk]
            recommendations = [
                {
                    "name": item_names.get(course_id, f"课程{course_id}"), 
                    "url": item_url.get(course_id, "")
                } 
                for course_id in recommended_courses
            ]
            return recommendations

# 这里是计算候选项目与用户历史项目之间的相似度（关键词匹配，项目未嵌入，采用文本分词匹配）
def f_mat(history, candidate_item, item_keywords_pos, item_keywords_neg):
    # Keyword matching model
    pos_hist = [i for i, fb in history if fb == 1] # Ipos
    neg_hist = [i for i, fb in history if fb == 0] # Ineg
    Dpos_c = item_keywords_pos.get(candidate_item, set())
    Dneg_c = item_keywords_neg.get(candidate_item, set())
    alpha_pos = sum(len(Dpos_c & item_keywords_pos.get(i, set())) for i in pos_hist)
    alpha_neg = sum(len(Dneg_c & item_keywords_neg.get(i, set())) for i in neg_hist)
    return alpha_pos - alpha_neg  # 返回差值作为分数

# 这里是计算候选项目与用户历史项目之间的相似度（项目嵌入embedding）
def f_sim(history, candidate_item, item_embeddings):
    # Similarity model
    pos_hist = [i for i, fb in history if fb == 1]
    neg_hist = [i for i, fb in history if fb == 0]
    emb_c = item_embeddings[candidate_item].cpu().numpy().reshape(1, -1)
    beta_pos = 0.0
    if pos_hist:
        pos_embs = np.vstack([item_embeddings[i].cpu().numpy() for i in pos_hist])
        beta_pos = float(np.max(cosine_similarity(emb_c, pos_embs)))
    beta_neg = 0.0
    if neg_hist:
        neg_embs = np.vstack([item_embeddings[i].cpu().numpy() for i in neg_hist])
        beta_neg = float(np.max(cosine_similarity(emb_c, neg_embs)))
    return beta_pos - beta_neg  # 返回差值作为分数

def generate_llm_recommendations_cold_start(major, interest_long_profile):
    """为冷启动用户基于专业和兴趣标签生成推荐内容"""
    
    system_prompt = load_system_prompt("cold_start")

    user_prompt = f"""## 新用户信息

### 用户专业背景
专业：{major if major else '未提供专业信息，请基于通用技术发展趋势推荐'}

### 用户兴趣标签
兴趣标签：{interest_long_profile if interest_long_profile else '未提供兴趣信息，请基于专业发展需求推荐'}

请生成推荐内容："""
    # 调用 Qwen 模型 API
    response = call_qwen_api(user_prompt, system_prompt)
    print(f"为专业[{major}]和兴趣[{interest_long_profile}]的用户生成了冷启动推荐")
    return response

def generate_llm_recommendations(user_history, item_content, user_profiles, user_id, k):
    """使用Qwen模型生成推荐内容"""
    # 构建用户历史描述
    pos_items = [item_content[i] for i in user_history]   # 提取项目的全部文本内容
 
    user_profile = user_profiles.get(user_id, "")

    # 格式化喜欢的学习资源内容
    pos_items_formatted = ""
    if pos_items:
        pos_items_formatted = "\n".join([f"  - {item}" for item in pos_items])
    else:
        pos_items_formatted = "  - 无相关历史记录"

    # 从外部文件加载系统提示词
    system_prompt = load_system_prompt("regular").format(k=k)

    # 构建用户提示词（仅包含具体的用户数据）
    user_prompt = f"""## 学生信息
    ### 用户画像
    {user_profile if user_profile else '暂无用户画像信息，请基于交互历史进行推断'}
    ### 历史学习偏好分析
    **该学生喜欢的学习资源内容：**
    {pos_items_formatted}
    请根据以上信息为该学生推荐合适的学习资源。"""

    # 调用 Qwen 模型 API，传递系统提示和用户提示
    response = call_qwen_api(user_prompt, system_prompt)

    return response

def call_qwen_api(user_prompt, system_prompt=None):
    """调用Qwen模型API（支持wcode.net和阿里云官方接口）"""
    # 从环境变量获取API密钥
    api_key = os.environ.get("QWEN_API_KEY", "sk-1039.K9YAazwwEGWtnG5vZy06KgD5kwkEkutAHO1NEL5TOEaThlkn")
    
    # 检测API密钥类型并设置对应的API端点
    if "wcode.net" in api_key or api_key.startswith("sk-1039"):
        # wcode.net API 配置 - 使用正确的端点
        api_url = "https://wcode.net/api/gpt/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # wcode.net 使用 OpenAI 格式的消息格式
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user", 
            "content": user_prompt
        })
        
        data = {
            "model": "qwen2.5-14b-instruct-1m",  # 使用正确的模型名称
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
    else:
        # 阿里云官方 API 配置
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 阿里云格式的消息格式
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user", 
            "content": user_prompt
        })
        
        data = {
            "model": "qwen-plus",
            "input": {
                "messages": messages
            },
            "parameters": {
                "result_format": "text",
                "max_tokens": 2000,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
    response = requests.post(api_url, headers=headers, json=data, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        
        # 处理不同API的响应格式
        if "wcode.net" in api_key or api_key.startswith("sk-1039"):
            # wcode.net 响应格式 (OpenAI格式)
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print("Qwen 模型调用成功 (wcode.net)")
                return content
            else:
                print(f"wcode.net API 响应格式异常: {result}")
                return "API响应格式异常"
        else:
            # 阿里云官方响应格式
            if "output" in result and "text" in result["output"]:
                print("Qwen 模型调用成功 (阿里云官方)")
                return result["output"]["text"]
            else:
                print(f"阿里云 API 响应格式异常: {result}")
                return "API响应格式异常"
                
    elif response.status_code == 401:
        print(f"API密钥无效，请检查密钥配置")
        print(f"当前使用的API密钥前缀: {api_key[:20]}...")
        return "API密钥无效，请检查环境变量配置"
    else:
        print(f"API 请求失败: {response.status_code}, {response.text}")
        return f"API请求失败: {response.status_code}"
            
def match_generated_content_to_items(generated_content, candidate_items, item_content):
    """将生成的内容与候选项目匹配"""
    # 初始化语义编码器
    semantic_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 计算生成内容的语义嵌入
    generated_embedding = semantic_encoder.encode([generated_content])
    
    # 构建候选项目的文本描述
    item_texts = {}
    for item_id in candidate_items:
        item_text = str(item_content.get(item_id, ""))
        item_texts[item_id] = item_text
    
    # 计算候选项目文本的语义嵌入
    item_text_list = list(item_texts.values())
    item_ids_list = list(item_texts.keys())
    
    if not item_text_list:
        return []
    
    item_semantic_embeddings = semantic_encoder.encode(item_text_list)
    
    # 计算相似度
    similarities = cosine_similarity(generated_embedding, item_semantic_embeddings)[0]
    
    # 返回相似度分数
    item_similarities = list(zip(item_ids_list, similarities))
    return item_similarities

def get_user_history_labels(user_history, candidate_items):
    positive_samples = [(item, 1) for item in user_history]
    
    num_positive = len(user_history)
    
    if len(candidate_items) < num_positive:
        negative_items = candidate_items
    else:
        negative_items = random.sample(candidate_items, num_positive)
    
    # 生成负样本：标记为 (item, 0)
    negative_samples = [(item, 0) for item in negative_items]
    
    # 3. 合并正负样本
    all_samples = positive_samples + negative_samples
    return all_samples

def recommend_top_k(user_history, item_pool, item_names, item_keywords_pos, item_keywords_neg, item_content, item_url, item_embeddings, user_profiles, user_id, k, alpha=0.1, beta=0.2, gamma=0.7):

    # 提取用户已交互过的项目ID
    interacted_items = user_history
    
    # 过滤掉已交互的项目
    candidate_items = [item for item in item_pool if item not in interacted_items]
    
    if len(candidate_items) == 0:
        print("警告：没有可推荐的新项目")
        return {}, {}
    
    user_history_labels = get_user_history_labels(user_history, candidate_items)


    # 1. 原有的关键词匹配和嵌入相似度计算
    mat_scores = []
    sim_scores = []
    
    for item in candidate_items:
        mat_score = f_mat(user_history_labels, item, item_keywords_pos, item_keywords_neg)
        sim_score = f_sim(user_history_labels, item, item_embeddings)
        mat_scores.append(mat_score)
        sim_scores.append(sim_score)
    
    # 2. Qwen模型生成推荐内容并匹配
    print(f"为用户 {user_id} 生成Qwen推荐...")
    generated_content = generate_llm_recommendations(user_history, item_names, user_profiles, user_id, k)
    
    # 获取所有候选项目与生成内容的相似度
    llm_similarities = match_generated_content_to_items(generated_content, candidate_items, item_names)
    llm_score_dict = dict(llm_similarities)
    llm_scores = [llm_score_dict.get(item, 0.0) for item in candidate_items]

    # 3. 归一化所有分数
    def normalize_scores(scores):
        if not scores:
            return scores
        min_score, max_score = min(scores), max(scores)
        if max_score > min_score:
            return [(s - min_score) / (max_score - min_score) for s in scores]
        else:
            return [0.0] * len(scores)
    
    mat_scores_norm = normalize_scores(mat_scores)
    sim_scores_norm = normalize_scores(sim_scores)
    llm_scores_norm = normalize_scores(llm_scores)
    
    total_scores = [
        alpha * m + beta * s + gamma * l 
        for m, s, l in zip(mat_scores_norm, sim_scores_norm, llm_scores_norm)
    ]
    
    # 排序并返回前k个
    scores = list(zip(candidate_items, total_scores))
    top_k_score = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    top_k = [item for item, score in top_k_score]
    top_k_names = [item_names.get(item, "未知课程") for item in top_k]
    top_k_urls = [item_url.get(item, "") for item in top_k]

    return top_k_names, top_k_urls
