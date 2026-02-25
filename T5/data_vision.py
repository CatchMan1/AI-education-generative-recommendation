import pandas as pd
import torch
import torch.utils.data as data
import numpy as np
import h5py
import platform
import sys
import os
from transformers import AutoModel, AutoTokenizer

class EmbDataset(data.Dataset):
    '''
    目前数据集暂无user_profile字段
    这里采用user_name代替user_profile作为emb.
    模型的输入数据是[user_profile_emb, item_seq_emb], [target_emb]，通过引入用户的profile增加个性化推荐的效果
    '''
    def __init__(self, rec_path, course_path, course_id_map, item_emb_h5_path, user_emb_h5_path):
        self.rec_path = rec_path
        self.course_path = course_path
        self.course_id_map = course_id_map
        self.item_emb_h5_path = item_emb_h5_path
        self.user_emb_h5_path = user_emb_h5_path
        self.rec_data, self.item_data, self.course_id_map = self.loading_data()
        self.item_info = self.mapping_id() # {num_id: item_info}
        # self.tokenizer, self.model = self.load_plm()
        self.item_embs = self.load_item_embeddings_from_h5(self.item_emb_h5_path)
        
        # 生成用户profile的embedding（基于姓名）
        self.user_profile_map = self.extract_user_profiles()
        self.user_profile_embs = self.load_user_embeddings_from_h5(self.user_emb_h5_path)
        # 构建序列推荐的训练样本
        self.samples, self.sample_user_ids = self.build_sequence_samples()

    def mapping_id(self):
        item_info = {}
        for item in self.item_data:
            for id, num_id in self.course_id_map.items():
                if id == item[0]:
                    item_info[num_id] = item[1] + item[2]
                    break
        return item_info

    def load_item_embeddings_from_h5(self, h5_path: str):
        with h5py.File(h5_path, 'r') as f:
            embs = f['item_embs'][:]
        return embs

    # def generate_item_embedding(self):
    #     embeddings = self.load_item_embeddings_from_h5(self.item_emb_h5_path)
    #     print('Item embeddings loaded from h5, shape: ', embeddings.shape)
    #     return embeddings
    
    def load_user_embeddings_from_h5(self, h5_path: str):
        with h5py.File(h5_path, 'r') as f:
            embs = f['user_embs'][:]
        return embs
    
    # def generate_user_embedding(self):
    #     embeddings = self.load_user_embeddings_from_h5(self.user_emb_h5_path)
    #     print('User embeddings loaded from h5, shape: ', embeddings.shape)
    #     return embeddings

    def loading_data(self):
        rec_data = []
        with h5py.File(self.rec_path, 'r') as f1:
            user_ids = f1['user_id'][:]          # 提取所有 user_id
            user_profile = f1['user_profile'].asstr(encoding='utf-8')[:] # 提取所有 user_profile
            item_lists = f1['item_id_list'][:]    # 提取所有变长数组
            rec_data = list(zip(user_ids, user_profile, item_lists))        

        item_data = []
        with h5py.File(self.course_path, 'r') as f2:
            item_ids = f2['item_id'].asstr(encoding='utf-8')[:]        # 提取所有 user_id
            item_name = f2['item_name'].asstr(encoding='utf-8')[:]
            item_info = f2['item_info'].asstr(encoding='utf-8')[:]  # 提取所有变长数组
            item_data = list(zip(item_ids, item_name, item_info))

        course_id_map = {}
        with h5py.File(self.course_id_map, 'r') as f3:
            course_id = f3['item_id'].asstr(encoding='utf-8')[:]
            course_num_id = f3['item_num_id'][:]
            course_id_map = dict(zip(course_id, course_num_id))
        
        return rec_data, item_data, course_id_map

    def build_sequence_samples(self, min_seq_len=2, max_seq_len=20):
        samples = []
        user_ids = []
        
        for user_id, user_name, item_list in self.rec_data:
            item_list = item_list.tolist() if isinstance(item_list, np.ndarray) else item_list
            
            # 过滤掉过短的序列
            if len(item_list) < min_seq_len:
                continue
            
            # 滑动窗口生成多个训练样本
            for i in range(1, len(item_list)):
                history = item_list[max(0, i-max_seq_len):i]

                target = item_list[i]
                
                samples.append((history, target))
                user_ids.append(user_id)  # 保存对应的user_id
        
        return samples, user_ids

    def get_sequence_embeddings(self, item_ids):
        embeddings = []
        for item_id in item_ids:
            embeddings.append(self.item_embs[item_id])
        return np.array(embeddings)

    def extract_user_profiles(self):
        user_profile_map = {}
        for user_id, user_name, item_list in self.rec_data:
            user_profile_map[user_id] = user_name
        return user_profile_map
    
    def __getitem__(self, index):
        history_ids, target_id = self.samples[index]
        user_id = self.sample_user_ids[index]
        # 获取用户profile embedding
        user_emb = self.user_profile_embs[user_id - 1]
        # 获取历史序列的embeddings
        seq_embs = self.get_sequence_embeddings(history_ids)
        
        # 将user_emb拼接到序列开头 [user_emb, item1, item2, ...]
        user_emb_expanded = np.expand_dims(user_emb, axis=0)  # (1, 768)
        full_seq_embs = np.concatenate([user_emb_expanded, seq_embs], axis=0)  # (seq_len+1, 768)
        # 获取目标item的embedding
        target_emb = self.item_embs[target_id]
        # 转换为tensor
        seq_embs = torch.FloatTensor(full_seq_embs)
        target_emb = torch.FloatTensor(target_emb)
        
        return {
            'seq_embs': seq_embs,   # (seq_len+1, 768) 第0位是user_emb，后面是item序列
            'target_emb': target_emb,    # (768,)
            'seq_len': len(history_ids) + 1,  # 包含user_emb的总长度
            'target_id': target_id,      # 目标item的ID（用于评估）
            'user_id': user_id           # 用户ID
        }

    def __len__(self):
        return len(self.samples)