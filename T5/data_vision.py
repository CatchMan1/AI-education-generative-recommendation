import pandas as pd
import torch
import torch.utils.data as data
import numpy as np
import h5py
import platform
import sys
from transformers import AutoModel, AutoTokenizer
class EmbDataset(data.Dataset):
    '''
    目前数据集暂无user_profile字段
    这里采用user_name代替user_profile作为emb.
    模型的输入数据是[user_profile_emb, item_seq_emb], [target_emb]，通过引入用户的profile增加个性化推荐的效果
    '''
    def __init__(self, rec_path, course_path, course_id_map):
        self.rec_path = rec_path
        self.course_path = course_path
        self.course_id_map = course_id_map
        self.rec_data, self.item_data, self.course_id_map = self.loading_data()
        self.item_info = self.mapping_id() # {num_id: item_info}
        self.tokenizer, self.model = self.load_plm()
        self.item_embs = self.generate_item_embedding(self.item_info, self.tokenizer, self.model)
        
        # 生成用户profile的embedding（基于姓名）
        self.user_profile_map = self.extract_user_profiles()
        self.user_profile_embs = self.generate_user_profile_embedding(self.user_profile_map, self.tokenizer, self.model)
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

    def load_plm(self, model_name='bert-base-uncased'):
        
        # 检查是否为Windows系统或者bitsandbytes不可用
        is_windows = platform.system() == "Windows"
        try:
            # 尝试导入和使用bitsandbytes（主要用于Linux/CUDA环境）
            if not is_windows:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                print("✅ 使用量化配置加载模型 (Linux/CUDA优化)")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config)
            else:
                raise ImportError("Windows环境，跳过量化配置")
        except (ImportError, Exception) as e:
            # 降级方案：不使用量化配置（适用于Windows或bitsandbytes不可用的情况）
            print(f"⚠️ 量化配置不可用 ({e.__class__.__name__})，使用标准配置")
            print("✅ 使用标准配置加载模型 (Windows兼容)")
            # 标准加载方式
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # 如果有GPU可用，将模型移到GPU
            if torch.cuda.is_available():
                print("检测到CUDA，将模型移至GPU")
                model = model.cuda()
            else:
                print("使用CPU运行模型")
        
        return tokenizer, model

    def generate_item_embedding(self, item_text_dic, tokenizer, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_item_id = max(item_text_dic.keys())
        order_texts = ["" if k == 0 else item_text_dic.get(k, "") for k in range(max_item_id + 1)]
        embeddings = []
        start, batch_size = 0, 20
        
        while start < len(order_texts):
            sentences = order_texts[start: start + batch_size]
            encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                        truncation=True, return_tensors='pt').to(device)
            outputs = model(**encoded_sentences)
            # 计算平均池化嵌入
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:,1:,:].sum(dim=1) / encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach()
            embeddings.append(mean_output)
            start += batch_size
        
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
        print('Item embeddings shape: ', embeddings.shape)
        return embeddings
    
    def generate_user_profile_embedding(self, user_profile_map, tokenizer, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        user_embs = {}
        sorted_users = sorted(user_profile_map.items(), key=lambda x: x[0])
        batch_size = 32
        
        for start_idx in range(0, len(sorted_users), batch_size):
            batch_users = sorted_users[start_idx:start_idx + batch_size]
            user_ids = [u[0] for u in batch_users]
            user_names = [u[1] for u in batch_users]
            
            encoded = tokenizer(user_names, padding=True, max_length=64,
                              truncation=True, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = model(**encoded)
                # 使用[CLS] token的embedding作为用户表示
                user_batch_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            for user_id, emb in zip(user_ids, user_batch_embs):
                user_embs[user_id] = emb
        
        print(f"User profile embeddings shape: ({len(user_embs)}, {list(user_embs.values())[0].shape[0]})")
        return user_embs
    
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
        user_emb = self.user_profile_embs[user_id]
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