import h5py
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class SASRecDataset(Dataset):
    def __init__(self, h5_path, max_len=50, mode='train', params=None):
        self.max_len = max_len
        self.mode = mode
        self.h5_path = h5_path
        self.params = params or {}
        self.rec_data = self._load_data()
        
        # 按用户构建时序序列 (S_u = (s_1, s_2, ..., s_{|S_u|}))
        user_hist = defaultdict(list)
        for user_id, user_name, item_sequence in self.rec_data:
            # item_sequence 已经是按时间排序的物品序列
            user_hist[user_id].extend(item_sequence)

        self.user_seqs = []
        for user_id, items in user_hist.items():
            if len(items) < self.params.get('min_seq_len', 3): continue  # 过滤过短序列

            if self.mode == 'train':
                # 训练集：去掉最后一个物品（测试目标），剩余做并行训练
                train_seq = items[:-1]  # 前|S_u|-1个物品用于训练
                if len(train_seq) >= 1:
                    self.user_seqs.append(train_seq)
            elif self.mode == 'test':
                # 测试集：留一法，预测最后一个物品
                self.user_seqs.append(items)
        
        # 计算物品总数（用于模型初始化）
        all_items = set()
        for items in user_hist.values():
            all_items.update(items)
        self.item_num = max(all_items) if all_items else 0  # 取最大物品ID作为总数

    def _load_data(self):
        with h5py.File(self.h5_path, 'r') as f:
            user_ids = f['user_id'][:]          # 提取所有 user_id
            user_profile = f['user_profile'].asstr(encoding='utf-8')[:] # 提取所有 user_profile
            item_lists = f['item_id_list'][:]    # 提取所有变长数组
            rec_data = list(zip(user_ids, user_profile, item_lists))
        return rec_data

    def __len__(self):
        return len(self.user_seqs)

    def __getitem__(self, idx):
        seq = self.user_seqs[idx]  # 原始用户序列 S^u = [s_1, s_2, ..., s_{|S^u|}]
        n = self.max_len

        if self.mode == 'train':
            # 假设 n 是 max_len
            # 1. 提取原始的输入和目标片段（保持相对偏移 1）
            # 如果原始序列 seq = [s1, s2, s3, s4]，n=10
            raw_input = seq[:-1]   # [s1, s2, s3]
            raw_target = seq[1:]   # [s2, s3, s4]

            # 2. 统一截断（只取最近的 n 个）
            raw_input = raw_input[-n:]
            raw_target = raw_target[-n:]

            # 3. 统一前补 0 (Pre-padding)
            pad_len = n - len(raw_input)
            
            s = [0] * pad_len + raw_input
            o_t = [0] * pad_len + raw_target

            return torch.tensor(s).long(), torch.tensor(o_t).long()

        elif self.mode == 'test':
            # 测试集：输入完整历史，预测最后一个物品
            if len(seq) < 2:
                return torch.zeros(n, dtype=torch.long), torch.tensor(0, dtype=torch.long)
            
            input_seq = seq[:-1]  # 除最后一个外的所有物品
            target = seq[-1]      # 最后一个物品作为预测目标
            
            if len(input_seq) >= n:
                s = input_seq[-n:]
            else:
                s = [0] * (n - len(input_seq)) + input_seq
            
            return torch.tensor(s).long(), torch.tensor(target).long()