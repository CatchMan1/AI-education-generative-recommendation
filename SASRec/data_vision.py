import h5py
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class SASRecDataset(Dataset):
    def __init__(self, h5_path, max_len=50, mode='train'):
        self.max_len = max_len
        self.mode = mode
        with h5py.File(h5_path, "r") as f:
            student_ids = f["interactions/student_ids"][:].astype(str)
            class_ids = f["interactions/class_ids"][:].astype(int)
            record_ids = f["interactions/record_ids"][:]

        # 构建物品映射
        unique_classes = sorted(list(set(class_ids)))
        self.class2idx = {cid: i + 1 for i, cid in enumerate(unique_classes)}  # 0为padding
        self.item_num = len(unique_classes)

        # 按用户构建时序序列 (S_u = (s_1, s_2, ..., s_{|S_u|}))
        user_hist = defaultdict(list)
        for s, c, r in zip(student_ids, class_ids, record_ids):
            user_hist[s].append((r, self.class2idx[c]))

        self.user_seqs = []
        for user_id, items in user_hist.items():
            # 按时间排序
            items = [x[1] for x in sorted(items, key=lambda x: x[0])]
            if len(items) < 3: continue  # 过滤过短序列

            if self.mode == 'train':
                # 训练集：取所有历史 (论文训练时用前t步预测t+1步)
                self.user_seqs.append(items[:-2])
            elif self.mode == 'test':
                # 测试集：完整序列 (论文测试时用前|S_u|-1步预测最后1步)
                self.user_seqs.append(items)

    def __len__(self):
        return len(self.user_seqs)

    def __getitem__(self, idx):
        seq = self.user_seqs[idx]  # 原始用户序列 S^u = [s_1, s_2, ..., s_{|S^u|}]
        n = self.max_len

        if self.mode == 'train':
            # 输入序列s = S^u[:-1]，截断/ padding到长度n
            input_seq = seq[:-1]  # S_1^u, S_2^u, ..., S_{|S^u|-1}^u
            # 构建输入序列s（固定长度n）
            if len(input_seq) >= n:
                s = input_seq[-n:]  # 截断：取最近n个
            else:
                s = [0] * (n - len(input_seq)) + input_seq  # padding：前面补0

            # 构建目标序列o_t
            o_t = []
            for t in range(n):
                s_t = s[t]
                # 1. s_t是padding项（0）→ o_t = <pad>（0）
                if s_t == 0:
                    o_t.append(0)
                # 2. 1 ≤ t < n → o_t = s_{t+1}
                elif 1 <= t < n:
                    if t + 1 < len(s):
                        o_t.append(s[t + 1])
                    else:
                        # t+1超出input_seq长度 → 取原序列最后一个物品S_{|S^u|}
                        o_t.append(seq[-1])
                # 3. t = n → o_t = S_{|S^u|}
                elif t == n:
                    o_t.append(seq[-1])

            return torch.tensor(s).long(), torch.tensor(o_t).long()

        elif self.mode == 'test':
            # 测试集：输入s = 完整序列[:-1]，目标为最后一个物品
            input_seq = seq[:-1]
            if len(input_seq) >= n:
                s = input_seq[-n:]
            else:
                s = [0] * (n - len(input_seq)) + input_seq
            target = seq[-1]
            return torch.tensor(s).long(), torch.tensor(target).long()