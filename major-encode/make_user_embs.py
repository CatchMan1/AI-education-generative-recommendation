import argparse
import os
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm

params = {
    "course_emb":"emb/course_item_embs.h5",
    "user_item_interact":"emb/user_item_interact.h5",
    "output_dir":'emb/user_item_emb.h5'
}


def save_user_item_emb(user_item_emb, output_path):
    user_ids = list(user_item_emb.keys())
    sample_uid = user_ids[0]
    sample_emb = user_item_emb[sample_uid]
    
    dim = sample_emb.shape[-1] if len(sample_emb.shape) > 1 else 768

    with h5py.File(output_path, 'w') as f:
        # 保存用户 ID
        f.create_dataset('user_id', data=np.array(user_ids, dtype='int32'))
        
        # 创建固定形状的 dataset: (用户数, 768)
        dset = f.create_dataset('user_embs', (len(user_ids), dim), dtype='float32')
        
        for i, uid in enumerate(user_ids):
            embs = np.array(user_item_emb[uid], dtype='float32')
            
            # 相加并求平均
            if embs.ndim > 1:
                mean_emb = np.mean(embs, axis=0)
            else:
                # 预防用户只有一个交互的情况
                mean_emb = embs
            
            # 写入 dataset
            dset[i] = mean_emb

    print(f"数据已成功聚合保存至 {output_path}，当前维度：({len(user_ids)}, {dim})")


def main(params):
    course_path = params["course_emb"]
    interact_path = params["user_item_interact"]
    # 读取课程emb
    with h5py.File(course_path, 'r') as f:
        embeddings = f['item_embs'][:] # [[emb1,.., ..], [emb2, ..., ...]]
    
    print("item_embs:", embeddings[:10])

    # 读取用户交互
    with h5py.File(interact_path, 'r') as f2:
        user_ids = f2['user_id'][:]          # 提取所有 user_id
        item_lists = f2['item_id_list'][:]    # 提取所有变长数组
        rec_data = list(zip(user_ids, item_lists))# [(np.int32(1), array([29, 63, 28, 52, 66], dtype=int32)), (np.int32(2), array([470, 140], dtype=int32)),...]
    
    print("item_embs:", embeddings[:10])
    user_item_emb = defaultdict(list)
    for (user_id, item_lists) in rec_data:
        if len(item_lists) >= 2:
            user_item_emb[user_id] = embeddings[item_lists[:-1]]
        else:
            user_item_emb[user_id] = embeddings[item_lists]    
    print("user_item_emb:", user_item_emb[1][:])
    output_dir = params["output_dir"]
    save_user_item_emb(user_item_emb, output_dir)


if __name__ == '__main__':

    main(params)
