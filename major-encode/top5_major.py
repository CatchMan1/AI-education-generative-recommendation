import os

import h5py
import numpy as np
from tqdm import tqdm

params = {
    "user_emb_path": "emb/user_item_emb.h5",
    "level1_path": "emb/level1_embs.h5",
    "level2_path": "emb/level2_embs.h5",
    "level3_path": "emb/level3_embs.h5",
    "output_dir": "../data/professional",
    "top_k": 5,
}


def cosine_similarity_batch(X, Y):
    """
    批量计算余弦相似度
    X: (N, D)
    Y: (M, D)
    返回: (N, M)
    """
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    sim = X_norm @ Y_norm.T  # (N, M)
    return sim


def process_level(user_embs, level_embs, top_k=5):
    """
    计算每个用户与专业库的余弦相似度，取Top-K
    user_embs: (N, 768)
    level_embs: (M, 768)
    返回: user_major_embs (N, top_k, 768), top_indices (N, top_k)
    """
    N = user_embs.shape[0]
    sim_matrix = cosine_similarity_batch(user_embs, level_embs)  # (N, M)
    
    # 取每个用户Top-K相似度的索引
    top_indices = np.argpartition(-sim_matrix, top_k - 1, axis=1)[:, :top_k]  # (N, top_k)
    
    # 按相似度排序（从高到低）
    top_sims = np.take_along_axis(sim_matrix, top_indices, axis=1)
    sort_order = np.argsort(-top_sims, axis=1)
    top_indices = np.take_along_axis(top_indices, sort_order, axis=1)
    
    # 取出对应的向量
    user_major_embs = level_embs[top_indices]  # (N, top_k, 768)
    
    return user_major_embs, top_indices


def save_h5(output_path, user_ids, user_major_embs):
    """
    保存为H5文件
    user_ids: (N,)
    user_major_embs: (N, top_k, 768)
    """
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('user_id', data=user_ids.astype('int32'))
        f.create_dataset('user_major_embs', data=user_major_embs.astype('float32'))
    print(f"Saved: {output_path}, shape: {user_major_embs.shape}")


def main(params):
    os.makedirs(params["output_dir"], exist_ok=True)
    
    # 读取用户向量 (N, 768)
    with h5py.File(params["user_emb_path"], 'r') as f:
        user_ids = f['user_id'][:]
        user_embs = f['user_embs'][:]
    N = user_embs.shape[0]
    print(f"Users: {N}, user_embs shape: {user_embs.shape}")
    
    # 三个层级
    level_configs = [
        ("level1", params["level1_path"]),
        ("level2", params["level2_path"]),
        ("level3", params["level3_path"]),
    ]
    i = 1
    for level_name, level_path in level_configs:
        with h5py.File(level_path, 'r') as f:
            level_embs = f['embs'][:]
        print(f"{level_name}: {level_embs.shape}")
        
        user_major_embs, top_indices = process_level(user_embs, level_embs, top_k=params["top_k"])
        print(f"  user_major_embs shape: {user_major_embs.shape}")
        
        output_path = os.path.join(params["output_dir"], f"prof_lvl{i}.h5")
        save_h5(output_path, user_ids, user_major_embs)
        i += 1
    
    print("All done!")


if __name__ == '__main__':
    main(params)
