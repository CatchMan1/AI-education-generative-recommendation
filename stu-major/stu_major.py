"""
build_top5_embs.py
------------------
为每位学生在三个专业层级库中检索余弦相似度 Top-5 专业向量，
并分别保存为三个 H5 文件，供模型训练时直接索引。

输出 shape: (N, 5, 768)  —— N 为学生数，5 为 Top-K，768 为向量维度

依赖:
    pip install h5py scikit-learn pandas numpy
"""

import os
import json
import ast
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
INTERACTION_CSV   = "/Volumes/study/其他/AI_edu/stu-major/interaction_records.csv"
LEVEL_EMB_PATHS   = {
    1: "/Volumes/study/其他/AI_edu/stu-major/level1_embs.h5",
    2: "/Volumes/study/其他/AI_edu/stu-major/level2_embs.h5",
    3: "/Volumes/study/其他/AI_edu/stu-major/level3_embs.h5",
}
OUTPUT_PATHS      = {
    1: "level1_top5_student_embs.h5",
    2: "level2_top5_student_embs.h5",
    3: "level3_top5_student_embs.h5",
}
TOP_K             = 5
EMB_DIM           = 768


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def check_files():
    """检查所有依赖文件是否存在"""
    all_exist = True
    if not os.path.exists(INTERACTION_CSV):
        print(f"❌ 找不到交互文件：{INTERACTION_CSV}")
        all_exist = False
    for level, path in LEVEL_EMB_PATHS.items():
        if not os.path.exists(path):
            print(f"❌ 找不到层级{level}向量文件：{path}")
            all_exist = False
    if all_exist:
        print("✅ 所有依赖文件都存在")
    return all_exist


# ─────────────────────────────────────────────
# Step 1: 加载学生向量（多条记录取平均）
# ─────────────────────────────────────────────
def load_student_matrix(csv_path: str):
    """
    读取 interaction_records.csv，对每位学生的多条 bert_vector 取均值，
    返回:
        student_ids   : list[str]，长度 N
        student_matrix: np.ndarray，shape (N, 768)
    """
    df = pd.read_csv(csv_path).dropna(subset=["student_id"])
    
    # 自动查找bert向量列（不区分大小写和空格）
    bert_col = None
    for col in df.columns:
        clean_col = col.strip().lower()
        if "bert" in clean_col and "vector" in clean_col:
            bert_col = col
            break
    
    if bert_col is None:
        raise ValueError(f"在CSV中找不到BERT向量列！现有列名：{list(df.columns)}")
    
    print(f"✅ 自动找到BERT向量列：'{bert_col}'")
    
    student_ids = df["student_id"].unique().tolist()

    student_vecs = []
    for sid in student_ids:
        sub = df[df["student_id"] == sid]
        vecs = []
        for v_str in sub[bert_col].dropna():
            try:
                # 使用ast.literal_eval兼容单引号格式
                v = np.array(ast.literal_eval(v_str), dtype=np.float32)
                if v.shape == (EMB_DIM,):
                    vecs.append(v)
            except Exception as e:
                print(f"⚠️  跳过无效向量：学生ID={sid}, 错误={str(e)[:50]}")
                continue
        avg = np.mean(vecs, axis=0) if vecs else np.zeros(EMB_DIM, dtype=np.float32)
        student_vecs.append(avg)

    student_matrix = np.stack(student_vecs, axis=0)  # (N, 768)
    print(f"[load] 学生数 N={len(student_ids)}，向量矩阵 shape={student_matrix.shape}")
    return student_ids, student_matrix


# ─────────────────────────────────────────────
# Step 2: 单层级检索
# ─────────────────────────────────────────────
def retrieve_top_k(student_matrix: np.ndarray,
                   prof_embs: np.ndarray,
                   top_k: int = TOP_K):
    """
    计算余弦相似度并返回 Top-K 结果。

    参数:
        student_matrix : (N, 768)
        prof_embs      : (M, 768)
        top_k          : 取前几个

    返回:
        top_k_vecs     : (N, K, 768)  Top-K 专业向量
        top_k_indices  : (N, K)       在专业库中的索引
        top_k_sims     : (N, K)       对应余弦相似度
    """
    sim_matrix    = cosine_similarity(student_matrix, prof_embs)          # (N, M)
    top_k_indices = np.argsort(sim_matrix, axis=1)[:, -top_k:][:, ::-1]  # (N, K) 降序
    top_k_sims    = sim_matrix[np.arange(len(student_matrix))[:, None], top_k_indices]
    top_k_vecs    = prof_embs[top_k_indices]                              # (N, K, 768)
    return top_k_vecs, top_k_indices, top_k_sims


# ─────────────────────────────────────────────
# Step 3: 保存结果到 H5
# ─────────────────────────────────────────────
def save_h5(out_path: str,
            student_ids,
            top_k_vecs: np.ndarray,
            top_k_indices: np.ndarray,
            top_k_sims: np.ndarray,
            prof_names: np.ndarray,
            prof_codes: np.ndarray,
            level: int):
    """
    保存 Top-K 检索结果到 H5 文件。

    H5 datasets:
        student_top5_vecs  : (N, K, 768) float32  ← 模型训练直接用
        student_ids        : (N,)         bytes
        top5_indices       : (N, K)       int32
        top5_similarities  : (N, K)       float32
        top5_names         : (N, K)       bytes
        top5_codes         : (N, K)       bytes
    """
    N, K, _ = top_k_vecs.shape
    sid_enc      = np.array([str(s).encode("utf-8") for s in student_ids])
    top5_names   = np.array([[prof_names[j] for j in row] for row in top_k_indices])
    top5_codes   = np.array([[prof_codes[j] for j in row] for row in top_k_indices])

    with h5py.File(out_path, "w") as f:
        f.create_dataset("student_top5_vecs",  data=top_k_vecs,              dtype="float32")
        f.create_dataset("student_ids",         data=sid_enc)
        f.create_dataset("top5_indices",        data=top_k_indices.astype(np.int32))
        f.create_dataset("top5_similarities",   data=top_k_sims.astype(np.float32))
        f.create_dataset("top5_names",          data=top5_names)
        f.create_dataset("top5_codes",          data=top5_codes)
        f.attrs["description"] = f"Level{level} Top-{K} profession vectors per student"
        f.attrs["N"]           = N
        f.attrs["K"]           = K
        f.attrs["dim"]         = EMB_DIM

    print(f"  ✅ 已保存 → {out_path}  (shape={top_k_vecs.shape})")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    # 0. 检查文件
    if not check_files():
        exit(1)
    
    # 1. 学生向量
    student_ids, student_matrix = load_student_matrix(INTERACTION_CSV)

    # 2. 逐层级检索并保存
    for level, emb_path in LEVEL_EMB_PATHS.items():
        with h5py.File(emb_path, "r") as f:
            prof_embs  = np.array(f["embs"],  dtype=np.float32)   # (M, 768)
            prof_codes = np.array(f["codes"])
            prof_names = np.array(f["names"])

        M = prof_embs.shape[0]
        print(f"\n[Level{level}] 专业库大小 M={M}")

        top_k_vecs, top_k_indices, top_k_sims = retrieve_top_k(
            student_matrix, prof_embs, top_k=TOP_K
        )

        # 打印前 3 名学生的 Top-5 结果供核验
        for i in range(min(3, len(student_ids))):
            names = [
                prof_names[j].decode("utf-8") if isinstance(prof_names[j], bytes) else str(prof_names[j])
                for j in top_k_indices[i]
            ]
            print(f"  {student_ids[i]}: {names}")
            print(f"             sims={np.round(top_k_sims[i], 4)}")

        save_h5(
            out_path     = OUTPUT_PATHS[level],
            student_ids  = student_ids,
            top_k_vecs   = top_k_vecs,
            top_k_indices= top_k_indices,
            top_k_sims   = top_k_sims,
            prof_names   = prof_names,
            prof_codes   = prof_codes,
            level        = level,
        )

    print("\n=== 全部完成 ===")
    print("训练时读取方式示例:")
    print("  with h5py.File('level1_top5_student_embs.h5', 'r') as f:")
    print("      vecs = f['student_top5_vecs'][student_idx]  # shape (5, 768)")


if __name__ == "__main__":
    main()