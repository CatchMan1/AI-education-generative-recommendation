import argparse
import json
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

def generate_user_profile_embedding(user_profile_map, tokenizer, model):
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

def load_course_id_map(course_id_map_path: str):
    with h5py.File(course_id_map_path, 'r') as f:
        course_id = f['item_id'].asstr(encoding='utf-8')[:]
        course_num_id = f['item_num_id'][:]
    return dict(zip(course_id, course_num_id))


def load_course_texts(course_path: str):
    with h5py.File(course_path, 'r') as f:
        item_ids = f['item_id'].asstr(encoding='utf-8')[:]
        item_name = f['item_name'].asstr(encoding='utf-8')[:]
        item_info = f['item_info'].asstr(encoding='utf-8')[:]
    return list(zip(item_ids, item_name, item_info))

def load_rec_data(rec_path: str):
    rec_data = []
    with h5py.File(rec_path, 'r') as f1:
        user_ids = f1['user_id'][:]          # 提取所有 user_id
        user_profile = f1['user_profile'].asstr(encoding='utf-8')[:] # 提取所有 user_profile
        item_lists = f1['item_id_list'][:]    # 提取所有变长数组
        rec_data = list(zip(user_ids, user_profile, item_lists)) 
    return rec_data

@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size: int, max_length: int):
    model.eval()
    embs = []
    for start in tqdm(range(0, len(texts), batch_size), desc='Encoding'):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        ).to(device)

        outputs = model(**encoded)
        masked_output = outputs.last_hidden_state * encoded['attention_mask'].unsqueeze(-1)
        mean_output = masked_output[:, 1:, :].sum(dim=1) / encoded['attention_mask'][:, 1:].sum(dim=-1, keepdim=True)
        embs.append(mean_output.detach().cpu())

    return torch.cat(embs, dim=0).numpy()


def encode(params):

    device = torch.device(params['device']) if params['device'] else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    course_id_map = load_course_id_map(params['course_id_map_path'])
    course_data = load_course_texts(params['course_path'])
    user_profile_data = load_rec_data(params['rec_path'])

    item_info = {}
    for item_id, item_name, item_info_text in course_data:
        if item_id in course_id_map:
            item_info[int(course_id_map[item_id])] = (item_name or '') + (item_info_text or '')

    max_item_id = max(item_info.keys())
    order_texts = ['' if k == 0 else item_info.get(k, '') for k in range(max_item_id + 1)]

    tokenizer = AutoTokenizer.from_pretrained(params['encode_model'])
    model = AutoModel.from_pretrained(params['encode_model']).to(device)

    embeddings = encode_texts(
        order_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=params['encode_batch_size'],
        max_length=params['encode_max_len'],
    ).astype(np.float32)

    user_profile_map = {user_id: user_name for user_id, user_name, _ in user_profile_data}
    user_embeddings = generate_user_profile_embedding(user_profile_map, tokenizer, model)

    # Save item embeddings to a separate HDF5 file
    item_emb_h5_path = params["item_emb_h5_path"]
    os.makedirs(os.path.dirname(os.path.abspath(item_emb_h5_path)), exist_ok=True)
    with h5py.File(item_emb_h5_path, 'w') as f:
        f.create_dataset('item_embs', data=embeddings, compression='gzip')
        meta = {
            'model_name': params['encode_model'],
            'max_length': params['encode_max_len'],
            'dim': int(embeddings.shape[1]),
        }
        f.create_dataset('meta', data=np.bytes_(json.dumps(meta, ensure_ascii=False)))

    # Save user embeddings to a separate HDF5 file
    user_emb_h5_path = params["user_emb_h5_path"]
    os.makedirs(os.path.dirname(os.path.abspath(user_emb_h5_path)), exist_ok=True)
    with h5py.File(user_emb_h5_path, 'w') as f:
        f.create_dataset('user_embs', data=np.array(list(user_embeddings.values()), dtype=np.float32), compression='gzip')

