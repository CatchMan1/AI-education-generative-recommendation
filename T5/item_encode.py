import argparse
import json
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


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

    os.makedirs(os.path.dirname(os.path.abspath(params["item_emb_h5_path"])), exist_ok=True)
    with h5py.File(params["item_emb_h5_path"], 'w') as f:
        f.create_dataset('item_embs', data=embeddings, compression='gzip')
        meta = {
            'model_name': params['encode_model'],
            'max_length': params['encode_max_len'],
            'dim': int(embeddings.shape[1]),
        }
        f.create_dataset('meta', data=np.string_(json.dumps(meta, ensure_ascii=False)))

