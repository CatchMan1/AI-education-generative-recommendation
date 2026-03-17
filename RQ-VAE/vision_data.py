import pandas as pd
import torch
import torch.utils.data as data
import numpy as np
import h5py
import json


class EmbDataset(data.Dataset):

    def __init__(self, h5_path):

        self.h5_path = h5_path
        self.embeddings, self.meta = self._load_data()
        self.dim = self.embeddings.shape[-1]
        print(f"[RQ-VAE] Loaded {len(self.embeddings)} embeddings from {h5_path}, dim={self.dim}")
    def _load_data(self):
        with h5py.File(self.h5_path, 'r') as f:
            embeddings = f['item_embs'][:]
            meta_raw = f['meta'][()]
            meta = json.loads(meta_raw.decode('utf-8'))
        return embeddings, meta

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
