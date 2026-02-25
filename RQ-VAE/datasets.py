import pandas as pd
import torch
import torch.utils.data as data
import numpy as np
import h5py
import json


class EmbDataset(data.Dataset):

    def __init__(self, h5_path):

        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as f:
            self.embeddings = f['item_embs'][:]
            meta_raw = f['meta'][()]
            self.meta = json.loads(meta_raw.decode('utf-8'))
        self.dim = self.embeddings.shape[-1]
        print(f"[RQ-VAE] Loaded {len(self.embeddings)} embeddings from {h5_path}, dim={self.dim}")

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
