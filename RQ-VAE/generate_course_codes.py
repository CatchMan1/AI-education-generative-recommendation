import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE

import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

# Configuration
h5_path = "../data/course_item_embs.h5"
ckpt_path = "./ckpt/course/best_collision_model.pth"  # Update this path
output_file = "../data/course/course_rqvae_codes.npy"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
data = EmbDataset(h5_path)

# Default model parameters (update these based on your trained model)
model = RQVAE(in_dim=data.dim,
                  num_emb_list=[256,256,256,256],  # Update based on your model
                  e_dim=64,                        # Update based on your model
                  layers=[512,256,128],            # Update based on your model
                  dropout_prob=0.0,
                  bn=False,
                  loss_type="mse",
                  quant_loss_weight=1.0,
                  kmeans_init=False,
                  kmeans_iters=100,
                  sk_epsilons=[0,0,0.003,0.01],    # Update based on your model
                  sk_iters=100,
                  )

# Load checkpoint if available
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"Loaded state dict from {ckpt_path}")
else:
    print(f"Warning: No checkpoint found at {ckpt_path}, using randomly initialized model")

model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data, num_workers=4,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

print("Generating codes...")
for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))

        all_indices.append(code)
        all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

# Handle collisions if needed
for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon = 0.0

tt = 0
while True:
    if tt >= 30 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(f"Iteration {tt}: Found {len(collision_item_groups)} collision groups")
    
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)
        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1

print("All indices number: ", len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate", (tot_item-tot_indice)/tot_item)

# Convert to numeric codes
all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

codes = []
for key, value in all_indices_dict.items():
    code = [int(item.split('_')[1].strip('>')) for item in value]
    codes.append(code)

codes_array = np.array(codes)

# Add an extra dimension to all codes
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

# Resolve duplicates by incrementing the last dimension
unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = unique_codes[counts > 1]

if len(duplicates) > 0:
    print("Resolving duplicates in codes...")
    for duplicate in duplicates:
        duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
        for i, idx in enumerate(duplicate_indices):
            codes_array[idx, -1] = i

new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = new_unique_codes[new_counts > 1]

if len(duplicates) > 0:
    print("There still have duplicates:", duplicates)
else:
    print("There are no duplicates in the codes after resolution.")

# Save the codes
os.makedirs(os.path.dirname(output_file), exist_ok=True)
print(f"Saving codes to {output_file}")
print(f"the first 5 codes: {codes_array[:5]}")
np.save(output_file, codes_array)

# Also save the mapping from index to code for reference
mapping_file = output_file.replace('.npy', '_mapping.json')
index_to_code = {i: code.tolist() for i, code in enumerate(codes_array)}
with open(mapping_file, 'w') as f:
    json.dump(index_to_code, f, indent=2)
print(f"Saved index-to-code mapping to {mapping_file}")
