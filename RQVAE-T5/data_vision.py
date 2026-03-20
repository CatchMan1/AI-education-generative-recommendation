import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
def process_data(file_path, max_len, PAD_TOKEN=0):

    processed_data = []
    with h5py.File(file_path, 'r') as f:
        # 1. 提取所有 history (这是个变长数组集合)
        # 2. 提取所有 target (这是个 ID 数组)
        histories = f['history'][:]
        targets = f['target'][:]

    # 将数据打包成你原始代码要求的字典格式
    for h, t in zip(histories, targets):
        processed_data.append({
            'history': h.tolist(), # 转回 list 格式
            'target': int(t)       # 确保 target 是标量整数
        })

    # Apply padding or truncation
    for item in processed_data:
        item['history'] = pad_or_truncate(item['history'], max_len, PAD_TOKEN)

    return processed_data

def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """
    Pad or truncate a sequence to a specified maximum length.

    Args:
        sequence (list): Input sequence.
        max_len (int): Maximum length for the sequence.

    Returns:
        list: Padded or truncated sequence.
    """
    if len(sequence) > max_len:
        # Truncate sequence
        return sequence[-max_len:]
    else:
        # Left pad sequence with PAD_TOKEN
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence
    
def item2code(code_path, codebook_size):
    """
    Convert itemID to code
    :param code_path: npy file path to store rqvae codes
    :return: dict item_to_code, code_to_item
    """
    data = np.load(code_path, allow_pickle=True)
    item_to_code = {}
    code_to_item = {}
    
    # for index, code in enumerate(data):
    #     item_to_code[index + 1] = code
    #     code_to_item[tuple(code)] = index + 1
    for index, code in enumerate(data):
        offsets = [c + i * codebook_size + 1 for i,c in enumerate(code)]
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1

    return item_to_code, code_to_item

class GenRecDataset(Dataset):
    def __init__(self, dataset_path, code_path, max_len, PAD_TOKEN=0, codebook_size=8):
        """
        Initialize the GenRecDataset.
        Args:
            dataset_path (str): Path to the dataset file.
            code_path (str): Path to the item-to-code mapping file.
            max_len (int): Maximum length for padding or truncation.
            PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.
        """
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.codebook_size = codebook_size
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        # Load item-to-code mapping
        self.item_to_code, self.code_to_item = item2code(code_path, codebook_size)
        # Process the dataset
        self.data = self._prepare_data()
        
    def _prepare_data(self):
        """
        Process the dataset and convert items to codes.
        Returns:
            list: Processed data with items converted to codes.
        """
        # Process the data using the process_data function
        processed_data = process_data(
            self.dataset_path, self.max_len, self.PAD_TOKEN
        )
        # Convert items to codes
        for item in processed_data:
            item['history'] = [self.item_to_code.get(x, np.array([self.PAD_TOKEN]*4)) for x in item['history']] # 码本数量
            item['target'] = self.item_to_code.get(item['target'], np.array([self.PAD_TOKEN]*4))
        return processed_data
    
    def __getitem__(self, index):
        """
        Get a single data item by index.
        Args:
            index (int): Index of the data item.
        Returns:
            dict: A dictionary containing 'history' and 'target'.
        """
        return self.data[index]
    
    def __len__(self):
        """
        Get the total number of data.
        Returns:
            int: Total number of data.
        """
        return len(self.data)

# Dataoding
class GenRecDataLoader(DataLoader):
    """
    GenRecDataLoader for Generative Recommendation tasks.
    
    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): Number of subprocesses to use for data loading.
        collate_fn (callable, optional): Function to merge a list of samples to form a mini-batch.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=None):
        collate_fn = self.collate_fn
        super(GenRecDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, collate_fn=collate_fn)
    
            
    def collate_fn(self, batch, pad_token=0):
        """
        crate attention mask for input sequence.
        
        Args:
            batch (list): List of samples from the dataset.
        
        Returns:
            dict: Batched data with padded sequences.
        """
        # Assuming each item in batch is a dictionary with 'history' and 'target'
        histories = [item['history'] for item in batch]
        targets = [item['target'] for item in batch]

        # Flatten histories and targets
        flattened_histories = torch.stack(
            [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in histories]
        )
        flattened_targets = torch.stack(
            [torch.tensor(target, dtype=torch.int64) for target in targets]
        )

        # Create attention masks for flattened histories
        attention_masks = torch.stack(
            [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
        )

        return {'history': flattened_histories, 'target': flattened_targets, 'attention_mask': attention_masks}
