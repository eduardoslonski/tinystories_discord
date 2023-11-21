from torch.utils.data import Dataset
import torch
import numpy as np

class GPTDataset(Dataset):
    def __init__(self, path, context_length):
        self.data = np.fromfile(path, dtype='uint16')
        self.context_length = context_length
    
    def __len__(self):
        return len(self.data) // self.context_length

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start+self.context_length+1
        chunk = torch.as_tensor(self.data[start:end].astype("int32")).long()
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}