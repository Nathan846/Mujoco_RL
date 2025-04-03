import torch
from torch.utils.data import Dataset
import json
import os
class ExpertTensorDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path)    
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.data[idx]
        return (
            torch.FloatTensor(state),
            torch.LongTensor([action]),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.FloatTensor([done])
        )