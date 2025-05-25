import torch
from torch.utils.data import Dataset
import random

class BERT4RecDataset(Dataset):
    def __init__(self, user_sequences, max_len=50, mask_prob=0.15, pad_token=0, mask_token_id=999999, ignore_index=-100):
        self.sequences = user_sequences["item_sequence"].tolist()
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.pad_token = pad_token
        self.mask_token_id = mask_token_id
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][-self.max_len:]  # truncate to max_len
        tokens = []
        labels = []

        for item in seq:
            if random.random() < self.mask_prob:
                tokens.append(self.mask_token_id)  # replace with [MASK] token
                labels.append(item)                # remember true value
            else:
                tokens.append(item)
                labels.append(self.ignore_index)   # ignore in loss

        # Padding (left)
        padding_len = self.max_len - len(tokens)
        tokens = [self.pad_token] * padding_len + tokens
        labels = [self.ignore_index] * padding_len + labels

        return torch.tensor(tokens), torch.tensor(labels)
