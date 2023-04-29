import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np

# Custom Dataset class
class CustomDataset(data.Dataset):
    def __init__(self, filepath, tokenizer, max_seq_length):
        self.data = pd.read_csv(filepath)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label, protected_attr = self.data.iloc[index]
        text = str(text)

        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens[:self.max_seq_length - 2] + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(input_mask, dtype=torch.long), torch.tensor(segment_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long), torch.tensor(protected_attr, dtype=torch.long)
