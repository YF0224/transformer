import torch
from torch.utils.data import Dataset
from copy import deepcopy
import numpy as np

vocab_list = ["[BOS]", "[EOS]", "[PAD]", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
bos_token = "[BOS]"
eos_token = "[EOS]"
pad_token = "[PAD]"

def process_data(source, target):
    max_length = 12
    if len(source) > max_length:
        source = source[:max_length]
    if len(target) > max_length - 1:
        target = target[:max_length - 1]
    source_id = [vocab_list.index(p) for p in source]
    target_id = [vocab_list.index(p) for p in source]
    target_id = [vocab_list.index("[BOS]")] + target_id + [vocab_list.index("[EOS]")]
    source_m = np.array([1] * max_length)
    target_m = np.array([1] * (max_length + 1))
    if len(source_id) < max_length:
        pad_len = max_length - len(source_id)
        source_id += [vocab_list.index("[PAD]")] * pad_len
        source_m[-pad_len:] = 0
    if len(target_id) < max_length + 1:
        pad_len = max_length - len(target_id) + 1
        target_id += [vocab_list.index("[PAD]")] * pad_len
        target_m[-pad_len:] = 0
    return source_id, source_m, target_id, target_m

class MyDataset(Dataset):
    def __init__(self, source_path, target_path):
        super().__init__()
        self.source_list = []
        self.target_list = []
        with open(source_path) as f:
            content = f.readlines()
            for line in content:
                self.source_list.append(deepcopy(line.strip()))
        with open(target_path) as f:
            content = f.readlines()
            for line in content:
                self.target_list.append(deepcopy(line.strip()))

    def __len__(self):
        return len(self.source_list)

    def __getitem__(self, index):
        source_id, source_m, target_id, target_m =process_data(self.source_list[index], self.target_list[index])
        return (torch.tensor(source_id, dtype=torch.long), torch.tensor(source_m, dtype=torch.long), torch.tensor(target_id, dtype=torch.long), torch.tensor(target_m, dtype=torch.long))

if __name__ == "__main__":
    test_data = MyDataset("source.txt", "target.txt")
    a, b, c, d = test_data[2]

