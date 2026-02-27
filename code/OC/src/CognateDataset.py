import torch
from torch.utils.data import Dataset

class CognateDataset(Dataset):
    def __init__(
        self,
        f_path
    ):
        super().__init__()
        self.path = f_path
        self.pairs = self.read_pairs(self.path)

    def read_pairs(self, f):
        pairs = []
        with open(f) as inf:
            for line in inf.readlines():
                line = line.strip()
                src_word, tgt_word = line.split(" ||| ")
                src_word = src_word.strip()
                tgt_word = tgt_word.strip()
                pairs.append((src_word, tgt_word))
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        return self.pairs[index]

