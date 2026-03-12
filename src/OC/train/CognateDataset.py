import torch
from torch.utils.data import Dataset
import os

class CognateDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileExistsError(f"Could not find file OC data path {data}")
            self.pairs = self.read_pairs(data)
        else:
            self.pairs = data

        if not self.validate(self.pairs):
            raise ValueError(f"Items must be (freq (int), src word (str), tgt word (str), NLD (float))")
        
    @staticmethod
    def validate(data):
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, tuple):
                return False
            if not len(item) == 4:
                return False
            freq, src, tgt, nld = item
            if not all([
                isinstance(freq, int),
                isinstance(src, str),
                isinstance(tgt, str),
                isinstance(nld, float)
            ]):
                return False
            return True
            
    def read_pairs(self, f):
        pairs = []
        with open(f) as inf:
            for line in inf.readlines():
                line = line.strip()
                freq, src_word, tgt_word, nld = line.split(" ||| ")
                src_word = src_word.strip()
                tgt_word = tgt_word.strip()
                pairs.append((src_word, tgt_word))
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        return self.pairs[index]

