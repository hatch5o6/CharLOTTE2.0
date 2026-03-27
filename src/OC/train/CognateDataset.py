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
            raise ValueError(f"Data must be a list of (src_word, tgt_word) tuples!")
        
    # @staticmethod
    # def validate(data):
    #     if not isinstance(data, list):
    #         return False
    #     for item in data:
    #         if not isinstance(item, tuple):
    #             return False
            
    #         if len(item) < 4:
    #             return False
            
    #         freqs = item[:-3]
    #         src, tgt, nld = item[-3:]

    #         for freq in freqs:
    #             if not (isinstance(freq, int) or isinstance(freq, float)):
    #                 return False
    #         if not all([
    #             isinstance(src, str),
    #             isinstance(tgt, str),
    #             isinstance(nld, float)
    #         ]):
    #             return False
    #     return True
    
    @staticmethod
    def validate(data):
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, tuple):
                return False
            if not len(item) == 2:
                return False
            for word in item:
                if not isinstance(word, str):
                    return False
        return True
            
    def read_pairs(self, f):
        pairs = []
        with open(f) as inf:
            for line in inf.readlines():
                line = line.strip()
                split_line = line.split(" ||| ")
                # if len(split_line) == 4:
                #     freq, src_word, tgt_word, nld = split_line
                # elif len(split_line) == 6:
                #     geo_mean, src_freq, tgt_freq, src_word, tgt_word, nld = split_line
                # else:
                #     message = "Lines must only have 4 (from parallel) or 6 (from NLD) columns:"
                #     message += "\n\t`freq ||| src_word ||| tgt_word || NLD`"
                #     message += "\n\tor\n\tgeo_mean ||| src_freq ||| tgt_freq ||| src_word ||| tgt_word ||| NLD`"
                #     raise ValueError(message)

                if len(split_line) < 4:
                    raise ValueError("Lines must have at least 4 columns where the first columns are frequencies and the last three columns are `src_word ||| tgt_word ||| nld`")
                src_word, tgt_word, _ = split_line[-3:]
                src_word = src_word.strip()
                tgt_word = tgt_word.strip()
                pairs.append((src_word, tgt_word))
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        return self.pairs[index]

