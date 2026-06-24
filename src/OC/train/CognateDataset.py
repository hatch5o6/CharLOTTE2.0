from torch.utils.data import Dataset
import os

from OC.utilities.utilities import read_oc_data

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
        oc_data = read_oc_data(f)
        for row in oc_data:
            src_word, tgt_word = row[-3:-1]
            pairs.append((src_word, tgt_word))
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        return self.pairs[index]

