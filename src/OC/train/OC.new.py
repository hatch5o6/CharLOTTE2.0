from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Encoder(nn.Module):
    def __init__(self, config, tokenizer):
        super(Encoder, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(len(tokenizer), config["OC_embed_size"], padding_idx=tokenizer.pad_idx())
        self.dropout = nn.Dropout(config["OC_dropout"])
        self.gru = nn.GRU(
            input_size=config["OC_embed_size"],
            hidden_size=config["OC_hidden_size"],
            num_layers=config["OC_num_layers"],
            batch_first=True,
            dropout=config["OC_dropout"],
            bidirectional=True
        )

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class Decoder(nn.Module):
    pass

class Seq2Seq(nn.Module):
    pass