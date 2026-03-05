from torch import Tensor, tensor

def read_file(f):
    with open(f) as inf:
        content = inf.read()
    return content

class CharTokenizer:
    def __init__(self, 
        bos="<bos>",
        eos="<eos>",
        pad="<pad>",
        unk="<unk>",
        vocab:dict=None
    ):
        if vocab is None:
            vocab = {}
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.unk = unk
        self.special_toks=[self.bos, self.eos, self.pad, self.unk]
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_toks)}
        self.special_tok_ids = [self.vocab[tok] for tok in self.special_toks]
        for tok, idx in vocab.items():
            if tok in self.vocab:
                raise ValueError(f"Duplicate tok: `{tok}`")
            if idx in self.vocab.values():
                raise ValueError(f"Duplicate idx: `{idx}`")
            self.vocab[tok] = idx
        self.id_to_char = {idx: tok for tok, idx in self.vocab.items()}
        assert sorted(list(self.vocab.keys())) == sorted(list(self.id_to_char.values()))
        assert sorted(list(self.vocab.values())) == sorted(list(self.id_to_char.keys()))
        self.next_id = len(self.vocab)
        for idx in self.vocab.values():
            assert self.next_id > idx
        for idx in self.id_to_char.keys():
            assert self.next_id > idx

    def to_idx(self, tok):
        return self.vocab.get(tok, self.vocab[self.unk])
    
    def pad_idx(self):
        return self.to_idx(self.pad)
    
    def bos_idx(self):
        return self.to_idx(self.bos)
    
    def eos_idx(self):
        return self.to_idx(self.eos)
    
    def unk_idx(self):
        return self.to_idx(self.unk)

    def build_vocab(self, corpus):
        """Build vocabulary from a text corpus."""
        if isinstance(corpus, str):
            try:
                text = set(read_file(corpus))
            except:
                text = set(corpus)
        else:
            if not isinstance(corpus, list):
                raise ValueError("corpus must be a text corpus (str), a file path (str), or a list of file paths.")
            text = set()
            for f in corpus:
                text.update(read_file(f))

        for ch in sorted(text):
            if ch not in self.vocab:
                self.vocab[ch] = self.next_id
                self.id_to_char[self.next_id] = ch
                self.next_id += 1
    
    def encode(self, text: str, include_special=True, max_len=None, return_tensor=False):
        toks = [ch if ch in self.vocab else self.unk for ch in text]
        
        if max_len and include_special:
            toks = toks[:max_len - 2]
        elif max_len:
            toks = toks[:max_len]

        if include_special:
            toks = [self.bos] + toks + [self.eos]

        if max_len:
            assert len(toks) <= max_len

        idxs = [self.vocab[ch] for ch in toks]
        if return_tensor:
            idxs = tensor(idxs)
        return toks, idxs
    
    def decode(self, idxs, remove_special=True):
        if isinstance(idxs, Tensor):
            idxs = idxs.tolist()
        if remove_special:
            to_remove = [idx for idx in self.special_tok_ids if idx != self.unk_idx()]
            idxs = [idx for idx in idxs if idx not in to_remove]
        return "".join([self.id_to_char[ch] if ch in self.id_to_char else self.unk for ch in idxs])
    
    def batch_decode(self, batch, remove_special=True):
        decoded_batch = []
        for sequence in batch:
            decoded_batch.append(self.decode(sequence, remove_special=remove_special))
        return decoded_batch
    
    def __len__(self):
        return len(self.vocab)
    
    def __str__(self):
        return f"CharTokenizer, vocab size={len(self.vocab)}"
    
    def __repr__(self):
        repr_vocab = {tok: idx for tok, idx in self.vocab.items() if tok not in self.special_toks}
        return f"CharTokenizer(bos={self.bos}, eos={self.eos}, pad={self.pad}, unk={self.unk}, vocab={repr_vocab})"

if __name__ == "__main__":
    tokenizer = CharTokenizer()
    tokenizer.build_vocab("This is a test.")
    print("BOS:", tokenizer.bos, tokenizer.bos_idx())
    print("EOS:", tokenizer.eos, tokenizer.eos_idx())
    print("PAD:", tokenizer.pad, tokenizer.pad_idx())
    print("UNK:", tokenizer.unk, tokenizer.unk_idx())
    print("VOCAB:", tokenizer.vocab)
    print("TO IDX `t`:", tokenizer.to_idx('t'), " `<pad>`:", tokenizer.to_idx("<pad>"))
    print("LEN:", len(tokenizer))
    print("STR:", str(tokenizer))
    print("REPR:", repr(tokenizer))

    text = "is an asset"
    print(f"\nENCODE `{text}`")
    print("\tNORMAL:", tokenizer.encode(text))
    print("\tNOT INCLUDE SPECIAL:", tokenizer.encode(text, include_special=False))
    print("\tMAX_LEN (8):", tokenizer.encode(text, max_len=8))
    print("\tMAX_LEN (8) and NOT INCLUDE SPECIAL", tokenizer.encode(text, include_special=False, max_len=8))
    print("\tMAX_LEN (8) and RETURN TENSOR", tokenizer.encode(text, max_len=8, return_tensor=True))

    _, tokens = tokenizer.encode(text)
    tokens_tensor = tensor(tokens)
    print(f"\nDECODE {tokens}")
    print("\tNORMAL:", tokenizer.decode(tokens))
    print("\tNOT REMOVE SPECIAL:", tokenizer.decode(tokens, remove_special=False))
    print("DECODE TENSOR", tokens_tensor)
    print("\tDECODE TENSOR:", tokenizer.decode(tokens_tensor))

