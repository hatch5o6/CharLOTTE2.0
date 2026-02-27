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
        vocab={}
    ):
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.unk = unk
        self.special_toks=[self.bos, self.eos, self.pad, self.unk]
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_toks)}
        for tok, idx in vocab.items():
            if tok in self.vocab:
                raise ValueError("Duplicate tok: `{tok}`")
            if idx in self.vocab.values():
                raise ValueError("Duplicate idx: `{idx}`")
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
    
    def encode(self, text: str, include_special=True, max_len=None):
        toks = [ch if ch in self.vocab else self.unk for ch in text]
        if max_len:
            toks = toks[:max_len]
        if include_special:
            if max_len:
                toks = toks[:-2]
            toks = [self.bos] + toks + [self.eos]
        if max_len:
            assert len(toks) == max_len
        idxs = [self.vocab[ch] for ch in toks]
        return toks, idxs
    
    def decode(self, idxs, remove_special=True):
        if remove_special:
            idxs = [idx for idx in idxs if idx not in self.special_toks]
        return "".join([self.id_to_char[ch] if ch in self.id_to_char else self.unk for ch in idxs])
    
    def __len__(self):
        return len(self.vocab)
    
    def __str__(self):
        return f"CharTokenizer, vocab size={len(self.vocab)}"
    
    def __repr__(self):
        repr_vocab = {tok: idx for tok, idx in self.vocab if tok not in self.special_toks}
        return f"CharTokenizer(bos={self.bos}, eos={self.eos}, pad={self.pad}, unk={self.unk}, vocab={repr_vocab})"

if __name__ == "__main__":
    pass
    # TODO write tests