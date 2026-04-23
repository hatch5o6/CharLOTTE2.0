# NMT Tokenizer Notes

## Training a Unigram Tokenizer with HuggingFace

Use `UnigramTrainer` from `tokenizers.trainers`. The result wraps into a `PreTrainedTokenizerFast`.

```python
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<en>", "<es>"]

trainer = UnigramTrainer(
    vocab_size=16000,
    special_tokens=special_tokens,
    unk_token="<unk>",
)

tokenizer = Tokenizer(Unigram())
tokenizer.pre_tokenizer = Metaspace()  # splits on whitespace, marks word boundaries with ▁

tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Post-processor: auto-inserts special tokens on every encode call
tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    pair="<bos> $A <eos> <bos> $B <eos>",
    special_tokens=[
        ("<bos>", tokenizer.token_to_id("<bos>")),
        ("<eos>", tokenizer.token_to_id("<eos>")),
    ],
)

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="<pad>",
    unk_token="<unk>",
    bos_token="<bos>",
    eos_token="<eos>",
)

fast_tokenizer.save_pretrained("./my_tokenizer")

# Load back
tokenizer = PreTrainedTokenizerFast.from_pretrained("./my_tokenizer")
```

**`Metaspace` vs `Whitespace`**: Metaspace is the standard choice for unigram/SentencePiece — it marks word beginnings with `▁` so the tokenizer knows where words start, which matters for decoding back to clean text.

**Training on a generator** (large corpora):
```python
def corpus_generator():
    for line in open("large_corpus.txt"):
        yield line.strip()

tokenizer.train_from_iterator(corpus_generator(), trainer=trainer, length=1000000)
```

**Adding language tags after wrapping**:
```python
fast_tokenizer.add_special_tokens({'additional_special_tokens': ['<en>', '<es>']})
```

---

## TemplateProcessing and BARTLightning.collate_fn

`BARTLightning.collate_fn` strips the tokenizer's auto-added special tokens with `[1:-1]`, then manually controls the sequence structure:

```python
tokenized_src = self.tokenizer(src).input_ids[1:-1]   # strips first and last token
tokenized_src = tokenized_src[:self.max_length - 2]   # -2 for lang tag + EOS
tokenized_src = tokenized_src + [self.tokenizer.eos_token_id]
if self.append_lang_tags:
    tokenized_src = [src_lang_token] + tokenized_src
```

Final sequence structure:
- Without lang tags: `[tokens...] <eos>`
- With lang tags: `<lang> [tokens...] <eos>`

This is the correct BART encoder format. BART expects `[tokens...] </s>` with no BOS on the encoder side. BOS (`decoder_start_token_id`) is prepended to the decoder input internally by the model.

**The `<bos> $A <eos>` TemplateProcessing works but is redundant** — BOS is added then immediately stripped by `[1:-1]`. It only needs to be there so `[1:-1]` always has something safe to strip.

**Cleaner alternative**: no TemplateProcessing, encode with `add_special_tokens=False`, let the collate own the sequence structure entirely:

```python
# In collate_fn — remove the [1:-1] strip and instead:
tokenized_src = self.tokenizer(src, add_special_tokens=False).input_ids
tokenized_src = tokenized_src[:self.max_length - 2]
tokenized_src = tokenized_src + [self.tokenizer.eos_token_id]
```

This makes it explicit that the collate owns the sequence structure and removes the implicit dependency on the tokenizer adding exactly one token on each side.

**Warning**: if you keep `<bos> $A <eos>` TemplateProcessing with the current `[1:-1]` strip, the `[1:-1]` is load-bearing — it will silently corrupt sequences if the template ever changes (e.g. to add two tokens at the start).
