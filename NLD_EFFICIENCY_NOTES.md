# NLD Cognate Extraction — Efficiency Notes

These notes apply to `src/OC/extract_cognates/CognatesFromNLD.py`, which extracts cognate pairs from monolingual word lists using Normalized Levenshtein Distance (NLD). It builds a full pairwise distance matrix via `rapidfuzz.process.cdist`, then filters pairs below a threshold `theta`.

---

## Bottleneck 1: Python nested loop (lines 46–54) — the big one

`cdist` computes the full N×M distance matrix in C (fast). But the code then re-traverses the entire matrix in a Python nested loop just to apply the threshold — 400 million iterations for 20K×20K words.

**Fix:** use `np.where` to find matching indices directly in C:

```python
import numpy as np

src_idxs, tgt_idxs = np.where(matrix <= theta)
cognate_list = [
    (src_cts[s], tgt_cts[t], src_words[s], tgt_words[t], matrix[s][t])
    for s, t in zip(src_idxs, tgt_idxs)
]
cognate_list.sort(key=lambda x: x[4])
```

Goes from minutes → milliseconds for that step.

---

## Bottleneck 2: Word reading (lines 33–36) — IO + tokenizer bound

~11 minutes for 10M lines. Mostly disk IO (unavoidable) plus `tokenizer(line)` called once per line in Python.

If the tokenizer is spaCy, switch from per-line calls to `nlp.pipe()`:

```python
# slow: one call per line
for line in lines:
    words = tokenizer(line)

# fast: batched, uses spaCy's internal parallelism
for doc in nlp.pipe(lines, batch_size=2000, n_process=4):
    words = [token.text for token in doc]
```

For non-spaCy tokenizers (NLTK, regex), multiprocessing with `Pool.imap` on chunked lines is the equivalent approach.

---

## Bottleneck 3: `cdist` — already optimal

`rapidfuzz.process.cdist` runs in C/C++ and is as fast as you'll get for all-pairs Levenshtein without a custom C extension. No changes needed.

---

## Bug: `write_cognates` will crash (line 81)

`" ||| ".join(row)` raises `TypeError` because `row` contains ints and floats, not strings.

**Fix:**
```python
outf.write(" ||| ".join(str(x) for x in row) + "\n")
```
