# OC Training Code — Bug Report

## Confirmed Bugs

### 1. `OCLightning.py:67–69` — `calc_loss` reshape dimension mismatch

`outputs[:, 1:, :]` has shape `(B, T-1, V)` = `B*(T-1)` elements, but is reshaped to `(B*T, V)`. Same for `target_ids[:, 1:].reshape(B * T)`. Will raise a `RuntimeError` at runtime.

```python
T = outputs.size(1)  # tgt_len, but after slicing [:, 1:] it's tgt_len-1
outputs[:, 1:, :].reshape(B * T, V)  # B*T != B*(T-1) → crash
```

**Fix:** Use `T = outputs.size(1) - 1` (or compute the sliced size directly).

---

### 2. `train.py:198` — Iterating over a string, not checkpoint files

`os.path.join(checkpoints_d)` with a single argument simply returns the string itself. Iterating over it yields individual characters, not checkpoint filenames.

```python
for chkpt_file in os.path.join(checkpoints_d):  # wrong — iterates over chars
```

**Fix:** Use `os.listdir(checkpoints_d)` or `glob.glob(os.path.join(checkpoints_d, "*.ckpt"))`.

---

### 3. `train.py:320` — `-h` flag conflicts with argparse's built-in `--help`

```python
parser.add_argument("-h", "--hyp_words_out", ...)  # -h is reserved for --help
```

Argparse raises a conflict error on startup.

**Fix:** Use a different short flag, e.g. `-o`.

---

### 4. `train.py:262` — `@log_mode_call` on `inference` drops extra arguments

The decorator's `wrapper` only accepts `(config)` and internally calls `f(config)`, discarding all other arguments. But `inference` requires 4 arguments: `(config, chkpt_file, source_words_f, hyp_words_out)`.

- Calling `f(*f_args)` on line 341 passes 4 args to `wrapper(config)` → `TypeError`.
- Even if the wrapper signature were fixed, the interior `f(config)` call would still miss 3 required args.

**Fix:** Either remove `@log_mode_call` from `inference`, or generalise the decorator to pass through `*args`.

---

### 5. `train.py:41,44` — `rank_zero_info` called with two positional arguments

`rank_zero_info` delegates to Python's `logging.info`, which uses `%`-style formatting. Passing a second positional argument without a `%` specifier in the format string causes a `TypeError`.

```python
rank_zero_info("DELETING:", d)  # "DELETING:" % (d,) → TypeError
rank_zero_info("CREATING:", d)  # same
```

**Fix:** Use f-strings:
```python
rank_zero_info(f"DELETING: {d}")
rank_zero_info(f"CREATING: {d}")
```

---

## Minor Issues

### 6. `OC.py:270,290` — `self.to(self.device)` in `Seq2Seq.__init__`

Setting `self.device` as an instance attribute on an `nn.Module` shadows PyTorch's built-in device logic. Calling `self.to(...)` inside `__init__` also conflicts with PyTorch Lightning's device management, particularly under DDP — Lightning moves the model to the correct device itself.

---

### 7. `CharTokenizer.py:14` — Mutable default argument `vocab={}`

```python
def __init__(self, ..., vocab={}):
```

The `{}` is shared across all instances created without an explicit `vocab` argument. It doesn't cause a bug here since the dict is only read and not mutated, but it is a Python footgun.

**Fix:** Use `vocab=None` with an interior `if vocab is None: vocab = {}`.

---

### 8. Mixed `lightning` / `pytorch_lightning` imports

`train.py` imports from `lightning` (the unified package), while `OCLightning.py` imports from `pytorch_lightning`. These are typically compatible aliases, but mixing them is inconsistent and may cause issues across package versions.

---

## Summary Table

| # | File | Line | Severity | Issue |
|---|------|------|----------|-------|
| 1 | `OCLightning.py` | 67–69 | **Bug** | `calc_loss` reshape: uses `T` instead of `T-1` after slicing `[:, 1:]` |
| 2 | `train.py` | 198 | **Bug** | `os.path.join(checkpoints_d)` returns a string, not a directory listing |
| 3 | `train.py` | 320 | **Bug** | `-h` flag conflicts with argparse's built-in `--help` |
| 4 | `train.py` | 262 | **Bug** | `@log_mode_call` on `inference` drops extra args; wrapper calls `f(config)` only |
| 5 | `train.py` | 41, 44 | **Bug** | `rank_zero_info("...", d)` causes logging `%`-format `TypeError` |
| 6 | `OC.py` | 270, 290 | Minor | `self.to(self.device)` in `__init__` conflicts with Lightning device management |
| 7 | `CharTokenizer.py` | 14 | Minor | Mutable default argument `vocab={}` |
| 8 | Both | — | Minor | Inconsistent `lightning` vs `pytorch_lightning` imports |
