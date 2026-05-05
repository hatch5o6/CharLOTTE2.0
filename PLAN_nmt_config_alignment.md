# Plan: Config/Code Alignment Check for NMT Training

## Context

Before running a first real NMT training run, we need confidence that `test.yaml` and the NMT training code are aligned. Two failure modes exist:
- **YAML key unused in code** — silent waste, or a sign the key was renamed in code but not in the config
- **Code accesses a key not in YAML** — KeyError crash at runtime, possibly deep into a long run

The existing `confirm_config_keys.py` (OC) already solves direction 1. We add NMT files to it, and add a required-key validator for direction 2.

## Key background: how config keys flow

1. **YAML** contains flat keys (`parent_nmt_batch_size`, `child_nmt_batch_size`, etc.)
2. `read_config()` (`src/utilities/read_data.py`) injects computed keys: `{oc,parent_nmt,child_nmt,simple_nmt}_warmup_steps`, `sc_model_ids`, `nmt_corpus`, `nmt_reverse` — none of these need to be in the YAML
3. `_set_nmt_config` decorator (`src/NMT/train/train.py:33`) copies `{parent|child|simple}_nmt_*` → `nmt_*` at train time. So `nmt_batch_size` is never in the YAML; `parent_nmt_batch_size` is.
4. `config["tokenizer"]` is written by `train_model` itself after tokenizer training — also not in YAML.

The validator must check YAML-level keys (prefixed), not the derived unprefixed ones.

---

## Strategy 1: Extend `confirm_config_keys.py` (YAML → code)

**File:** `src/OC/train/confirm_config_keys.py`

The existing tool loads a YAML, grabs all top-level keys, and grep-searches for each key name (quoted) across a list of source files. Keys with zero matches are printed as unused.

**Change:** Add NMT source files to the search list alongside the OC files:
- `src/NMT/train/train.py`
- `src/NMT/train/NMTTokenizer.py`
- `src/NMT/train/BARTLightning.py`
- `src/utilities/read_data.py`
- `src/utilities/train_utilities.py`

This is a 3–5 line change. Run it once; any reported "unused" key needs a judgment call: SLURM-only keys (`*_mem`, `*_qos`, `*_timeout`, `email`, `methods`) are expected — anything else is worth investigating.

---

## Strategy 2: Add `_validate_nmt_config()` in `read_config()` (code → YAML)

**File:** `src/utilities/read_data.py`

Add a `REQUIRED_NMT_KEYS` constant and a `_validate_nmt_config(config)` function that collects all missing keys and raises a single `ValueError` listing them — so the full gap is visible in one shot rather than one KeyError at a time.

```python
REQUIRED_NMT_KEYS = [
    # shared
    "seed", "data", "sc_model_id_prefix", "save", "experiment_name",
    # NMT shared (tokenizer + datamodule)
    "nmt_vocab_size", "nmt_tokenizer_training_size", "nmt_tokenizer_ratios", "nmt_max_length",
    # warmup source (read_config computes warmup from these)
    "oc_max_steps",
    "parent_nmt_max_steps", "child_nmt_max_steps", "simple_nmt_max_steps",
    # per-prefix training + architecture params (copied to nmt_* by _set_nmt_config)
    *[f"{p}_nmt_{k}"
      for p in ("parent", "child", "simple")
      for k in (
          "batch_size", "n_gpus", "device", "patience", "save_top_k",
          "val_interval", "gradient_clip_val",
          "enc_num_layers", "enc_att_heads", "enc_ffn_dim", "enc_layerdrop",
          "dec_num_layers", "dec_att_heads", "dec_ffn_dim", "dec_layerdrop",
          "max_position_embeddings", "d_model", "dropout", "activation",
          "log_val_samples",
      )],
]

def _validate_nmt_config(config):
    missing = [k for k in REQUIRED_NMT_KEYS if k not in config]
    if missing:
        raise ValueError(
            f"Config is missing {len(missing)} required key(s):\n" +
            "\n".join(f"  {k}" for k in missing)
        )
```

Called as the first thing in `read_config()` after `read_yaml()`.

---

## Files to modify

| File | Change |
|------|--------|
| `src/OC/train/confirm_config_keys.py` | Add NMT source file paths to the search list (3–5 lines) |
| `src/utilities/read_data.py` | Add `REQUIRED_NMT_KEYS` + `_validate_nmt_config()`, call from `read_config()` |

---

## Verification

1. Run `confirm_config_keys.py` after extending it. Review "unused" keys — SLURM-only keys are expected; others warrant investigation.
2. Temporarily remove a key from `test.yaml` and call `read_config()` — should raise `ValueError` immediately listing the missing key(s).
3. Run with `test.yaml` intact — `read_config()` passes validation silently.
