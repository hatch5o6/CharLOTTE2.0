# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CharLOTTE** (Character-Level Orthographic Transfer for Translation Enhancement) is an NLP research system for improving cross-lingual knowledge transfer in NMT.

The core idea:
1. Train **OC (Orthographic Correspondence)** models — character-level seq2seq BiGRUs — on a cognate prediction task: given a word in source language A, predict its cognate in related language B.
2. By learning to predict cognates, OC models learn character-level mappings (orthographic correspondences) from A to B.
3. These learned mappings can reshape words in A to follow the spelling conventions of B, closing the orthographic gap between the two languages.
4. The reshaped text is then used to enhance knowledge transfer from A to B in a downstream **NMT** task.

The OC training code is substantially implemented in `code/OC/src/`. The NMT component (`code/NMT/src/`) is a future phase.

## Environment & Commands

Dependencies are managed with `uv`. The virtual environment is at `.venv/`.

```bash
# Install dependencies
uv sync

# Run OC training
cd code/OC/src
python train.py -c ../../configs/test.yaml -m TRAIN

# Run OC evaluation
python train.py -c ../../configs/test.yaml -m EVAL

# Run OC inference
python train.py -c ../../configs/test.yaml -m INFERENCE -k <checkpoint.ckpt> -w <source_words_file> -o <output_file>

# Run individual modules directly (each has a __main__ block for testing)
python CharTokenizer.py
python OCLightning.py
python metrics.py
```

Scripts in `code/OC/src/` import each other by module name (no package), so they must be run from that directory.

## Architecture

### OC (Orthographic Correspondence) Model

The OC model is a character-level seq2seq system for learning mappings between cognate word pairs across languages.

**Data format** (`CognateDataset`): Each line of a data file has the format:
```
<freq> ||| <src_word> ||| <tgt_word> ||| <nld>
```

**Pipeline**:
1. `CharTokenizer` — builds character-level vocab from corpus; special tokens `<bos>`, `<eos>`, `<pad>`, `<unk>` always occupy indices 0–3. Separate src and tgt tokenizers are used but must have matching special tokens.
2. `CognateDataset` — reads word pairs from data files.
3. `OCDataModule` (in `OCLightning.py`) — PyTorch Lightning DataModule; handles collation with padding.
4. `Seq2Seq` (in `OC.py`) — BiGRU encoder + GRU decoder with Luong dot-product attention. Training uses teacher forcing; inference uses beam search.
5. `OCLightning` — PyTorch Lightning wrapper; uses AdamW with linear warmup schedule. Monitors `val_loss` for checkpointing and early stopping.
6. `train.py` — CLI entry point; reads a YAML config and dispatches to `train_model`, `eval_models`, or `inference`.

**Config keys** (all prefixed `oc_`): `oc_train`, `oc_val`, `oc_device`, `oc_n_gpus`, `oc_max_steps`, `oc_batch_size`, `oc_learning_rate`, `oc_weight_decay`, `oc_gradient_clip_val`, `oc_save_top_k`, `oc_patience`, `oc_val_interval`, `oc_enc_embed_dim`, `oc_enc_hidden_dim`, `oc_enc_num_layers`, `oc_dec_embed_dim`, `oc_dec_hidden_dim`, `oc_dec_num_layers`, `oc_dropout`, `oc_max_length`, `oc_n_beams`, `oc_log_train_samples` (log training samples every N batches). Warmup steps are auto-computed as `oc_max_steps // 20` and injected as `oc_warmup_steps`. Additional top-level keys: `seed`, `experiment_name`, `save`. `oc_qos` is SLURM-only (not read by Python code). `pl_cl`, `pl_tl`, `cl_tl` are future language-pair placeholders (currently null).

**Output structure**: Saves to `{save}/{experiment_name}/OC/` with subdirs `checkpoints/`, `data/`, `predictions/`, `logs/`, `tb/`.

**Metrics**: `calc_chrF` and `calc_charBLEU` via sacrebleu (character-level BLEU). Best checkpoint selected by chrF.

### NMT Module

`code/NMT/src/` exists but `nmt.py` and `train.py` are currently empty stubs.

### Key design notes

- `Seq2Seq.forward()` with `tgt=None` triggers beam search inference; with `tgt` provided it does teacher-forced training.
- `OC.new.py` is an in-progress alternative implementation (untracked).
- The `oc_device` config key is used as `accelerator` in `L.Trainer` — must be a valid Lightning accelerator value (e.g., `"cuda"`, `"cpu"`). The active `OC.py`'s `Seq2Seq` does NOT store `oc_device` as a `torch.device`; it uses `src.device` for all tensor creation, which is the correct Lightning-compatible approach.
