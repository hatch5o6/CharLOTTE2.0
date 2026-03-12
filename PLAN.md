# CharLOTTE 2.0 — Project Plan

## Overview

**CharLOTTE** (Character-Level Orthographic Transfer for Translation Enhancement) is an NLP research system for improving cross-lingual knowledge transfer in neural machine translation (NMT).

### Terminology

- **Parent language** — a high-resource language closely related to the child language (e.g. Spanish).
- **Child language** — a low-resource language related to the parent (e.g. Aragonese).
- **Source / target language** — the input and output languages of the NMT task (may be unrelated to the cognate pair).

### Pipeline

The pipeline has three major phases:
1. **Cognate Extraction** — Extract cognate word pairs from parent/child parallel corpora to produce training data for OC models.
2. **OC (Orthographic Correspondence)** — Train one character-level seq2seq model per parent→child language pair on the cognate prediction task, learning orthographic mappings from parent to child. Apply the trained model to reshape every word in the parent-language text of a translation corpus so that it exhibits the spelling conventions of the child language.
3. **NMT** — Train an NMT model on source→parent (or parent→target) data, then fine-tune on source→child (or child→target) data using the reshaped parent corpus as a substitute for child-language data. The reshaped corpus closes the orthographic gap between parent and child, improving transfer. The parent language always occupies the target side for source→parent corpora and the source side for parent→target corpora, so the reshaped side varies accordingly.

Standalone OC model quality is evaluated using chrF and character-level BLEU (best checkpoint selected by chrF). End-to-end quality is measured by downstream NMT performance, but this is a future concern — current focus is on producing a good OC model.

---

## Phase 1: Cognate Extraction — PARTIALLY IMPLEMENTED

Goal: extract cognate word pairs from parent/child parallel sentence corpora to produce training data for OC models. Output format per line: `<freq> ||| <parent_word> ||| <child_word> ||| <nld>`, where `freq` is the co-occurrence frequency of the word pair in the corpus.

**Strategy:**
- Use **FastAlign** (via eflomal + fast_align symmetrization) for word alignment — subprocess-based integration
- Filter aligned pairs by NLD (normalized Levenshtein distance) threshold `theta` (default 0.5) to identify cognate candidates

**Pipeline sketch:**
```
prepare_fastalign_input()  →  run_fastalign()  →  parse_alignments()  →  extract_cognates()
```

Each stage operates on files for debuggability and recoverability.

Note: pre-extracted cognate data already exists and is referenced in `src/configs/test.yaml`, so OC training (Phase 2) can proceed before this module is fully complete.

**Implementation status:**

| File | Location | Status |
|------|----------|--------|
| `CognatesFromParallel.py` | `src/OC/extract_cognates/` | **Done** — eflomal/fast_align integration, NLD filtering |
| `CognatesFromMonolingual.py` | `src/OC/extract_cognates/` | Stub — function signature only |
| `utilities.py` | `src/OC/utilities/` | **Done** — `split()`, `write_oc_data()` |
| `word_tokenizers.py` | `src/OC/utilities/` | **Done** — language-specific tokenizers (spaCy, NLTK, Camel, IndICNLP) |
| `pipeline.py` | `src/Pipeline/Pipeline/` | **Done** — orchestrates extract → split → write OC data |
| `read_data_csv.py` | `src/Pipeline/utilities/` | **Done** — reads language pair CSV |

**Config keys used by the pipeline:** `pl_cl_para` / `pl_cl_mono` (exactly one must be set), `theta`, `oc_val_ratio`, `seed`, `experiment_name`, `save`.

**Remaining work:** implement `CognatesFromMonolingual.py` for monolingual extraction path.

---

## Phase 2: OC Module

### 2.1 OC Training Code — COMPLETE

One OC model is trained per parent→child language pair.

Core files in `src/OC/train/`:

| File | Role | Status |
|------|------|--------|
| `CharTokenizer.py` | Character-level tokenizer with BOS/EOS/PAD/UNK | Done |
| `CognateDataset.py` | PyTorch Dataset for cognate word pairs | Done |
| `OC.py` | BiGRU encoder + GRU decoder with Luong attention + beam search | Done |
| `OCLightning.py` | PyTorch Lightning wrapper (training, validation, predict steps) | Done |
| `train.py` | CLI entry point; TRAIN / EVAL / INFERENCE modes | Done |
| `metrics.py` | chrF and character-level BLEU via sacrebleu | Done |
| `OC_bahdanau.py` | Alternative Seq2Seq with Bahdanau (additive) attention | Experimental |
| `confirm_config_keys.py` | Developer utility: verifies config key usage across source files | Done |
| `OCPipeline.py` | Future pipeline orchestration stub | Stub |

Config lives in `src/configs/`. Test config: `src/configs/test.yaml`.

Full bug history documented in `OC_TRAIN_ERRORS_3_5.md`.

#### Hyperparameter Search

Optuna will be used to tune OC model hyperparameters. **Note:** Optuna is not yet in `pyproject.toml` and must be added as a dependency before this work begins. The NLD threshold (default: 0.5) is also tunable within the same sweep. The NMT model (Phase 3) uses transformer-base with fixed hyperparameters and is not subject to Optuna search.

A minimal stub `OCPipeline.py` also exists in `src/OC/train/` for future pipeline orchestration.

### 2.2 OC Inference / Text Reshaping — NOT STARTED

Goal: apply a trained OC model to every word in the parent-language side of a translation corpus, replacing each word with the model's prediction to produce a reshaped corpus that exhibits child-language spelling conventions. Replacement strategy (e.g. based on model confidence or unconditional) is TBD.

---

## OC Model Design Principles

### Framework Reuse

Where well-maintained libraries provide implementations of components used in the OC model — such as seq2seq BiGRU architectures or Luong dot-product attention — those libraries should be evaluated for either:
- **Direct use** as drop-in replacements for the corresponding custom implementation, or
- **Validation** of the custom implementation against their reference behaviour.

Any library adopted must satisfy two constraints:
1. **Optuna compatibility** — must not obstruct hyperparameter search (i.e. model construction must be parameterisable and callable from within an Optuna trial).
2. **Pipeline compatibility** — must integrate cleanly into a Python-based pipeline without requiring framework-specific entrypoints or opaque abstractions.

The current OC model is built directly on PyTorch primitives (`nn.GRU`, `nn.Linear`, `F.softmax`), which satisfy both constraints by default. Any higher-level library adopted must meet the same bar.

---

## Phase 3: NMT Module — NOT STARTED

`src/NMT/train/nmt.py` and `src/NMT/train/train.py` are currently empty stubs.

**Architecture:** transformer-base (fixed; no hyperparameter search).

**Training strategy:** train on source→parent (or parent→target), then fine-tune on source→child (or child→target) using the reshaped parent corpus from Phase 2 as a substitute for child-language data.

Further planning TBD pending completion of Phase 2.

---

## Environment

- Dependency manager: `uv` (venv at `.venv/`)
- Training framework: PyTorch Lightning (`lightning` package)
- Logging: CSV + TensorBoard (`tensorboard` package)
- Metrics: `sacrebleu`
- Alignment: FastAlign (external binary, subprocess integration)
