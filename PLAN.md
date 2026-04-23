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
3. **NMT** — Train an NMT model on parent→target (or source→parent) data, then fine-tune on child→target (or source→child) data using the reshaped parent corpus (parent') as the parent-target set. The reshaped corpus closes the orthographic gap between parent and child, improving transfer. The parent language always occupies the target side for source→parent corpora and the source side for parent→target corpora, so the reshaped side varies accordingly.

Standalone OC model quality is evaluated using chrF and character-level BLEU (best checkpoint selected by chrF). End-to-end quality is measured by downstream NMT performance, but this is a future concern — current focus is on producing a good OC model.

---

## Phase 1: Cognate Extraction — PARTIALLY IMPLEMENTED

Goal: extract cognate word pairs from parent/child corpora to produce training data for OC models. Three extraction methods are supported, all producing cognate pairs filtered by NLD ≤ theta.

### Output Formats

All methods write cognate files with ` ||| ` delimiters. `validation.py` auto-detects format by column count:

| Method | File format | Columns |
|--------|-------------|---------|
| charlotte | `ct \|\|\| word1 \|\|\| word2 \|\|\| nld` | 4 |
| web | `ct \|\|\| word1 \|\|\| word2 \|\|\| nld` | 4 |
| fuzz | `src_ct \|\|\| tgt_ct \|\|\| word1 \|\|\| word2 \|\|\| nld` | 5 |

### Methods

**`charlotte`** — uses fast_align on an existing PL↔CL parallel corpus:
```
write_fast_align_input()  →  fast_align (fwd + rev)  →  atools symmetrize  →  get_word_pairs()  →  finalize_parallel()
```

**`web`** (With Engineered Bitext) — synthesizes a PL↔CL corpus when none exists:
```
train TL→PL NMT  →  translate TL side of CL↔TL corpus  →  synthetic PL↔CL  →  fast_align  →  get_word_pairs()  →  finalize_parallel()
```
Blocked on NMT module.

**`fuzz`** — monolingual: finds best NLD match for every PL word in the CL vocabulary:
```
get_words(pl)  +  get_words(cl)  →  cdist NLD matrix  →  greedy 1-to-1 matching  →  finalize_monolingual()
```

### Shared Post-Processing (`cognate_filters.py`)

All three methods pass their candidates through shared filters before writing:
1. Short-word filter — drop pairs where either word has `len < 4`
2. Number handling — if both words are purely numeric (`str.isdigit()`): discard original pair, add identity pairs `(word, word, nld=0.0)` with original count
3. NLD filter — drop pairs where `NLD(word1, word2) > theta`

### Shared Validation Set

A single val set is drawn from `charlotte` cognate pairs (or `web` if no parallel data). This ensures all OC models — regardless of which extraction method produced their training data — are evaluated on the same gold-standard pairs.

**Flow:**
1. Run all configured methods → write cognate files
2. Call `get_train_val_split(pairs, size=900)` from `TrainValSplit.py`:
   - Val pairs sampled from charlotte/web file (NLD-stratified, default size 900)
   - Val pairs removed from each method's training set via `get_train_split(pairs, val_pairs)`
   - Outputs: shared `.val` file + per-method `.train` files
3. Each OC model trained on its method's `.train` file; all evaluated on shared `.val` file

### Implementation Status

| File | Location | Status |
|------|----------|--------|
| `CandidatesFromParallel.py` | `src/OC/extract_cognates/` | **Done** — fast_align integration; returns candidate word pairs |
| `FuzzyCandidates.py` | `src/OC/extract_cognates/` | **Done** — greedy NLD matrix matching; returns candidate word pairs |
| `CandidatesFromSyntheticParallel.py` | `src/OC/extract_cognates/` | **Stub** — blocked on NMT module |
| `FilterCognatePairs.py` | `src/OC/extract_cognates/` | **Done** — NLD filtering, short-word filter, decimal identity pairs; handles both 4- and 5-column formats |
| `TrainValSplit.py` | `src/OC/extract_cognates/` | **Done** — `get_train_val_split()` (NLD-stratified) and `get_train_split()` both implemented |
| `utilities.py` | `src/OC/utilities/` | **Done** — `NLD()`, `write_oc_data()` |
| `word_tokenizers.py` | `src/OC/utilities/` | **Done** — language-specific tokenizers (spaCy, NLTK, Camel, IndICNLP) |
| `pipeline.py` | `src/Pipeline/Pipeline/` | **Partial** — extract, filter, and val set split all wired; debug `exit()` on line 200 blocks job submission |
| `read_data.py` | `src/OC/data/` | **Done** — reads `data` key from YAML config into pl/cl file path pairs |

**Config keys used by the pipeline:** `data` (list of `[data_folder, pl, cl, tl]` lists), `theta`, `oc_val_size` (default 900), `seed`, `experiment_name`, `save`.

**Remaining work:**
- Remove debug `exit()` from `pipeline.py` (line 200) once pipeline is verified end-to-end
- Implement `CandidatesFromSyntheticParallel.py` once NMT module is ready

### Pipeline Tests (`src/Pipeline/Pipeline/test_pipeline.py`)

**Strategy:** mock `extract_candidates` to return a fixed known set of candidates; use real `filter_cognate_pairs`, `get_train_val_split`, `get_train_split`, `write_oc_data`, `read_oc_data`; use pytest `tmp_path` for all file I/O.

| Test | What it checks |
|------|----------------|
| `test_get_scenario_directory_create` | `create=True` produces the expected directory tree with all subdirs |
| `test_get_scenario_directory_no_create_missing` | `create=False` raises `FileNotFoundError` when directories are absent |
| `test_get_scenario_directory_no_create_exists` | `create=False` succeeds when the tree already exists |
| `test_get_oc_data_val_building_mode` | `.val` file written, `.train` file NOT written, `.cognates` file written; returned dict has correct keys and existing val paths |
| `test_get_oc_data_full_pipeline_mode` | `.train` file written; val content on disk matches reference; no word pair in train appears in val |
| `test_shared_val_set_across_methods` | Run `get_oc_data` for two different methods with the same `validation_set_files`; all resulting val files have identical content; no val pair appears in any method's train file |
| `test_only_pl_cls_filtering` | With two pairs in config, `only_pl_cls={(es, an)}` processes only that pair; other pair gets no val file |
| `test_only_pl_cls_empty_set` | `only_pl_cls=set()` produces empty `oc_data` (no pairs processed) |
| `test_only_pl_cls_rejects_with_validation_set_files` | Passing both `only_pl_cls` and `validation_set_files` raises `AssertionError` |
| `test_config_not_mutated_by_run_experiment` | After `run_experiment` returns, caller's `config["experiment_name"]` is unchanged |
| `test_assert_no_train_contamination_pass` | Disjoint train/val word pairs → no assertion |
| `test_assert_no_train_contamination_overlap` | Word pair appearing in both train and val → `AssertionError` |
| `test_assert_no_train_contamination_duplicate_in_train` | Duplicate word pair within train → `AssertionError` |
| `test_cognates_file_consistency_check_match` | Second call to `get_oc_data` with same candidates finds existing `.cognates` file and passes |
| `test_cognates_file_consistency_check_mismatch` | Second call with different candidates finds existing `.cognates` file and raises `AssertionError` |
| `test_validate_pl_cl_configs_pass` | Correctly constructed `pl_cl_configs` with only `oc_train`/`oc_val` added → no assertion |
| `test_validate_pl_cl_configs_wrong_path` | Mutated `oc_train` path → `AssertionError` |
| `test_validate_pl_cl_configs_extra_key` | Unexpected extra key in a pl_cl_config → `AssertionError` |

---

## Phase 2: OC Module

### 2.1 OC Training Code — COMPLETE (bugs present)

One OC model is trained per parent→child language pair.

Core files in `src/OC/train/`:

| File | Role | Status |
|------|------|--------|
| `CharTokenizer.py` | Character-level tokenizer with BOS/EOS/PAD/UNK | Done |
| `CognateDataset.py` | PyTorch Dataset for cognate word pairs | Done |
| `OC.py` | BiGRU encoder + GRU decoder with Luong attention + beam search | Done |
| `OCLightning.py` | PyTorch Lightning wrapper (training, validation, predict steps) | Done |
| `train.py` | CLI entry point; TRAIN / EVAL / INFERENCE modes | Done (bugs — see below) |
| `metrics.py` | chrF and character-level BLEU via sacrebleu | Done |
| `OC_bahdanau.py` | Alternative Seq2Seq with Bahdanau (additive) attention | Experimental |
| `confirm_config_keys.py` | Developer utility: verifies config key usage across source files | Done |

Config lives in `src/configs/`. Test config: `src/configs/test.yaml`.

Full bug history documented in `OC_TRAIN_ERRORS_3_5.md`.

**Bugs in `train.py`:**
1. `get_datamodule()` (lines 56–57) and `inference()` (lines 232–233) call `OCDataModule` with `train_f=`/`val_f=` but `OCDataModule.__init__` expects `train=`/`val=` — raises `TypeError` on TRAIN and INFERENCE.
2. INFERENCE CLI args passed to `inference()` in wrong order (line 314): `[chkpt_file, source_words, hyp_words_out]` but signature is `inference(config, source_words_f, hyp_words_out, chkpt_file=None)`.
3. `eval_models()` stores sacrebleu score objects directly in `scores` dict; `write_scores()` calls `json.dumps(scores)` which raises `TypeError` because sacrebleu objects are not JSON serializable — should store `.score` (float) instead.

#### Hyperparameter Search

Optuna will be used to tune OC model hyperparameters. **Note:** Optuna is not yet in `pyproject.toml` and must be added as a dependency before this work begins. The NLD threshold (default: 0.5) is also tunable within the same sweep. The NMT model (Phase 3) uses transformer-base with fixed hyperparameters and is not subject to Optuna search.

### 2.2 OC Inference / Text Reshaping — PARTIAL (bugs present)

Goal: apply a trained OC model to every word in the parent-language side of a translation corpus, replacing each word with the model's prediction to produce a reshaped corpus that exhibits child-language spelling conventions. Replacement strategy (e.g. based on model confidence or unconditional) is TBD.

`src/OC/reshape/reshape.py` exists with `prepare_source_words()` and `reshape_data()` implemented, but has two bugs that must be fixed before use:
1. `from OC.utilities.word_preprocessing import clean, is_only_punct` — `is_only_punct` is private (`_is_only_punct`); causes `ImportError`
2. `clean(w)` and `clean(word)` called without the required `long_enough` argument (lines 11 and 27); causes `TypeError`

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

## Phase 3: NMT Module — PARTIAL

**Architecture:** BART (`BartForConditionalGeneration`) initialized from scratch via `BartConfig` — encoder-decoder transformer. Fixed hyperparameters; no Optuna search.

**Training strategy:** train on parent→target (or source→parent), then fine-tune on child→target (or source→child) using the reshaped parent corpus (parent') from Phase 2. Research ends at evaluation (EVAL mode); `inference()` will remain a stub.

### Implementation Status

| File | Role | Status |
|------|------|--------|
| `train.py` | CLI entry point; TRAIN / EVAL modes | Done (INFERENCE stub intentional) |
| `BARTLightning.py` | BART model + Lightning DataModule | Done |
| `NMTTokenizer.py` | Unigram tokenizer training and loading | Done |
| `ParallelDatasets.py` | `CharLOTTEParallelDataset` for parallel corpora | Done |
| `utilities/metrics.py` | chrF++, spBLEU, charBLEU via sacrebleu | Done (bug — see below) |
| `utilities/read_data.py` | `read_config()`, `get_set()`, data path helpers | Done |

**Known bugs:**
1. `utilities/metrics.py` — `print_score` decorator hardcodes `wrapper(hyp, ref)` with no `**kwargs`, making `calc_spBLEU`'s `tokenizer` parameter unreachable after decoration. Default `"flores200"` is always used.

**Config keys:** all `nmt_*` keys present in `test.yaml` as stubs; must be filled before training.

---

## Environment

- Dependency manager: `uv` (venv at `.venv/`)
- Training framework: PyTorch Lightning (`lightning` package)
- Logging: CSV + TensorBoard (`tensorboard` package)
- Metrics: `sacrebleu`
- Alignment: FastAlign (external binary, subprocess integration; eflomal removed — non-deterministic)
