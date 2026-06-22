# CharLOTTE 2.0 — End-to-End Testing Plan

## Context

CharLOTTE is an **experimental research framework** whose job is to produce a valid comparison table (`NMT_scores.txt`) of:
- Standard NMT baselines (simple, parent→child transfer)
- OC-augmented NMT (same baselines, but source text is reshaped to look more like the child language via OC model predictions)

"High confidence" in this framework means three things:
1. **Data validity**: no contamination between train/val, OC reshaping produces correct text, cognate filtering is correct
2. **Reproducibility**: identical config → identical scores (already addressed for NMT)
3. **Pipeline correctness**: each step produces outputs in the format the next step expects; partial pipeline runs via `--pipeline` CLI work correctly

The strategy is **tiers**: fast unit tests on logic-heavy functions first, then component integration tests, then pipeline CLI tests. Skip testing every internal function — focus on silent-failure boundaries.

---

## Bugs Noticed While Planning
(Tests will surface these — worth fixing before or while writing tests)

| File | Line | Bug |
|------|------|-----|
| `pipeline.py` | 157 | `mem_gb=["basic_mem"]` — list literal instead of `config["basic_mem"]` |
| `pipeline.py` | 131, 153 | `config["mail"]` — should be `config["email"]` (all other places use `email`) |
| `pipeline.py` | 769 | `args.inlude_pairs` — typo, should be `args.include_pairs` |
| `pipeline.py` | 773–787 | CLI validation logic is inverted: `set(PIPELINE).intersection(args.pipeline) != set()` is **always True** for valid input; should check `set(args.pipeline).difference(set(PIPELINE)) != set()` |

---

## Tier 1: Fast Unit Tests
*Target: < 30 seconds per test, pure logic, no disk I/O or model loading*

### 1a. `src/OC/extract_cognates/tests/test_cognates.py` *(new file)*

Tests `Cognates._filter_cognate_pairs` and `make_cognates` (mock `extract_candidates`).

| Test | What it verifies |
|------|-----------------|
| Pair with NLD > theta is excluded | Threshold filtering is correct |
| Pair with NLD == theta is included | Threshold is inclusive |
| Pair with NLD < theta is included | Basic pass-through |
| Word shorter than `long_enough` is excluded | Min word length filter |
| Decimal pairs are consolidated and always included regardless of theta | Decimal pair logic |
| 3-tuple input `(freq, word1, word2)` and 5-tuple `(freq1, freq2, word1, word2)` both work | Both input formats supported |
| Output tuples are 4-tuples for parallel and 5-tuples for fuzz | Output format correct |
| `make_cognates` writes `.cognates` file and re-reading it matches return value | write→read round-trip |
| `make_cognates` re-run on same data asserts existing file matches | Idempotency check |

### 1b. `src/OC/extract_cognates/tests/test_train_val_split.py` *(new file)*

Tests `TrainValSplit.get_train_val_split` and `get_train_split`.

| Test | What it verifies |
|------|-----------------|
| No word pair appears in both train and val | Non-contamination (core guarantee) |
| No source word appears more than once across train+val | Source-side uniqueness |
| No target word appears more than once across train+val | Target-side uniqueness |
| Val size is approximately `size` with tolerance for bucket caps | Val size constraint |
| Buckets at lower NLD (closer cognates) are represented in val | NLD stratification |
| `get_train_split` with a val set removes val pairs from training | Cross-method shared val logic |
| 4-tuple input (parallel) and 5-tuple input (fuzz) both work | Both data formats |
| Reproducible: same seed → same split | Seed reproducibility |
| `_ensure_unique_words` removes pairs sharing a source or target word | Dedup logic |

### 1c. `src/OC/reshape/tests/test_reshape.py` *(new file)*
*This is the core hypothesis of CharLOTTE — correctness here is critical.*

Tests `reshape.reshape_data` and `reshape.prepare_source_words`.

| Test | What it verifies |
|------|-----------------|
| Known word in mappings is replaced in output | Basic replacement works |
| Word not in mappings is passed through unchanged | Unknown words preserved |
| Replacement is substring-aware (only replaces cleaned_word within token) | Partial token replacement |
| Line count in output equals line count in input | No lines added/dropped |
| Output file is created with correct `output_tag` | File naming |
| Function raises if output file already exists | No silent overwrite |
| `prepare_source_words` output is valid OC data format (4-tuple, freq=-1, tgt=`<N/A>`) | Source words file format |
| `prepare_source_words` output has no duplicate words | Words are deduplicated |
| `prepare_source_words` words are sorted | Deterministic ordering |

### 1d. Fill in `src/utilities/tests/test_metrics.py` *(existing empty file)*

| Test | What it verifies |
|------|-----------------|
| `calc_chrF_plus_plus(hyp, ref)` with perfect match returns 100 | Score range |
| `calc_chrF_plus_plus(hyp, ref)` with empty hyp returns near 0 | Degenerate case |
| `calc_spBLEU` and `calc_BLEU` return floats | Type check |
| Mismatched hyp/ref lengths raise via `@validate_lens` | Length validation decorator |

---

## Tier 2: Component Integration Tests
*Target: 1–5 minutes per test, uses toydata, no GPU needed*

### 2a. `src/OC/train/tests/test_oc_cycle.py` *(new file)*

Runs the full OC train → eval → inference cycle on toydata (local mode, CPU, minimal steps).
Uses a config derived from the existing OC test config with `oc_device: cpu`, `oc_max_steps: 20`.

| Test | What it verifies |
|------|-----------------|
| `train_model(config)` completes and creates checkpoint files | OC training runs |
| `eval_models(config)` produces `predictions/scores.json` with expected keys | OC eval runs |
| `scores.json` keys are checkpoint paths + `"BEST_VAL_chrF"` | Scores format correct |
| `inference(config, source_words_f, chkpt_file)` produces a predictions file with same line count as input | OC inference produces correct output |
| OC train is reproducible: same config twice → same checkpoint filenames and same val scores | OC reproducibility |

### 2b. `src/Pipeline/Pipeline/tests/test_prepare_oc_data.py` *(new file)*

Runs `_write_oc_data` and `prepare_OC_data` on toydata (local, no HPC).

| Test | What it verifies |
|------|-----------------|
| `_write_oc_data` with `charlotte` method creates `OC/charlotte/xx-yy/` directory structure | Directory creation |
| `train.txt` and `val.txt` exist in `data/` subdir | Files are written |
| `train.txt` and `val.txt` are valid OC data format | Format correct |
| No pair appears in both train and val | Contamination check |
| `_assert_no_train_contamination` raises on contaminated sets | Guard function works |

---

## Tier 3: Pipeline CLI Integration Tests
*Target: tests that invoke `pipeline.py` via subprocess with `use_hpc: False` and toydata*

**Requires a new config**: `src/configs/test/test.xx_yy-->zz.local.yaml` — identical to existing test configs but with `use_hpc: False` and minimal steps (`oc_max_steps: 20`, `*_nmt_max_steps: 200`). This config enables running pipeline steps locally for fast verification.

### 3a. `src/Pipeline/Pipeline/tests/test_pipeline_steps.py` *(new file)*

Each test function:
1. Cleans the output directory
2. Runs `python src/Pipeline/Pipeline/pipeline.py -c <local_config> --pipeline <step(s)>` via subprocess
3. Asserts expected output files/directories exist with correct structure

| Test | Pipeline args | Key assertions |
|------|--------------|---------------|
| `test_baselines` | `--pipeline baselines --nmt_models simple` | `NMT/NMT_simple_*/predictions/scores.json` exists; `NMT_scores.txt` written |
| `test_prepare_oc_charlotte` | `--pipeline prepare_OC --methods charlotte` | `OC/charlotte/xx-yy/data/train.txt` exists; no val contamination |
| `test_prepare_oc_fuzz` | `--pipeline prepare_OC --methods fuzz` | `OC/fuzz/xx-yy/data/train.txt` exists |
| `test_oc_step` | `--pipeline prepare_OC OC --methods charlotte` | OC checkpoint files exist; `predictions/scores.json` exists; `words_for_inference.txt` exists |
| `test_oc_reshape` | `--pipeline prepare_OC OC OC_reshape --methods charlotte` | Reshaped files `train.xx.txt.<model_id>` exist for train/val/test |
| `test_oc_nmt` | `--pipeline prepare_OC OC OC_reshape OC_NMT --methods charlotte --nmt_models parent` | `NMT/OC_NMT_parent_*/predictions/scores.json` exists |
| `test_full_pipeline` | `--pipeline baselines prepare_OC OC OC_reshape OC_NMT --methods charlotte --nmt_models parent simple` | `NMT_scores.txt` exists; contains lines for both `NMT_simple` and `charlotte_NMT_parent` |
| `test_method_filter` | `--pipeline prepare_OC --methods charlotte` vs `--methods fuzz` | Only the specified method directory is created |
| `test_include_pairs_filter` | `--pipeline prepare_OC --include_pairs xx,yy,zz` | Only `xx-yy` cognate data is created; other pairs skipped |

---

## Tier 4: Full Reproducibility (Already Addressed)

`src/NMT/train/tests/test_train_jobs.py` covers NMT training reproducibility for simple, parent, and child models. No new work needed here.

What's **not yet covered** and would be a natural extension:
- Full pipeline run twice → identical `NMT_scores.txt` (requires tiers 1–3 all passing first)
- OC training reproducibility (covered by Tier 2a above)

---

## Recommended Implementation Order

1. **First**: Fix the four bugs in `pipeline.py` (listed above) — they will block Tier 3 tests
2. **Tier 1** — fast to write, catches silent logic errors early, no external dependencies
   - `test_reshape.py` first (highest risk, core hypothesis)
   - `test_train_val_split.py` (non-contamination guarantee)
   - `test_cognates.py`
   - fill `test_metrics.py`
3. **Create `test.xx_yy-->zz.local.yaml`** (needed for Tiers 2 and 3)
4. **Tier 2** — `test_oc_cycle.py` first (unblocks Tier 3 OC steps), then `test_prepare_oc_data.py`
5. **Tier 3** — `test_pipeline_steps.py`, starting with `test_baselines` and `test_prepare_oc_charlotte`

---

## Verification

After all tiers are implemented:
```bash
# Run all fast unit tests
pytest src/OC/extract_cognates/tests/ src/OC/reshape/tests/ src/utilities/tests/test_metrics.py -v

# Run OC cycle integration test (CPU, ~2 min)
pytest src/OC/train/tests/test_oc_cycle.py -v

# Run pipeline CLI tests (local mode, ~30 min total)
python src/Pipeline/Pipeline/tests/test_pipeline_steps.py --tests test_baselines test_prepare_oc_charlotte test_oc_step test_full_pipeline
```
