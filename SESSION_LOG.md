# Session Log — 2026-04-09

## Topics Covered

---

### 1. CSV → YAML data config cleanup

The pipeline previously read data paths from `.csv` files with headers like `src_lang,tgt_lang,src_path,tgt_path` and `pl,cl,tl,pl_tl_path,cl_tl_path`. These were replaced by a `data` key in the YAML config files:

```yaml
data:
  - - ${DATA_HOME}/data/CharLOTTE_data/
    - es  # pl
    - an  # cl
    - en  # tl
```

A scan of the repo found residual CSV references in:
- `src/Pipeline/utilities/read_data.py` — old CSV reader (dead code, not imported)
- `src/NMT/data/toy/multi_rom/train.csv` and `src/NMT/data/toy/es-an_en/train.csv` — old toy data files
- `src/NMT/train/BARTLightning.py:325-326` — `__main__` test block (NMT future work)
- `PLAN.md`, `CLAUDE.md`, `MEMORY.md` — documentation references to `read_data_csv.py` and `pl_cl_para`/`pl_cl_mono` config keys

**Actions taken:** Updated `PLAN.md`, `CLAUDE.md`, and `MEMORY.md` to remove references to the old CSV approach and replace with the `data` key. Code files and NMT toy data were left alone.

---

### 2. `pipeline.py` NameError fix

Running `pipeline.py` crashed with:
```
NameError: name 'function' is not defined
```
`function` is not a valid Python type annotation. The parameter annotation `extract_cognates: function` was identified as the cause. Decision: remove the annotation entirely (simplest fix). The edit was proposed but the user interrupted before it was applied.

---

### 3. Eflomal non-determinism investigation

The user noticed that repeated runs of the `charlotte` cognate extraction method produced different alignments, word pairs, and cognate lists each time.

**Finding:** The source of randomness is entirely eflomal, not the user's code. Eflomal is a Gibbs sampler (MCMC-based aligner) that runs `n_samplers=3` independent random chains by default. The Python/Cython API exposes no `seed` parameter — neither `Aligner.__init__`, `Aligner.align`, nor the underlying `eflomal.align` Cython function accept one.

Everything downstream of eflomal in `CognatesFromParallel.py` is deterministic.

**Options discussed:**
1. Cache and reuse `.fwd`/`.rev` files (already written as debug output)
2. `n_samplers=1` — reduces variance but doesn't eliminate randomness
3. Use eflomal CLI directly — the C binary may accept `--seed`

**Resolution:** The user replaced eflomal with fast_align, which is deterministic.

---

### 4. Switching `CognatesFromParallel.py` from eflomal to fast_align

The user edited `CognatesFromParallel.py` to use fast_align directly (forward + reverse runs + atools symmetrization) instead of eflomal. The new `fast_align()` function was reviewed. Two bugs were identified:

- Missing `capture_output=True` and `text=True` on all three subprocess calls — `result.stdout` would be `None`
- Also missing `check=True`

The user fixed these. On re-review, the code looked correct.

Remaining dead code noted: `from eflomal import Aligner` and `aligner = Aligner()` at the top (only used by the now-unused `eflomal_align` function).

---

### 5. Three-method cognate extraction — planning session

**Context:** The project compares three cognate extraction methods:
- `charlotte` — fast_align on existing PL↔CL parallel data (`CognatesFromParallel.py`) — implemented
- `web` (With Engineered Bitext) — trains TL→PL NMT, translates CL↔TL corpus to get synthetic PL↔CL corpus, then runs fast_align (`CognatesFromSyntheticParallel.py`) — stub, blocked on NMT module
- `fuzz` — monolingual; finds best NLD match per word using full distance matrix (`CognatesFromNLD.py`) — implemented

**Key design decisions made:**

- **Shared post-processing (`cognate_filters.py`):** All three methods should pass candidates through a shared filter: short-word filter (min length 4), number handling (identity pairs), NLD threshold.
- **Number handling:** If both words in a pair are purely numeric, discard the original pair and add identity pairs for each number, inheriting the original count.
- **Minimum word length:** 4 characters.
- **Output format:**
  - charlotte/web: `(ct, word1, word2, nld)` — 4-column
  - fuzz: `(src_ct, tgt_ct, word1, word2, nld)` — 5-column (preserved for `validation.py` compatibility)
- **Shared val set:** Val set drawn from `charlotte` cognate pairs (or `web` if no parallel data). All OC models — regardless of extraction method — are evaluated on the same gold-standard val set. `validation.py`'s `find_val_set` already handles stratified sampling; it needs a `val_source_idx` parameter so it draws from charlotte/web rather than the intersection of all methods.
- **`split()` in `utilities.py` is no longer used** for val-set selection.
- **WEB method:** Trains a TL→PL NMT model (using `BARTLightning.py`) and translates the TL side of the CL↔TL corpus to produce synthetic PL↔CL parallel data. Blocked on NMT module.

**`PLAN.md` was updated** to reflect all three methods, the shared val set approach, new config key `oc_val_size`, and the `cognate_filters.py` module.

---

### 6. Aborted implementation

After the plan was approved via ExitPlanMode, implementation began automatically (misunderstanding — user thought ExitPlanMode only approved the plan document). Three files were modified before the user said stop:

- `src/OC/extract_cognates/cognate_filters.py` — created (new file)
- `src/OC/extract_cognates/CognatesFromParallel.py` — rewritten
- `src/OC/extract_cognates/CognatesFromNLD.py` — rewritten

All three changes were reverted from conversation history. The diffs and full content of `cognate_filters.py` are recorded in `ABORTED_CHANGES.md` in the project root.

**Memory saved:** Do not write code after ExitPlanMode approval — wait for an explicit instruction to proceed.

---

### 7. Session housekeeping

- Stale task list (tasks marked completed/in-progress during the aborted run) was cleared.
- `ABORTED_CHANGES.md` written to project root with diffs of the aborted changes.
- This session log written to `SESSION_LOG.md`.
