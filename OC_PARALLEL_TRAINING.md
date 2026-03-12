# OC Parallel Training — Design Notes

## Problem

`pipeline.py` needs to train one OC model per parent→child language pair. With multiple pairs, sequential training is slow. The goal is to support parallelism in a way that:
- Works on an HPC cluster (SLURM)
- Works on a local workstation (no SLURM)
- Keeps `pipeline.py` a reusable library (not SLURM-specific)

## Recommended Approach: `submitit` + Executor Pattern

### `submitit`

[`submitit`](https://github.com/facebookincubator/submitit) (Meta/FAIR) provides an `AutoExecutor` that runs the same Python code either as local subprocesses or as SLURM jobs, depending on the environment. No sbatch script generation required.

```python
import submitit

executor = submitit.AutoExecutor(folder=logs_dir)
# On SLURM, configure partition/resources:
executor.update_parameters(
    timeout_min=720,
    slurm_partition="gpu",
    gpus_per_node=1,
    slurm_qos=config.get("oc_qos"),
)
jobs = [executor.submit(train_model, cfg) for cfg in per_pair_configs]
results = [job.result() for job in jobs]  # blocks until all finish
```

When not on SLURM, `AutoExecutor` falls back to `LocalExecutor` (local subprocesses). The calling code is identical in both cases.

Add to `pyproject.toml`:
```toml
"submitit>=1.5.0",
```

### Executor Pattern in `pipeline.main()`

Make the executor an optional parameter so `pipeline.py` stays executor-agnostic:

```python
def main(config, executor=None):
    ...
    per_pair_configs = build_training_configs(base_config, oc_data, train_dirs)
    if executor is None:
        # Default: sequential (no extra dependencies required)
        for cfg in per_pair_configs:
            train_model(cfg)
    else:
        jobs = [executor.submit(train_model, cfg) for cfg in per_pair_configs]
        [job.result() for job in jobs]
```

Users on SLURM pass a configured `submitit.AutoExecutor`. Users on a workstation can pass a `concurrent.futures.ProcessPoolExecutor` (stdlib). Users who don't care get sequential execution by default.

## Required Refactor: Separate Data Prep from Training

Currently `main()` blends cognate extraction with training. Split it into two stages:

**Stage 1** (data prep — already implemented):
Extract cognates → split → write files → populate `oc_data`.

**Stage 2** (new):
Build one config per language pair and submit to executor.

```python
def build_training_configs(base_config, oc_data, train_dirs):
    configs = []
    for (src_lang, tgt_lang), (train_path, val_path) in oc_data.items():
        cfg = base_config.copy()
        cfg["oc_train"] = train_path
        cfg["oc_val"] = val_path
        cfg["save"] = train_dirs[(src_lang, tgt_lang)]["checkpoints"]
        # ... other per-pair overrides
        configs.append(cfg)
    return configs
```

## CUDA + Subprocesses Caveat

PyTorch/CUDA requires `spawn` (not `fork`) when launching GPU training in subprocesses. `submitit` handles this correctly. If using `ProcessPoolExecutor` directly:

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
executor = ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"))
```

With sequential execution (default) there is no issue.

## Summary

| Approach | Local | SLURM | Extra deps |
|----------|-------|-------|------------|
| Sequential (default) | Yes | Yes (slow) | None |
| `ProcessPoolExecutor` | Yes | No | None (stdlib) |
| `submitit.AutoExecutor` | Yes | Yes | `submitit` |

The recommended path: add `submitit`, default `main()` to sequential, let callers inject an executor for parallelism.
