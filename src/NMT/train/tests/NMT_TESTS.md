# NMT Test Plan

Test files live in `src/NMT/train/tests/`. All tests use pytest `tmp_path` for file I/O — no dependency on external data. Toy sentence fixtures should be written inline as small lists of strings.

---

## test_NMTTokenizer.py

### `train_unigram`

| Test | What it checks |
|------|----------------|
| `test_train_unigram_saves_to_expected_path` | Return value is `{save}/UnigramTokenizer`; directory exists after training |
| `test_train_unigram_loadable` | Tokenizer saved by `train_unigram` can be loaded back via `load_tokenizer` without error |
| `test_train_unigram_vocab_size` | `len(tokenizer) == vocab_size` after loading |
| `test_train_unigram_special_tokens_present` | pad, unk, bos, eos tokens are all in the vocabulary |
| `test_train_unigram_lang_toks` | Language tokens passed via `lang_toks` appear in the vocabulary |
| `test_train_unigram_missing_save_dir` | `FileNotFoundError` raised when `save` directory does not exist |
| `test_train_unigram_determinism` | Train twice on identical files with identical params; compare saved `tokenizer.json` files byte-for-byte. **This test confirms whether UnigramTrainer is deterministic.** If files differ, training is non-deterministic and results cannot be reproduced. |

### `_dedupe`

| Test | What it checks |
|------|----------------|
| `test_dedupe_removes_duplicates` | List with duplicate lines → output has no duplicates |
| `test_dedupe_preserves_order` | First occurrence of each line is preserved; relative order maintained |
| `test_dedupe_empty` | Empty list → empty list |
| `test_dedupe_no_duplicates` | List with no duplicates → unchanged |

### `_upsample`

| Test | What it checks |
|------|----------------|
| `test_upsample_reaches_quota` | Output length equals `quota` |
| `test_upsample_quota_larger_than_data` | Works correctly when `quota` is many multiples of `len(data)` |
| `test_upsample_exact_multiple` | Works when `quota` is an exact multiple of `len(data)` |

### `_get_share_size`

| Test | What it checks |
|------|----------------|
| `test_get_share_size_equal_ratios` | Equal ratios → `training_data_size // n_langs` |
| `test_get_share_size_unequal_ratios` | Unequal ratios → floor division by total shares |

### `_validate_data_ratios`

| Test | What it checks |
|------|----------------|
| `test_validate_data_ratios_valid` | Well-formed dict passes |
| `test_validate_data_ratios_not_a_dict` | Non-dict input → `False` |
| `test_validate_data_ratios_missing_files_key` | Item missing `files` key → `False` |
| `test_validate_data_ratios_missing_ratio_key` | Item missing `ratio` key → `False` |
| `test_validate_data_ratios_ratio_not_int` | `ratio` is a float → `False` |
| `test_validate_data_ratios_files_not_list` | `files` is a string → `False` |
| `test_validate_data_ratios_files_item_not_string` | `files` contains a non-string → `False` |

### `_make_scenario_data`

| Test | What it checks |
|------|----------------|
| `test_make_scenario_data_downsamples` | When corpus is larger than quota, output file has exactly `quota` lines |
| `test_make_scenario_data_upsamples` | When corpus is smaller than quota, output file has exactly `quota` lines |
| `test_make_scenario_data_writes_notes_json` | `notes.json` is written to the save directory |
| `test_make_scenario_data_returns_file_paths` | Return value is a list of file paths that all exist |
| `test_make_scenario_data_total_lines` | Sum of all output file line counts equals sum of all quotas |

### `make_tokenizer_data`

| Test | What it checks |
|------|----------------|
| `test_make_tokenizer_data_creates_directory_structure` | `tokenizer_dir/`, `tokenizer_dir/data/`, and per-scenario subdirs are all created |
| `test_make_tokenizer_data_returns_correct_scenario_keys` | Returned dict has the correct `(pl, cl, tl)` tuple keys |
| `test_make_tokenizer_data_fails_if_dir_exists` | `AssertionError` raised if `tokenizer_dir` already exists |
| `test_make_tokenizer_data_files_exist` | All file paths in the returned dict exist on disk |

---

## test_ParallelDatasets.py

**Fixture strategy:** write small toy `.txt` files to `tmp_path` (e.g. 5 sentence pairs each for pl-tl and cl-tl directories).

### `CharLOTTEParallelDataset`

| Test | What it checks |
|------|----------------|
| `test_dataset_parent_mode_paths` | `mode="parent"` reads from the `{pl}-{tl}` directory |
| `test_dataset_child_mode_paths` | `mode="child"` reads from the `{cl}-{tl}` directory |
| `test_dataset_oc_mode_paths` | `mode="oc"` reads from the `{pl}-{cl}` directory |
| `test_dataset_len` | `len(dataset)` matches the number of sentence pairs in the files |
| `test_dataset_getitem_keys` | Each item has keys `src`, `tgt`, `src_lang`, `tgt_lang`, `src_path`, `tgt_path` |
| `test_dataset_getitem_content` | `src` and `tgt` strings match the corresponding lines in the files |
| `test_dataset_reverse` | `reverse=True` swaps `src`/`tgt` content and `src_lang`/`tgt_lang` |
| `test_dataset_sc_model_ids_appends_suffix` | With `sc_model_ids` set, `src_path` ends with the model ID suffix |
| `test_dataset_sc_model_ids_wrong_mode` | `sc_model_ids` with `mode != "parent"` raises `ValueError` |
| `test_dataset_sc_model_ids_missing_scenario` | Scenario not in `sc_model_ids` raises `ValueError` |
| `test_dataset_mismatched_lengths` | src and tgt files with different line counts raise `ValueError` |
| `test_dataset_invalid_mode` | Unknown `mode` raises `ValueError` |
| `test_dataset_invalid_div` | Unknown `div` raises `ValueError` |
| `test_dataset_invalid_datasets_not_list` | Non-list/tuple dataset item raises `ValueError` |
| `test_dataset_invalid_datasets_wrong_length` | Dataset item of length != 4 raises `ValueError` |
| `test_dataset_multiple_scenarios` | Two `[data_folder, pl, cl, tl]` entries → items from both are concatenated |

---

## test_BARTLightning.py

**Fixture strategy:** use a tiny trained `PreTrainedTokenizerFast` (or a mock tokenizer) and a minimal config dict. Keep everything CPU-only.

### `BARTDataModule.collate_fn`

| Test | What it checks |
|------|----------------|
| `test_collate_fn_output_keys` | Returns dict with `source_ids`, `target_ids`, `source_lens`, `target_lens`, `source_lang_ids`, `target_lang_ids` |
| `test_collate_fn_pads_source_to_longest` | Shorter source sequences are padded with `pad_token_id` to match the longest |
| `test_collate_fn_pads_target_with_minus100` | Target padding uses `-100`, not `pad_token_id` |
| `test_collate_fn_eos_appended` | EOS token is the last non-padding token in each source and target sequence |
| `test_collate_fn_truncates_source` | Source tokens beyond `max_length - seq_buffer` are truncated |
| `test_collate_fn_truncates_target` | Target tokens beyond `max_length - seq_buffer` are truncated |
| `test_collate_fn_no_lang_tags_bilingual` | `append_lang_tags=False` → `source_lang_ids` and `target_lang_ids` are tensors of `None` |
| `test_collate_fn_lang_tags_multilingual` | `append_lang_tags=True` → first token of each sequence is the language tag token id |
| `test_collate_fn_max_length_respected_with_lang_tags` | With lang tags, buffer is 2 not 1; total length ≤ `max_length` |

### `BARTDataModule`

| Test | What it checks |
|------|----------------|
| `test_datamodule_invalid_mode` | Unknown `mode` raises `ValueError` |
| `test_datamodule_invalid_data_not_list` | Data item that is not a list/tuple raises `ValueError` |
| `test_datamodule_invalid_data_wrong_length` | Data item of length != 4 raises `ValueError` |
| `test_datamodule_setup_creates_datasets` | After `setup()`, `train_dataset`, `val_dataset`, and `test_dataset` are all non-None |
| `test_datamodule_train_dataloader_shuffles` | `train_dataloader()` has `shuffle=True` |
| `test_datamodule_val_dataloader_no_shuffle` | `val_dataloader()` has `shuffle=False` |

### `BARTLightning`

| Test | What it checks |
|------|----------------|
| `test_bart_model_initializes` | `BARTLightning(config, tokenizer)` constructs without error |
| `test_bart_model_config_applied` | `model.config.encoder_layers` etc. match the values passed in config |
| `test_training_step_returns_scalar_loss` | `training_step(batch, 0)` returns a scalar tensor |
| `test_validation_step_returns_scalar_loss` | `validation_step(batch, 0)` returns a scalar tensor |
| `test_predict_step_output_structure` | `predict_step(batch, 0)` returns a list of `(src_str, tgt_str, pred_str)` triples |
| `test_predict_step_no_minus100_in_target` | Decoded target strings in `predict_step` output do not contain the `<unk>` or garbage token that `-100` would decode to — confirms the masking fix is correct |
| `test_predict_step_batch_size_consistent` | Number of triples returned equals batch size |
| `test_generate_returns_token_ids` | `generate(source_ids=..., ...)` returns a 2D integer tensor |

---

## test_train.py

**Fixture strategy:** use `tmp_path`; pre-build the expected directory structure (`NMT/checkpoints/`, `NMT/predictions/`, etc.) where needed.

### Helper functions

| Test | What it checks |
|------|----------------|
| `test_get_save_dir_exists` | Returns `{save}/{experiment_name}/NMT` when the directory exists |
| `test_get_save_dir_missing` | Raises `FileNotFoundError` when the NMT directory does not exist |
| `test_write_scores_writes_json` | `_write_scores(scores, d)` writes a valid JSON file to `d/scores.json` |
| `test_write_scores_json_serializable` | Scores dict containing floats serializes without error (guards against sacrebleu object serialization bug) |
| `test_write_preds_creates_subdirs` | One subdir per checkpoint is created under `preds_d` |
| `test_write_preds_writes_val_and_test` | Each subdir contains `validation.preds.txt` and `test.preds.txt` |
| `test_write_preds_no_dir_collision` | Multiple checkpoints each get their own separate subdir |
| `test_get_preds_from_outputs` | Extracts the third element of each `(source, target, pred)` triple |

### `_get_best_val_scores`

| Test | What it checks |
|------|----------------|
| `test_get_best_val_scores_selects_highest_chrf` | Checkpoint with highest `VAL_chrF++` is selected as best |
| `test_get_best_val_scores_selects_highest_spbleu` | Checkpoint with highest `VAL_spBLEU` is selected when `use_metric="spBLEU"` |
| `test_get_best_val_scores_adds_best_key` | `BEST_VAL_chrF++` key is added to the scores dict |
| `test_get_best_val_scores_best_entry_has_all_divs` | Best entry contains `VAL_chrF++`, `VAL_spBLEU`, `TEST_chrF++`, `TEST_spBLEU` |
| `test_get_best_val_scores_assertion_on_wrong_metric` | Unknown `use_metric` raises `AssertionError` |
| `test_get_best_val_scores_no_duplicate_best_key` | Calling twice raises `AssertionError` (key already present) |

### `_get_scores`

| Test | What it checks |
|------|----------------|
| `test_get_scores_returns_correct_keys` | Returns `{div}_chrF++` and `{div}_spBLEU` keys for the given div |
| `test_get_scores_source_mismatch_raises` | `AssertionError` when decoded sources don't match `gt_source` |
| `test_get_scores_target_mismatch_raises` | `AssertionError` when decoded targets don't match `gt_target` |
| `test_get_scores_invalid_div` | `ValueError` for div other than `"VAL"` or `"TEST"` |

### `run_inference`

| Test | What it checks |
|------|----------------|
| `test_run_inference_returns_triples` | Output is a flat list of `(src_str, tgt_str, pred_str)` triples |
| `test_run_inference_length_matches_dataset` | Total number of triples equals the number of examples in the dataloader |
