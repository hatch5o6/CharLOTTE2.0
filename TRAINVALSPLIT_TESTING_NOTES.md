# Testing Notes: `get_train_val_split()`

`src/OC/train/TrainValSplit.py`

## What it does

Given a list of cognate-pair tuples — 4-tuples `(freq, word1, word2, nld)` from
parallel data, or 5-tuples `(freq1, freq2, word1, word2, nld)` from
monolingual/fuzzy data — builds a `(train, val)` split:

1. **Validate input** — all pairs are tuples, all the same length, length is 4 or 5.
2. **Dedupe** — sort by NLD ascending, then drop any pair whose source or
   target word has already appeared (`_ensure_unique_words`), keeping the
   first (lowest-NLD) occurrence.
3. **Re-sort by quality** — 4-tuples by raw frequency (`_sort_by_pair_freq`),
   5-tuples by geometric mean of the two frequencies (`_sort_by_geo_freq`),
   both descending.
4. **Bucket by NLD** — split `[0, theta]` into `n_buckets` equal-width
   buckets; top bucket is inclusive of `theta`.
5. **Compute val quotas per bucket** — start at `size // n_buckets` per
   bucket. If a bucket is too small (`quota >= max_fraction * len(bucket)`),
   cap it at `max_fraction * len(bucket)` and add the shortfall to a
   `deficit`. Redistribute the deficit round-robin to uncapped buckets
   (which can also become capped along the way) until the deficit is paid
   off or every bucket is capped.
6. **Split** — within each bucket, the top `quota` items (highest frequency)
   go to `val`; the rest go to `train`.
7. **Shuffle** `train` and `val` independently (seeded), return `(train, val)`.

Net effect: `val` is stratified across NLD buckets, biased toward the
highest-frequency pairs in each bucket, capped so no bucket contributes more
than `max_fraction` of its own pairs.

## Why `len(val)` is only *approximately* `size`

`len(val) <= size` always — it can fall short for two distinct, fully
deterministic reasons (not random noise):

1. **Floor division**: `default_quota = int(size / n_buckets)`, so
   `n_buckets * default_quota <= size`. If `size` doesn't divide evenly,
   up to `n_buckets - 1` slots are lost before any capping logic runs.
2. **Capping with unresolved deficit**: if all buckets become capped before
   the deficit reaches zero, the redistribution loop exits early and the
   remaining deficit is simply dropped.

So don't test this with a fuzzy tolerance (`pytest.approx`, `abs(diff) < N`)
— the shortfall is exactly computable from the inputs.

## Testing strategy

Avoid one big golden-output test asserting the full `(train, val)` tuple —
five interacting stages (dedupe → sort → bucket → quota → shuffle) make that
brittle and hard to hand-verify. Prefer **invariant tests** plus a couple of
**exact, hand-traceable** cases.

### 1. Invariant tests (robust, low-maintenance)

Run on synthetic data, assert properties rather than exact output:

- `set(train) | set(val)` == the deduped input pairs; `set(train) & set(val) == set()`.
- `len(train) + len(val) == len(deduped pairs)`.
- No source word and no target word appears more than once across `train + val` combined (post-dedupe global uniqueness).
- Every pair's NLD falls in the bucket range implied by `int(nld / bucket_range)`.

### 2. Val-size testing (no fuzzy tolerance — compute the exact expected value)

**a. Floor-division case (no capping)**
Make every bucket large enough that no capping triggers
(`len(bucket) > default_quota / max_fraction` for all buckets). Then:
```python
expected = (size // n_buckets) * n_buckets   # <= size
assert len(val) == expected
```
E.g. `size=10, n_buckets=4` → `expected == 8`. Assert exactly `8`, not "close to 10".

**b. Capping/deficit case**
Make one or more buckets deliberately tiny so they get capped, with the
rest large enough to absorb the deficit. Hand-trace the quota loop (it's
deterministic, independent of the shuffle/seed) and assert the exact
resulting `len(val)`. Also test the case where *every* bucket is tiny, so
the deficit never fully resolves — confirm `len(val) < size` by the
specific computable amount, and that the loop terminates (`len(capped) == n_buckets`).

**c. General bound as an optional property test**
For broader random-input coverage:
```python
assert len(val) <= size
assert len(val) >= size - (n_buckets - 1)   # only valid when no bucket is capped
```
The lower bound only holds when capping doesn't happen — use large random
pair lists to keep capping unlikely, or skip that assertion when capping is
forced. Useful as a sanity net, but shouldn't replace (a)/(b) since it can
mask a quota-math bug that happens to stay within the loose bound.

### 3. Quota arithmetic in isolation

This is the most complex, error-prone part (lines 47–72). Cases to hand-pick:

- All buckets large → no capping, simple even split.
- One tiny bucket → capped, deficit redistributes round-robin starting at `b=1`.
- All buckets tiny → `len(capped) == n_buckets` reached before deficit hits 0; no infinite loop.
- A bucket with 0 items → `max_fraction * 0 == 0`, quota capped at 0 immediately, no divide-by-zero.

### 4. Determinism / seeding

- Same `seed` → identical `train`/`val` *order* across two calls.
- Different `seed` → same *membership* (same pairs split into train vs val), different order. This decouples "did shuffle work" from "did the split itself change."

### 5. Length-4 vs length-5 dispatch

Construct a case where `_sort_by_pair_freq` (4-tuple) and `_sort_by_geo_freq`
(5-tuple) would produce different val-set membership for the same
underlying frequencies, and confirm the right one fired based on input
tuple length.

### 6. Edge/error cases

- Inconsistent tuple lengths → `ValueError` (already covered).
- Invalid length (not 4 or 5) → `ValueError` (already covered).
- `size` larger than total pairs.
- NLD exactly equal to `theta` — verify it lands in the last bucket via the `bucket_index -= 1` correction.
- `n_buckets=1` — degenerates to a single bucket, should still work.

### What to avoid

The commented-out `get_train_val_split()` test currently in
`test_TrainValSplit.py` tries to assert the full `(train, val)` tuple
exactly. Replace that style with the invariant + exact-value tests above.
If you do want one fully-traceable example, assert `train`/`val` membership
as **sets**, not ordered lists — shuffling makes order non-deterministic by
design even though it's seeded.
