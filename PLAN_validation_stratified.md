# Plan: NLD-Stratified Validation Set Selection

## Context

`find_val_set()` currently selects the top-`size` cognate pairs from the intersection of all cognate files, sorted by frequency (highest-quality parallel source first). This produces a val set heavily skewed toward exact matches (NLD=0.0) because exact matches tend to be high-frequency words. The fix is stratified sampling across NLD buckets to ensure a representative distribution of orthographic similarity.

## File to Modify

`src/OC/extract_cognates/validation.py` — only `find_val_set()`

## Implementation

Replace the val selection block (current lines 24–30):

```python
cognates_in_common.sort(reverse=True)
val = cognates_in_common[:size]
```

With stratified sampling logic:

```python
cognates_in_common.sort(reverse=True)  # frequency-descending within each bucket

# --- Stratified sampling by NLD ---
# NLD is the last element of each entry
exact    = [x for x in cognates_in_common if x[-1] == 0.0]
non_zero = [x for x in cognates_in_common if x[-1] >  0.0]

# Bucket 0 quota: size // 11
quota_exact = size // 11

# Buckets 1–10: 10 equal-width bins over (0, 1.0], proportional allocation
remaining_quota = size - quota_exact
buckets = []
for i in range(10):
    lo, hi = i * 0.1, (i + 1) * 0.1
    bucket = [x for x in non_zero if lo < x[-1] <= hi]
    buckets.append(bucket)

total_non_zero = len(non_zero)
val = exact[:quota_exact]
for bucket in buckets:
    if total_non_zero == 0:
        break
    quota = round(len(bucket) / total_non_zero * remaining_quota)
    val.extend(bucket[:quota])
```

Notes:
- `cognates_in_common` is already sorted frequency-descending before bucketing, so `bucket[:quota]` naturally picks the highest-frequency pairs within each NLD range.
- `round()` rather than `floor()` keeps the total closest to `size` (may be off by ±a few due to rounding across 10 buckets — acceptable).
- No redistribution of leftovers; the val set may be 1–5 items short of `size` in edge cases, which is fine.
- The rest of `find_val_set()` (train split, writing outputs) is unchanged.

## Verification

Run the existing test script:
```bash
bash src/OC/extract_cognates/validation.sh
```
Then inspect the output file to confirm NLD distribution is spread across buckets rather than concentrated at 0.0. The `.all_in_common` file can serve as a reference for the full candidate pool.
