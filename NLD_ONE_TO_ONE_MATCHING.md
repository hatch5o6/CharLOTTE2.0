# One-to-One Cognate Matching in CognatesFromNLD.py

`src/OC/extract_cognates/CognatesFromNLD.py` currently allows a src_word or tgt_word to appear in multiple cognate pairs. These notes cover how to enforce one-to-one matching (each word appears in at most one pair) and how to resolve conflicts.

---

## The Problem

After filtering by `theta`, multiple src_words may match the same tgt_word (and vice versa). A decision rule is needed for cases like:

- NLD(A, B) is the best match for A
- NLD(C, B) is the best match for C
- → A and C are competing for B

---

## Option 1: Greedy (recommended)

Sort all candidate pairs by NLD ascending, then greedily assign each pair only if neither word has been claimed yet. Conflicts resolve naturally: the lower-NLD pair wins, and the loser must find its next-best partner (or goes unmatched).

Plugging into the `np.where` approach from `NLD_EFFICIENCY_NOTES.md`:

```python
src_idxs, tgt_idxs = np.where(matrix <= theta)
distances = matrix[src_idxs, tgt_idxs]

order = np.argsort(distances)  # C-level sort
src_idxs, tgt_idxs, distances = src_idxs[order], tgt_idxs[order], distances[order]

used_src, used_tgt = set(), set()
cognate_list = []
for s, t, d in zip(src_idxs, tgt_idxs, distances):
    if s not in used_src and t not in used_tgt:
        cognate_list.append((src_cts[s], tgt_cts[t], src_words[s], tgt_words[t], d))
        used_src.add(s)
        used_tgt.add(t)
```

Fast — `np.argsort` runs in C; the Python loop only iterates over pairs that passed the `theta` filter, which is typically much smaller than N×M.

---

## Option 2: Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`)

Finds the globally optimal one-to-one assignment (minimizes total NLD across all pairs). Not practical here:

- O(N³) time — infeasible for vocabularies of 20K+ words
- Assigns every src to some tgt regardless of `theta` (requires post-filtering)

---

## Why Greedy is the Right Call

Greedy is appropriate for cognate extraction because:

1. You want *confident* cognates, not full coverage — greedy naturally prioritizes the lowest-NLD (most confident) pairs first.
2. The global optimum from Hungarian isn't meaningful in this context; you're not solving a transportation problem.
3. Edge cases where greedy is suboptimal (C loses B and has no good fallback) are rare with real vocabulary data and don't justify O(N³).
