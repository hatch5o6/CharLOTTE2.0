# Aborted Implementation — Cognate Extraction Modularization

Changes were written to code files and then reverted. This file records what was done
so the implementation can be resumed deliberately.

---

## Files affected

| File | Change |
|------|--------|
| `src/OC/extract_cognates/cognate_filters.py` | **Created** (new file — deleted on revert) |
| `src/OC/extract_cognates/CognatesFromParallel.py` | **Modified** |
| `src/OC/extract_cognates/CognatesFromNLD.py` | **Modified** |

---

## `cognate_filters.py` (new file, full content)

```python
from collections import Counter

from OC.utilities.utilities import NLD


def finalize_parallel(candidates, theta, min_len=4):
    """Filter and format cognate candidates from parallel alignment methods (charlotte, web).

    candidates: [(ct, word1, word2), ...]
    returns:    [(ct, word1, word2, nld), ...] sorted by nld ascending
    """
    result, identity_pairs = _filter(candidates, theta, min_len, monolingual=False)
    for (w, _), ct in identity_pairs.items():
        result.append((ct, w, w, 0.0))
    result.sort(key=lambda x: x[-1])
    return result


def finalize_monolingual(candidates, theta, min_len=4):
    """Filter and format cognate candidates from monolingual methods (fuzz).

    candidates: [(src_ct, tgt_ct, word1, word2), ...]
    returns:    [(src_ct, tgt_ct, word1, word2, nld), ...] sorted by nld ascending
    """
    result, identity_pairs = _filter(candidates, theta, min_len, monolingual=True)
    for (w, _), ct in identity_pairs.items():
        result.append((ct, ct, w, w, 0.0))
    result.sort(key=lambda x: x[-1])
    return result


def _filter(candidates, theta, min_len, monolingual):
    """Shared filtering logic for both formats.

    Returns (result_list, identity_counter) where identity_counter accumulates
    numeric identity pairs keyed by (word, word).
    """
    result = []
    identity_pairs = Counter()

    for entry in candidates:
        if monolingual:
            src_ct, tgt_ct, word1, word2 = entry
            ct = src_ct
        else:
            ct, word1, word2 = entry

        # Short-word filter
        if len(word1) < min_len or len(word2) < min_len:
            continue

        # Number handling: both purely numeric → collect identity pairs, skip original
        if word1.isdigit() and word2.isdigit():
            identity_pairs[(word1, word1)] += ct
            identity_pairs[(word2, word2)] += ct
            continue

        # NLD filter
        nld = NLD(word1, word2)
        if nld > theta:
            continue

        if monolingual:
            result.append((src_ct, tgt_ct, word1, word2, nld))
        else:
            result.append((ct, word1, word2, nld))

    return result, identity_pairs
```

---

## `CognatesFromParallel.py` diff

```diff
-from tempfile import NamedTemporaryFile
-from sloth_hatch.sloth import read_lines, read_content, write_content, write_lines, log_function_call
-from eflomal import Aligner
+from sloth_hatch.sloth import read_lines, write_lines, log_function_call
 
 from OC.utilities.word_tokenizers import get_tokenizer
 from OC.utilities.word_preprocessing import clean, is_only_punct
-from OC.utilities.utilities import NLD
+from OC.extract_cognates.cognate_filters import finalize_parallel
 
-# (no aligner = Aligner() — already removed by user before this edit)
 
 @log_function_call
 def extract_cognates(...):
     ...
-    word_pairs = sort_word_pairs(get_word_pairs(sent_pairs, alignments))
-    write_lines([str(w) for w in word_pairs], cognate_list_out + ".word_pairs")
-    cognates = get_cognates(word_pairs, theta=theta)
-    write_lines([str(c) for c in cognates], cognate_list_out + ".cognate_list")
+    word_pairs = get_word_pairs(sent_pairs, alignments)
+    write_lines([f"{ct} ||| {w1} ||| {w2}" for (w1,w2),ct in sorted(word_pairs.items(), key=lambda x: -x[1])], cognate_list_out + ".word_pairs")
+    candidates = [(ct, w1, w2) for (w1, w2), ct in word_pairs.items()]
+    cognates = finalize_parallel(candidates, theta=theta)
+    from OC.utilities.utilities import write_oc_data
+    write_oc_data(cognates, cognate_list_out + ".cognate_list")
     return cognates
 
-def eflomal_align(...):   # removed (dead code)
-    ...
 
 # get_word_pairs: NBSP check changed from literal NBSP char to explicit \u00a0,
 # and " " not in word_pair_x changed to substring check on each element.
 # (Behavior equivalent — see note below.)
 
-def sort_word_pairs(...):  # removed — replaced by finalize_parallel
-def get_cognates(...):     # removed — replaced by finalize_parallel
-def are_cognates(...):     # removed — replaced by finalize_parallel
```

**Note on NBSP change:** The original code used a literal non-breaking space character
`\u00a0` inside string literals that rendered as `" "`. The rewrite made this explicit.
Behavior is equivalent. On revert, the literal character was preserved from conversation
history, so this is a non-issue.

---

## `CognatesFromNLD.py` diff

```diff
-from sloth_hatch.sloth import read_lines, read_content, write_content, write_lines, log_function_call
+from sloth_hatch.sloth import read_lines, write_lines, log_function_call
 
 from OC.utilities.word_tokenizers import get_tokenizer
 from OC.utilities.word_preprocessing import clean, is_only_punct
 from OC.utilities.utilities import write_oc_data
+from OC.extract_cognates.cognate_filters import finalize_monolingual
 
 def extract_cognates(...):
     ...
-    cognate_list = set()
-    used_src = set()
-    used_tgt = set()
-    for dist, s, t in dists:
-        ...
-        if src_word not in used_src and tgt_word not in used_tgt:
-            if dist <= theta:
-                cognate_list.add((src_ct, tgt_ct, src_word, tgt_word, dist))
-                used_src.add(src_word)
-                used_tgt.add(tgt_word)
-    cognate_list = sorted(cognate_list, key=lambda x: x[4])
-    write_oc_data(cognate_list, cognate_list_out)
-    return cognate_list
+    used_src = set()
+    used_tgt = set()
+    candidates = []
+    for dist, s, t in dists:
+        ...
+        if src_word not in used_src and tgt_word not in used_tgt:
+            if dist <= theta:
+                candidates.append((src_cts[s], tgt_cts[t], src_words[s], tgt_words[t]))
+                used_src.add(src_word)
+                used_tgt.add(tgt_word)
+    cognates = finalize_monolingual(candidates, theta=theta)
+    write_oc_data(cognates, cognate_list_out)
+    return cognates
```

The logic change is minimal: `cognate_list` set replaced by `candidates` list feeding
into `finalize_monolingual`. The greedy 1-to-1 matching loop itself is unchanged.
NLD is recomputed inside `finalize_monolingual` (negligible overhead).
