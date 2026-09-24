# Final NPZ protocol repair verification

## Scope

- Strict downstream builders and validators now accept only the canonical processed root (or an explicitly supplied canonical root for isolated tests), exactly `train.npz`, `valid.npz`, and `test.npz`, and reject preloaded/fabricated datasets.
- The processed loader validates source-compatible organizer label encodings without truncation, requires Unicode IDs and float32 `rho_content`, and validates class integrality/range before casting internal arrays.
- Downstream manifests reload canonical files and cross-check schema, sample coverage, split sizes, source paths, file hashes, and mask hash provenance.
- Q2 manifests enforce aligned feature/native-mask versions, strict seeds/fractions, bounds/count/effective-fraction semantics, and complete-condition zero semantics.

## Verification

```text
pytest -q tests/test_protocol_guard.py tests/test_processed_data.py tests/test_q2_masks.py
...........................................................              [100%]
59 passed in 9.09s
```

```text
pytest -q
........................................................................ [ 70%]
..............................                                           [100%]
102 passed in 10.51s
```
