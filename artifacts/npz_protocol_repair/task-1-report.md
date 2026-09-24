# Task 1 NPZ Protocol Repair

## Changes

- Required and validated canonical `Q`, `P`, `q`, and `rho_content` metadata, preserving them on `ProcessedSplit` as `Q`, `P`, `q`, and `rho_content`.
- Enforced `aligned_50` feature shapes and `float32` dtypes: `XT=(N,50,768)`, `XA=(N,50,74)`, `XV=(N,50,35)`.
- Rejected non-boolean native/token/sample masks instead of coercing them.
- Kept native masks separate from synthetic Q2 masks and propagated canonical metadata through `with_observed_mask`.
- Extended manifests with canonical field schema (`dtype` and dimensions), observed per-split shapes, SHA-256 hashes, and explicit sample ID coverage.
- Added regression tests for required metadata, canonical dimensions, mask dtypes, and manifest schema/coverage.

## Verification

Command:

```text
python -m pytest tests/test_processed_data.py -q
```

Output:

```text
...............                                                          [100%]
16 passed in 1.18s
```

Command:

```text
python -m pytest -q
```

Output:

```text
..........................................................               [100%]
59 passed in 1.56s
```

## Concerns

`rho_content` is validated as a three-column `[text, audio, vision]` content-relative native missing-rate vector, with audio/vision rates computed against text-native positions. No model, loss, or run code was changed.
