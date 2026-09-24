# Task 3: deterministic Q2 mask manifest

Implemented `e_emotion.robustness.q2_masks` with a method-agnostic generator,
stable JSON writer/loader, and structural/dataset-aware validator. The public
matrix is `complete,T,A,V,TA,TV,AV x beginning,middle,end x 0.1,0.3,0.5` and
uses `experiment_seed=2026` and `mask_seed=2026` by default.

Each entry records split and sample ID, native valid lengths, coordinate-domain
lengths, half-open synthetic start/end positions, requested/effective fractions,
synthetic counts, both seeds, and the native mask version. Coordinate domains
are `mT`, `mT & mA`, and `mT & mV`; native missingness remains outside the
synthetic mask. `complete` entries have zero synthetic deletions and preserve
the native mask. The hash is SHA-256 over canonical JSON excluding the hash
field itself. Output paths are supplied by the caller.

The fixture manifest used by the focused tests has hash
`1388a7c6e17640d851aed0631d6ee6e5dda2873df2105a304b8b03d66701d65c`.

Validation:

- `python -m pytest tests/test_q2_masks.py -q` -> 3 passed
- `python -m pytest -q` -> 94 passed
