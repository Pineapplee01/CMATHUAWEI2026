# Task 2: Downstream NPZ protocol repair

## Outcome

Added `e_emotion.evaluation.protocol_guard`, a reusable strict comparison boundary
that delegates NPZ schema validation to `e_emotion.data` and adds downstream audit
checks. New runs cannot silently use historical PKL data or write artifacts outside
their method-owned directory.

## Guard behavior

- Rejects `.pkl` inputs and single-split inputs; only a processed directory containing
  `train.npz`, `valid.npz`, and `test.npz` is accepted.
- Reuses `load_processed_dataset` for required fields, dtypes, dimensions, masks,
  labels, finite values, and cross-split ID uniqueness.
- Optionally verifies expected split sizes and expected SHA-256 values, and records
  canonical source paths, hashes, sizes, schema, and sample coverage.
- Requires normalization provenance and a 64-character mask manifest SHA-256.
- Records experiment/mask seeds, checkpoint and threshold provenance, and explicit
  native/synthetic/observed mask semantics.
- Requires checkpoint, threshold, and arbitrary artifact paths to remain below the
  method's artifact root, with the artifact root itself below the method root.
- Validates method/artifact roots and all provenance files as existing canonical
  absolute paths, requiring checkpoint and threshold provenance to be present.
- Requires integer experiment and mask seeds, matching normalization/provenance
  records, exact canonical mask semantics, and the exact 13-field schema.
- Requires every split source path to be an existing canonical `.npz` regular file
  and always recomputes its SHA-256 before accepting the manifest.
- Provides `build_downstream_manifest` and `validate_downstream_manifest` plus the
  compatibility aliases exported from `e_emotion.evaluation`.

## Registry and contract

`docs/data/contract.md` now describes the strict NPZ boundary, PKL rejection,
provenance requirements, and method-owned artifact rule. `configs/baselines.yaml`
adds the guard registry and `strict_npz_status: not_ready` to current reference and
planned method entries until their server entrypoints emit complete provenance.
Historical run directories and labels are not renamed or overwritten.

## Tests

- `python -m pytest tests/test_protocol_guard.py -q` -> **32 passed**
- `python -m pytest -q` -> **91 passed**

Focused rejection coverage includes legacy PKL input, mismatched hashes, wrong split
sizes, missing/invalid provenance fields, malformed mask semantics, relative or
nonexistent roots/files, external artifact roots, and incomplete manifests.

## Concerns / follow-up

The guard validates provenance and paths but does not create artifact directories or
write manifests; callers should persist the returned JSON only after their run
directory has been selected. Existing historical runs remain outside this strict
boundary by design.
