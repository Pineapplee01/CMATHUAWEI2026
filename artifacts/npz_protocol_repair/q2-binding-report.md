# Q2 Binding Repair

## Scope

- Q2 mask manifests now bind `train`/`valid`/`test` source paths and SHA-256
  data hashes.  Existing in-memory fixtures use a deterministic content hash
  only when their synthetic source path does not exist; real NPZ inputs always
  use the source file hash.
- Canonical Q2 protocol version, native mask version, exact 7 x 3 x 3 matrix,
  and source hashes are validated before a bound downstream run is accepted.
- `mask_manifest_path` is optional for backwards-compatible builders.  When
  supplied it must be inside the method artifact root and is loaded and
  cross-checked against the claimed hash and canonical dataset.
- The processed root validator requires all three split NPZ files, permits the
  known preprocessing metadata files (`scaler_params.npz`,
  `XT_raw_before_zscore_train.npz`,
  `preprocess_report.json`, `model_input_contract.json`, and
  `bert_encode_report.json`), and rejects unknown files.
- Continuous-mask computation is cached per split/modality/position/fraction,
  preserving deterministic output while avoiding repeated work on the full
  matrix.

## Validation

- Focused: `pytest -q tests/test_q2_masks.py tests/test_protocol_guard.py`
- Full suite: `pytest -q`
- Result: **110 passed**.

## Compatibility note

Strict downstream manifests now require `mask_manifest_path` and reject a
hash-only claim.  Historical callers may explicitly pass
`legacy_compatibility=True`; such manifests are marked `mask_binding_mode` as
`legacy_hash_only` and cannot validate in strict mode.

`normalization_source` remains a provenance label for compatibility.  A
normalization parameter file hash is not required by the current builder, so
the report does not claim scaler-content verification; this remains a separate
follow-up contract item.
