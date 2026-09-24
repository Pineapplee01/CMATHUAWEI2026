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

The existing hash-only `build_downstream_manifest` arguments remain accepted;
strict content binding is activated by supplying `mask_manifest_path`, which is
the path recorded and verified in the downstream manifest.
