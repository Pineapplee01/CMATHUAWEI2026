# AUMDF NPZ Boundary Repair

Date: 2026-09-24

## Contract

- New AUMDF training and checkpoint evaluation default to `strict_data=True`.
- Strict loading accepts only the exact canonical Appendix 2 processed root:
  `/user_home/gaojianan/CPMCM/AAAdata/Appendix_2/标准化/对齐版本/processed`.
- The root must contain `train.npz`, `valid.npz`, and `test.npz`.
- The known metadata files `scaler_params.npz`, `preprocess_report.json`,
  `model_input_contract.json`, `bert_encode_report.json`, and the upstream
  backup `XT_raw_before_zscore_train.npz` are explicitly allowed alongside the
  three split files.
- Strict loading rejects legacy `.pkl`, arbitrary processed directories, and a
  single split `.npz` path before attempting to deserialize or train.
- Historical test fixtures retain compatibility only through explicit
  `strict=False` / `strict_data=False` arguments.

## Regression Coverage

`references/AUMDF/tests/test_data_metrics.py` covers legacy pickle rejection,
arbitrary processed-root rejection (including a complete set of split files),
single-split rejection, strict default behavior, and canonical metadata/
upstream-backup acceptance.

## Verification

- `conda run -n base python -m pytest references/AUMDF/tests -q`: **23 passed**
- `conda run -n base python -m pytest tests/test_protocol_guard.py tests/test_q2_masks.py -q`: **43 passed**
- `conda run -n base python -m compileall -q references/AUMDF/aumdf references/AUMDF/run.py`: **passed**
- `git diff --check`: **passed**

The real canonical data root was not available in this workspace, so the
acceptance test uses a schema-valid temporary canonical root containing all
listed metadata files and the upstream backup filename.
