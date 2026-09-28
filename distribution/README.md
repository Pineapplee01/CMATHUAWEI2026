# CPMCM Huawei 2026 Distribution

This directory is the local delivery bundle for the CPMCM Huawei Cup 2026
project. It contains the normalized Python package and the actual server-side
code used for the contest workflow. The local delivery also contains the
original contest attachments provided by the problem statement, but those raw
files are intentionally excluded from the GitHub publication.

It is not a model/checkpoint bundle. Pretrained models, external feature
extractors, trained weights, generated intermediate arrays, and generated result
tables remain excluded unless a README explicitly says otherwise.

## Layout

- `src/multimodal_emotion/`: normalized standalone Python package; its
  `preprocess/` module contains the canonical preprocessing pipeline.
- `code/final_submission/`: final submission scripts and run notes copied from
  `server_full_flow_code/AAA提交版代码及结果`; generated `主要结果` outputs are
  intentionally excluded.
- `code/preprocess/`: recovered preprocessing scripts for Appendix 1, Appendix 2,
  and Appendix 4 workflows.
- `code/shared/`, `code/configs/`, `code/tests/`: shared utilities, experiment
  configurations, and validation tests from the server-side organized snapshot.
- `code/server_full_flow_root/`: root-level environment, requirements, manifest,
  packaging, and audit helper files from `server_full_flow_code`.
- `data/`: local original contest attachments copied from
  `G:\CMath\2026华为杯\E题数据`, standardized to English directory names.
  GitHub retains only `data/README.md`; restore the raw files locally before
  running any data-dependent workflow.
- `results/`: initially README-only directory for derived NPZ files, caches,
  predictions, and explanations. Scripts must never write into `data/`.
- `reference/`: README-only placeholders for model/checkpoint/tool
  dependencies that are not bundled.
- `docs/path_mapping.md`: legacy Chinese paths mapped to canonical English paths.
- `docs/archive_index.md`: historical archive index; archive filenames are not renamed.

## Reference Assets

Contest data is bundled in the local `data/` directory but excluded from GitHub
because of GitHub file-size limits. Pretrained models, trained checkpoints, and
feature-extraction tools are intentionally represented only by README files in
`reference/`. By default, code reads them from that directory. To keep
them elsewhere, point this environment variable at a directory containing
`models/`, `tools/`, and `checkpoints/`:

```bash
CPMCM_HUAWEI2026_REFERENCE_ROOT=/path/to/reference-assets
```

The local `data/` directory contains no generated `.npz` intermediates,
checkpoints, or pretrained model directories.

## Setup

Install the Python dependencies before invoking preprocessing, prediction, or
explanation modules:

```bash
python -m pip install -e ".[runtime,preprocess,test]"
```

This installs PyTorch and Transformers but does not download a pretrained model
or any external tool. Populate only the documented README placeholders that the
workflow you intend to run requires.

## Development Checks

```bash
python -m compileall src tests
python -m pytest tests
python -m pip wheel . --no-deps -w dist
```

The distribution does not define `console_scripts` and does not promise a
stable public command-line interface. Use the documented modules and scripts as
research code.
