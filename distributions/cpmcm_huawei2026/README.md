# CPMCM Huawei 2026 Code Distribution

This directory is a standalone, English-named code distribution derived from
the submission-aligned `server_full_flow_code` snapshot.

It is intentionally not a complete offline runtime bundle. Raw data, pretrained
models, external feature extractors, and checkpoints remain external assets.
See `external_assets/*/README.md` and `docs/external_assets.md`.

## Layout

- `src/cpmcm_huawei2026/preprocessing/`: feature and model-input construction.
- `src/cpmcm_huawei2026/prediction/`: Problem 2 aligned and unaligned prediction code.
- `src/cpmcm_huawei2026/explanation/`: Problem 3 explanation code.
- `src/cpmcm_huawei2026/submission/`: path mapping and manifest helpers.
- `src/cpmcm_huawei2026/shared/`: path, data, and plotting helpers.
- `docs/path_mapping.md`: legacy Chinese paths mapped to canonical English paths.
- `docs/archive_index.md`: historical archive index; archive filenames are not renamed.

## Asset Root

Runtime modules resolve `AAAdata`, `AAAmodel`, and `AAAcheckpoints` from the
environment variable below:

```bash
CPMCM_HUAWEI2026_ASSET_ROOT=/path/to/project-assets
```

If the variable is absent, paths resolve relative to this distribution root.

## Development Checks

```bash
python -m compileall src tests
python -m pytest tests
python -m pip wheel . --no-deps -w dist
```

The distribution does not define `console_scripts` and does not promise a
stable public command-line interface. Use the documented modules and scripts as
research code.
