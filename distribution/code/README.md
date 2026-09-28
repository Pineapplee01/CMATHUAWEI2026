# Code Bundle

This directory contains the actual code chain copied from
`G:\CMath\2026华为杯\server_full_flow_code`.

Included:

- `final_submission/`: final submission code and run notes from `AAA提交版代码及结果`.
- `preprocess/`: recovered preprocessing scripts for the Appendix 1, Appendix 2,
  and Appendix 4 data flows.
- `shared/`: shared path, data, and plotting helpers.
- `configs/`: organized server-side experiment and pipeline configurations.
- `tests/`: validation tests migrated from the organized server snapshot.
- `server_full_flow_root/`: root README, migration manifest, environment files,
  requirements files, and audit/package helper scripts.

Excluded on purpose:

- `主要结果` generated output directories from the final submission tree;
- trained checkpoint files and model weights;
- downloaded pretrained models and external tools;
- generated caches and build artifacts.

The normalized Python package remains in `../src/multimodal_emotion`. This
`code/` directory is the traceable server-code snapshot for audit and migration
purposes.
