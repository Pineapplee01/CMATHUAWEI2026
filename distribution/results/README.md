# Generated Artifacts

This directory is intentionally empty except for this README in a fresh
distribution. It is the only in-tree destination for derived artifacts.

| Workflow | Output directory |
| --- | --- |
| Problem 1 feature extraction | `appendix_1/` |
| Problem 2 preprocessing | `appendix_2/aligned/` or `appendix_2/unaligned/` |
| Problem 2 predictions | `appendix_2/predictions/` |
| Problem 3 explanations | `appendix_4/explanations/` |

Problem 2 preprocessing creates `train.npz`, `valid.npz`, `test.npz`,
`scaler_params.npz`, and `preprocess_report.json`. These files are derived from
the bundled contest PKL files and are intentionally not versioned.
