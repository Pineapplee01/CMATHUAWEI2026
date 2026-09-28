# Reference Assets

The contest attachments are bundled under `../data/`. The project keeps only
README placeholders for assets that are required by execution but excluded from
this distribution:

| Category | Directory | Required by |
| --- | --- | --- |
| Pretrained models | `../reference/models/` | Problems 1-3 |
| Feature extraction tools | `../reference/tools/` | Problem 1 |
| Trained checkpoints | `../reference/checkpoints/` | Problems 2-3 |

To store these assets outside the distribution, set
`CPMCM_HUAWEI2026_REFERENCE_ROOT` to a directory containing the same three
subdirectories. `reference/` is never a source of contest data and `results/`
is never a source of model or tool files.
