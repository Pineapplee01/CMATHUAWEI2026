# Appendix 2 Preprocessing

These scripts are the distribution-adapted form of the recovered complete
Appendix 2 preprocessing logic. They read the actual contest files directly:

```text
data/appendix_2/aligned_50.pkl
data/appendix_2/unaligned_50.pkl
```

Each file contains top-level `train`, `valid`, and `test` mappings. The scripts
construct Effective Axis (`P`) and Observation Mask (`O`), encode text with the
local BERT model, fit modality scalers on observed training values only, and
write the derived NPZ splits below `results/appendix_2/`.

```bash
python code/preprocess/problem2/preprocess_aligned.py
python code/preprocess/problem2/preprocess_unaligned.py
```

Use `--raw` and `--output` to override defaults. `--output` must not point into
`data/`. The external BERT checkout belongs at
`reference/models/bert-base-uncased/`.

`RECOVERY_PROVENANCE.md` records the historical server source and hashes. It is
evidence for provenance, not the current distribution path contract.
