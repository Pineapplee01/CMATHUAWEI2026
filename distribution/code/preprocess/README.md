# Preprocessing Scripts

This directory contains distribution-adapted copies of the recovered server
preprocessing logic. The immutable server snapshot remains under the
project-level `../../../server_full_flow_code/` directory.

| Directory | Input | Output |
| --- | --- | --- |
| `problem1/` | `data/appendix_1/mosei_raw_videos_100/` | `results/appendix_1/` |
| `problem2/` | `data/appendix_2/{aligned_50,unaligned_50}.pkl` | `results/appendix_2/{aligned,unaligned}/` |
| `problem3/` | `data/appendix_4/{aligned,unaligned}/` | In-memory model batches and `results/appendix_4/` outputs |

The `data/` directory is immutable contest input. Scripts reject a Problem 2
output path inside `data/`.

```bash
python code/preprocess/problem1/method_a.py extract --config code/preprocess/problem1/config_a.json
python code/preprocess/problem1/method_b.py extract --config code/preprocess/problem1/config_b.json
python code/preprocess/problem2/preprocess_aligned.py
python code/preprocess/problem2/preprocess_unaligned.py
```

Problem 1 requires the external tools and models described in
`../../reference/`. Problem 2 requires only the local BERT checkout in
`reference/models/bert-base-uncased/` in addition to its Python runtime.
