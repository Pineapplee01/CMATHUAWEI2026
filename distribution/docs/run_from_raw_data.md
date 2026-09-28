# Running From Contest Data

The bundled contest data supports these input chains:

| Problem | Raw input | What can be generated | Additional requirement |
| --- | --- | --- | --- |
| 1 | Attachment 1 videos and labels | Text, audio, and vision features | BERT, OpenFace, COVAREP, MFA, `ffmpeg`, and `matlab` |
| 2 | Attachment 2 aligned/unaligned PKL bundles | Model-ready NPZ splits and scalers | Local BERT checkout |
| 2 prediction | Attachment 3 PKL samples | Prediction CSV | Problem 2 NPZ splits, BERT, and trained checkpoint |
| 3 | Attachment 4 feature PKL samples | Explanation results | Problem 2 NPZ splits, BERT, and trained checkpoint |

Attachment 2 is the only training-data preprocessing chain. It reads the
contest-supplied top-level `train`, `valid`, and `test` mappings directly; it
does not expect obsolete `train.pkl`, `valid.pkl`, or `test.pkl` directories.

All derived files are written below `../results/`. `../data/` remains a
read-only contest attachment bundle.

Before running a Python workflow, install the optional runtime dependencies:

```bash
python -m pip install -e ".[runtime,preprocess]"
```
