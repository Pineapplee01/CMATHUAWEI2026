# Preprocess Module

This package contains the normalized preprocessing code used by the local
distribution. It is the maintained counterpart to the historical copies in
`distribution/code/preprocess/`.

| Module | Input | Output |
| --- | --- | --- |
| `problem1/` | Attachment 1 videos and labels | `results/appendix_1/` multimodal features |
| `appendix2_aligned.py` | `data/appendix_2/aligned_50.pkl` | `results/appendix_2/aligned/` NPZ splits and scalers |
| `appendix2_unaligned.py` | `data/appendix_2/unaligned_50.pkl` | `results/appendix_2/unaligned/` NPZ splits and scalers |
| `appendix4_aligned.py` | Attachment 4 aligned PKL features | Aligned explanation-model batch |
| `appendix4_unaligned.py` | Attachment 4 unaligned PKL features | Unaligned explanation-model batch |

The module never generates new feature files from Attachment 3 or Attachment 4
videos. Those attachments provide contest-supplied PKL features directly.

External models and tools are documented in `../../../reference/`.
