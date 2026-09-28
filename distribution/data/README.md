# Contest Data

This directory contains only data provided by the contest problem statement.
Directory names are standardized to English; file contents and identifiers are
not transformed.

## GitHub Publication

The raw attachment files are intentionally kept out of GitHub. The local copy
is approximately 3.8 GiB, and the Attachment 2 PKL files exceed GitHub's
single-file limit. A clone contains this README only; obtain the contest
attachments separately and restore the layout below without renaming files.

Source copied from:

`G:\CMath\2026华为杯\E题数据`

## Layout

| Directory | Original contest attachment | Contents |
| --- | --- | --- |
| `appendix_1/mosei_raw_videos_100/` | Attachment 1 | Original videos and `label-100.xlsx` for Problem 1. |
| `appendix_2/` | Attachment 2 | `aligned_50.pkl`, `unaligned_50.pkl`, and `label.xlsx`; each PKL contains `train`, `valid`, and `test`. |
| `appendix_3/{aligned,unaligned}/` | Attachment 3 | Missing-modality PKL samples for Problem 2 prediction. |
| `appendix_4/{aligned,unaligned}/` | Attachment 4 | Feature PKL samples and videos for Problem 3 explanation. |

Derived files belong in `../results/`, never in this directory.

Excluded on purpose:

- generated `.npz` intermediate arrays;
- generated `.csv` prediction or result tables;
- generated figures and key frames;
- pretrained model directories;
- trained checkpoints and weight files;
- system files such as `.DS_Store`.
