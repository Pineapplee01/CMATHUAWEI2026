# Appendix 1 Preprocessing

The Method A and Method B scripts transform the original Attachment 1 videos
and labels into multimodal features.

Input: `data/appendix_1/mosei_raw_videos_100/`.

Output: `results/appendix_1/method_a/` or `results/appendix_1/method_b/`.

Required placeholders and expected locations are documented under
`../../../reference/`:

- `models/bert-base-uncased/`
- `tools/openface/FeatureExtraction`
- `tools/covarep/`
- `tools/mfa/`

The system also needs `ffmpeg`, `matlab`, and `mfa` on `PATH`.
