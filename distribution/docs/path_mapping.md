# Legacy Path Mapping

The Canonical Distribution uses English package paths. The legacy Chinese paths
remain traceable through this mapping and through PR #1.

| Legacy path | Canonical path |
| --- | --- |
| `AAA提交版代码及结果/问题一/代码/common.py` | `src/multimodal_emotion/preprocess/problem1/common.py` |
| `AAA提交版代码及结果/问题一/代码/method_a.py` | `src/multimodal_emotion/preprocess/problem1/method_a.py` |
| `AAA提交版代码及结果/问题一/代码/method_b.py` | `src/multimodal_emotion/preprocess/problem1/method_b.py` |
| `preprocess/problem2/preprocess_aligned.py` | `src/multimodal_emotion/preprocess/appendix2_aligned.py` |
| `preprocess/problem2/preprocess_unaligned.py` | `src/multimodal_emotion/preprocess/appendix2_unaligned.py` |
| `AAA提交版代码及结果/问题二/代码/aligned.py` | `src/multimodal_emotion/prediction/problem2_aligned.py` |
| `AAA提交版代码及结果/问题二/代码/unaligned.py` | `src/multimodal_emotion/prediction/problem2_unaligned.py` |
| `AAA提交版代码及结果/问题二/代码/q2_utils.py` | `src/multimodal_emotion/prediction/q2_utils.py` |
| `AAA提交版代码及结果/问题三/代码/preprocess_aligned.py` | `src/multimodal_emotion/preprocess/appendix4_aligned.py` |
| `AAA提交版代码及结果/问题三/代码/preprocess_unaligned.py` | `src/multimodal_emotion/preprocess/appendix4_unaligned.py` |
| `AAA提交版代码及结果/问题三/代码/aligned.py` | `src/multimodal_emotion/explanation/problem3_aligned.py` |
| `AAA提交版代码及结果/问题三/代码/unaligned.py` | `src/multimodal_emotion/explanation/problem3_unaligned.py` |
| `AAA提交版代码及结果/问题三/代码/q3_utils.py` | `src/multimodal_emotion/explanation/q3_utils.py` |
| `AAAdata/Appendix_1/` | `data/appendix_1/` |
| `AAAdata/Appendix_2_raw/` | `data/appendix_2/` |
| `AAAdata/Appendix_2/{对齐版本,未对齐版本}/` | `results/appendix_2/{aligned,unaligned}/` |
| `AAAdata/Appendix_3/{对齐版本,未对齐版本}/` | `data/appendix_3/{aligned,unaligned}/` |
| `AAAdata/Appendix_4/{对齐版本,未对齐版本}/` | `data/appendix_4/{aligned,unaligned}/` |
| `AAAmodel/bert-base-uncased/` | `reference/models/bert-base-uncased/` |
| `AAAmodel/{OpenFace,covarep,mfa}/` | `reference/tools/` |
| `AAAcheckpoints/` and `AAAmodel/checkpoints/` | `reference/checkpoints/` |
