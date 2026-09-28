# Recovered Appendix 2 generation logic

These files were recovered from Cursor agent transcript history on the server because the live CPMCM tree no longer contains the Appendix 2 NPZ generation scripts.

Source transcript records:
- aligned write: `/user_home/gaojianan/.cursor/projects/user-home-gaojianan/agent-transcripts/7b5ebc1b-257c-4d7d-81ca-6842948132ce/7b5ebc1b-257c-4d7d-81ca-6842948132ce.jsonl`, line 149, path `/user_home/gaojianan/CPMCM/AAA提交版代码及结果/问题二/代码/数据预处理/对齐版本/preprocess_po_aligned.py`
- unaligned write: `/user_home/gaojianan/.cursor/projects/user-home-gaojianan/agent-transcripts/d15982ae-ab20-42ff-8ab8-c24021c8e84d/d15982ae-ab20-42ff-8ab8-c24021c8e84d.jsonl`, line 447, path `/user_home/gaojianan/CPMCM/AAAdata/Appendix_2/标准化/未对齐版本/预处理/preprocess_po_unaligned.py`

Recovered files:
- `preprocess_po_aligned.original_write.py`: exact aligned script body from the Write record.
- `preprocess_po_unaligned.original_write.py`: exact unaligned script body from the Write record.
- `preprocess_po_aligned.recovered_nested_final.py`: aligned script after the observed ROOT correction for the historical nested `数据预处理/对齐版本` location.
- `preprocess_po_unaligned.recovered_nested_final.py`: unaligned script after the observed RAW/report/ROOT corrections for the historical nested `数据预处理/未对齐版本` location.
- `preprocess_aligned.flat_final.py` and `preprocess_unaligned.flat_final.py`: reconstructed flat-location variants matching the later `问题二/代码/*.py` placement seen in the transcript.

Current server search result: the live tree only has Problem 3 `preprocess_aligned.py` / `preprocess_unaligned.py`; Appendix 2 generator scripts are not present as ordinary files anymore.

SHA256:
- `preprocess_aligned.flat_final.py` (8349 bytes): `91669297dc388d83a6a3ae8bcf0afc237b9abfda8304e03178a305f5293fa765`
- `preprocess_po_aligned.original_write.py` (8349 bytes): `9fd8220f3ccd8861f8610347e3803075c12724906ee199eeedf9af077caf38be`
- `preprocess_po_aligned.recovered_nested_final.py` (8349 bytes): `a6457b454804f01e6f70bfbdfea84812546dbc7eeec7f32ce8a378cca23b03fa`
- `preprocess_po_unaligned.original_write.py` (9861 bytes): `9bd084f3de17aac57af9eac7ea65a8267376c89dc46d1140760a1beed0202704`
- `preprocess_po_unaligned.recovered_nested_final.py` (9953 bytes): `425fb5fa27274eb9d64873ec0229d32b1c82aae5cf6b59605bfe5e80ea1aa34d`
- `preprocess_unaligned.flat_final.py` (9953 bytes): `25e9cc9cba7698181319bf7aff740daf6178dbb54185bd0d14d1e3cc8dca8472`
