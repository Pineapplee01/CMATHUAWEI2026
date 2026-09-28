# Appendix 4 Input Construction

These helpers convert the contest-provided Appendix 4 PKL features into model
batches for the aligned and unaligned explanation models. They read:

```text
data/appendix_4/aligned/
data/appendix_4/unaligned/
```

No video feature extraction occurs here. Videos are retained for review only;
the PKL files provide text, audio, and vision features. The helpers use the
Problem 2 scaler output from `results/appendix_2/` and BERT from
`reference/models/bert-base-uncased/`.
