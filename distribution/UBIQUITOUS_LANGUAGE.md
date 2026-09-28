# Ubiquitous Language

## Project Lifecycle

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Submission Snapshot** | The preserved final-delivery tree used as audit evidence. | final code, submit folder |
| **Canonical Distribution** | The English-named, self-contained Python code distribution. | normalized copy |
| **Archive Index** | A document that describes historical material without renaming it. | archive cleanup |
| **Path Mapping** | A table mapping Legacy Paths to canonical distribution paths. | path notes |

## Multimodal Data

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Aligned Layout** | A 50-slot layout shared by text, audio, and vision. | aligned data |
| **Unaligned Layout** | A layout with 50 text slots and 500 audio/vision slots. | unaligned data |
| **Effective Axis** | A modality mask showing structurally valid slots. | P mask only |
| **Observation Mask** | A modality mask showing observed values within the Effective Axis. | O mask only |
| **Scaler Parameters** | Train-set normalization statistics stored in `scaler_params.npz`. | standardization file |
| **Contest Attachment** | An immutable original file supplied with the competition problem. | raw asset, original data |
| **Bundled Contest Data** | The English-named local layout of Contest Attachments under `data/`. | external data, AAAdata |
| **Result Artifact** | A derived runtime file written under `results/`. | processed raw data, final data |
| **External Asset** | A model, trained checkpoint, or feature-extraction tool excluded from the distribution. | dependency file, external data |

## Processing

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Preprocessing Pipeline** | A reproducible transformation from contest attachment features to model input. | data prep |
| **Prediction Pipeline** | The Problem 2 model path from model-ready input to sentiment class and intensity. | inference code |
| **Explanation Pipeline** | The Problem 3 path that perturbs Observation Masks to explain predictions. | interpretability script |

## Example Dialogue

> **Dev:** "Can I feed Appendix 4 videos directly to the **Explanation Pipeline**?"
>
> **Domain expert:** "No. Appendix 4 uses provided `.pkl` feature files as **External Assets**; videos are only for media review."
>
> **Dev:** "So for the **Unaligned Layout**, audio and vision stay on 500-slot axes?"
>
> **Domain expert:** "Exactly. The **Observation Mask** is built on each native axis and interpreted inside the **Effective Axis**."

> **Dev:** "Where should the Appendix 2 NPZ files be written?"
>
> **Domain expert:** "They are **Result Artifacts**, so they go under `results/`; the **Bundled Contest Data** remains unchanged."

## Flagged Ambiguities

- "final version" must be avoided unless it means **Submission Snapshot**.
- "preprocessed data" must specify whether it refers to raw `.pkl` features, regenerated NPZ files, or model-ready tensors.
- "external data" must not mean **Bundled Contest Data**: contest attachments are present under `data/`, while **External Assets** are models, checkpoints, and tools only.
