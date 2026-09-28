# Domain Context

## Submission Preservation

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Submission Snapshot** | The immutable code and small artifacts copied from the final contest delivery. | final folder, submit code |
| **Canonical Distribution** | The English-named standalone Python project derived from the Submission Snapshot. | cleaned copy, refactor folder |
| **Legacy Path** | A path from the server or submission-aligned tree retained for traceability. | old name, Chinese folder |
| **Path Mapping** | A documented relationship from a Legacy Path to a Canonical Distribution path. | rename notes |

## Data And Features

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **External Asset** | Data, model weights, pretrained models, or external tools required at runtime but not packaged. | dependency blob, local file |
| **Aligned Layout** | A multimodal sample layout where text, audio, and vision share 50 slots. | aligned version |
| **Unaligned Layout** | A multimodal sample layout where text uses 50 slots and audio/vision use native 500-slot axes. | native layout |
| **Effective Axis** | The mask `P` indicating where a modality can be meaningfully read. | padding mask |
| **Observation Mask** | The mask `O` indicating observed feature values inside the Effective Axis. | available mask |
| **Preprocessing Pipeline** | The process that transforms contest attachments into model-ready arrays and masks. | data cleaning |

## Relationships

- A **Canonical Distribution** is derived from exactly one **Submission Snapshot**.
- A **Path Mapping** connects one or more **Legacy Paths** to a canonical path.
- An **Observation Mask** is always interpreted inside an **Effective Axis**.
- An **Aligned Layout** and an **Unaligned Layout** are separate input contracts.
- An **External Asset** is required for runtime but excluded from the package.
