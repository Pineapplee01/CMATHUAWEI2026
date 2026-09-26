# Baseline Workspace Memory

This directory is the tracked memory layer for the Baseline Workspace.

- catalog is the source of truth for methods and vendor snapshots.
- conductor records the Baseline-specific product, stack, workflow and tracks.
- operations records server layout and synchronization rules.
- results contains compact Git-tracked experiment summaries.
- history preserves pre-restructure records without treating them as current status.

Raw sources live in ../vendor. Raw checkpoints, predictions and logs live
outside this directory in artifacts/problem2.
