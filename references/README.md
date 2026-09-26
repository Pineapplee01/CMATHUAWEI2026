# Baseline Workspace

This directory mirrors the server Baseline reference workspace.

- [Memory](memory/README.md) holds method identity, source provenance,
  operations, decisions and compact result summaries.
- [Vendor](vendor/) holds immutable third-party source snapshots with their
  own Git history.
- [History](memory/history/) holds records created before the workspace
  restructure.

Project-maintained baseline implementations live in src/e_emotion/baselines.
Raw runs, checkpoints, predictions and logs live in artifacts/problem2.
