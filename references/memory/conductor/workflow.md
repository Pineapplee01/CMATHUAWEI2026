# Baseline Workspace Workflow

1. Resolve a method through catalog/methods.yaml.
2. Resolve its source snapshot through catalog/vendor-lock.json.
3. Run the registered adapter with one view and one fixed seed.
4. Write raw evidence through ArtifactStore.
5. Record both the source data view and the actual model input view; reports
   never aggregate them into one table.
6. Verify checkpoint, source data, Q2 predictions, metrics and view identity.
7. Update the tracked compact summary in results/problem2.

No queue may construct method source, run or report paths directly.
