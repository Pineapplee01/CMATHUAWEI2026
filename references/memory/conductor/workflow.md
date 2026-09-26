# Baseline Workspace Workflow

1. Resolve a method through catalog/methods.yaml.
2. Resolve its source snapshot through catalog/vendor-lock.json.
3. Run the registered adapter with one view and one fixed seed.
4. Write raw evidence through ArtifactStore.
5. Verify checkpoint, source data, Q2 predictions and metrics.
6. Update the tracked compact summary in results/problem2.

No queue may construct method source, run or report paths directly.
