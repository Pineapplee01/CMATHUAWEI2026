"""Report the current main-method state from the Baseline Workspace registry."""

from __future__ import annotations

from e_emotion.problem2_fair.registry import load_registry


def main() -> int:
    record = load_registry().get("problem2_main")
    print(f"{record.method_id}: {record.state}; new seeds require retraining")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
