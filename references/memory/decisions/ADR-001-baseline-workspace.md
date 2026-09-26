# ADR-001: Baseline Workspace Layout

## Decision

reference contains only memory, runtime, vendor and its navigation README.
The Python environment is /Baseline/.venv; raw experiment evidence is
/Baseline/artifacts/problem2.

## Consequences

Method identity, source provenance and capability state have one source in the
method registry. Vendor code remains immutable. Project adaptations are regular
Python modules. Historical records remain inspectable but do not enter current
fair reports.
