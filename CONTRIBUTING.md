# Contributing

1. Create a focused branch and include tests for behavior changes.
2. Do not hard-code paper table values as experiment output.
3. Keep all compared methods on the same data split, action set, solver, and evaluation tolerance.
4. Record new assumptions in `ASSUMPTIONS.md`.
5. Run `pytest` and `python -m cfq.cli smoke` before opening a pull request.
6. For new datasets, document source, license, favorable target, immutable features, categorical groups, bounds, and cost weights.
