# Release 1 Test Strategy

## CPU Unit Tests
- Execute all ops in `ops.py`.
- Ensure numerical stability and backward correctness (numerical gradient).

## GPU E2E Tests
- Test in Chromium via Playwright.
- Verify Tensor mapping, execution constraints, and quotas.

## Memory Leak Tests
- 1000 iterations of forward/backward/step.
- Peak memory shouldn't increase.
