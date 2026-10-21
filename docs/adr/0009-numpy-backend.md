# ADR 0009: NumPy arrays, with performance-critical kernels in one module

**Status:** Accepted, 2026-10-21

## Context

The simulation holds preallocated float64 buffers and runs many small vectorized operations. It's memory-bound rather than compute-bound, runs locally on a CPU, and must keep every random draw reproducible per trajectory.

## Decision

Use NumPy for buffers and kernels, with `numpy.random.Generator` streams spawned by `SeedSequence` for every draw. Performance-critical kernels, such as the group aligner, the samplers, and the apply step, live in one `ops` module.

## Alternatives considered

- **PyTorch tensors.** They offer a GPU path, but a GPU doesn't help a memory-bound workload, and Apple's GPU backend has no float64. Some PyTorch distributions also can't take a dedicated generator, and the model libraries all speak NumPy.

## Consequences

- No tensor-array conversion at model boundaries, and float64 is the default.
- mypy can check array element types, and Hypothesis can generate arrays.
- A GPU backend remains possible later, behind the `ops` module.
