# sunny-batched-geneig

A batched complex Hermitian-definite generalized eigenvalue solver implemented with cuSolverDx + NVRTC, targeting the matrix-pair workload of Sunny.jl quantum spin dynamics simulations at SNS-scale neutron scattering computations.

## Prerequisites

- CUDA Toolkit 12.6.3+ (12.8 confirmed)
- NVIDIA MathDx 25.12.1 (cuSolverDx + bundled CUTLASS)
- NVIDIA GPU with compute capability sm_70 or newer (sm_120 / Blackwell confirmed)
- CMake 3.24+

Set `MATHDX_ROOT` to the MathDx install root (the directory containing `nvidia/mathdx/25.12/`). The provided `scripts/env.sh` defaults it to `$HOME/opt/nvidia-mathdx-25.12.1-cuda12`.

## Build

```
source scripts/env.sh
cmake -B build -S .
cmake --build build -j
```

## Run smoke test

```
./build/src/smoke
```

Expected output:

```
cuSolverDx smoke OK, shared mem bytes: 8448
```
