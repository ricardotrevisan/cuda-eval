# CUDA Intro

A hands-on introduction to CUDA GPU programming, following NVIDIA's [Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda).

## Contents

| File | Description |
|------|-------------|
| `main.cpp` | CPU baseline — array addition on 1M elements using standard C++ |
| `add.cu` | CUDA version — same computation offloaded to the GPU using Unified Memory |
| `nsys_easy` | Helper script wrapping `nsys profile` + `nsys stats` for quick CUDA profiling |

## Progression

1. **CPU (main.cpp)** — plain C++ loop, single-threaded
2. **GPU (add.cu)** — CUDA kernel with Unified Memory (`cudaMallocManaged`), running 1 block of 256 threads with stride-loop pattern

## Building

```bash
# CPU version
g++ -o main main.cpp

# CUDA version
nvcc -o add_cuda add.cu
```

## Running

```bash
./add_cuda
# Max error: 0
```

## Profiling with Nsight Systems

```bash
# Direct nsys command
/opt/nvidia/nsight-systems/2026.2.1/bin/nsys profile -t cuda --stats=true ./add_cuda

# Or use the nsys_easy wrapper
./nsys_easy ./add_cuda

# With custom options
./nsys_easy -t cuda,osrt -o my_output ./add_cuda
```

### nsys_easy options

| Flag | Default | Description |
|------|---------|-------------|
| `-t` | `cuda` | Trace options (e.g. `cuda,osrt`) |
| `-s` | `none` | Sampling |
| `-c` | `none` | CPU context switch tracing |
| `-o` | `nsys_easy` | Output file name |
| `-r` | `cuda_gpu_sum` | Stats report name |

## References

- [Even Easier Introduction to CUDA — NVIDIA Blog](https://developer.nvidia.com/blog/even-easier-introduction-cuda)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
