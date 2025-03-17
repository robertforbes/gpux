# gpux
GPU Experiments.

This repository constains some simple examples to try out programming an Nvidia GPU.

## Building

Examples can be built on the command line using the Nvidia compiler, e.g.
```
nvcc  -Wno-deprecated-gpu-targets -o vec_add vec_add.cu
nvcc  -Wno-deprecated-gpu-targets -o gpu_info gpu_info.cu
```
