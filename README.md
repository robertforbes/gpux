# gpux
GPU Experiments.

This repository contains some simple examples to try out programming an Nvidia GPU.

## Building

Examples can be built on the command line using the Nvidia compiler, e.g.
```
nvcc  -Wno-deprecated-gpu-targets -o vec_add vec_add.cu
nvcc  -Wno-deprecated-gpu-targets -o gpu_info gpu_info.cu
```
The dynamic parallelism experiment requires an extra option:
```
nvcc  -Wno-deprecated-gpu-targets -rdc=true -o gpu_parent_child gpu_parent_child.cu
```
