# gpux
GPU Experiments.

This repository contains some simple examples to try out programming an Nvidia GPU.

1. gpu_info.cu         - Reads some device properties for each connected CUDA device.
2. gpu_fp.cu           - Experiment in passing function pointers to the device.
3. gpu_event.cu        - Use of CUDA events for timestamping.
4. gpu_rand.cu         - Use of curand API for random number generation. 
5. gpu_parent_child.cu - Launching child kernels on device. 
6. vec_add.cu          - Experiments with vector addition and block/thread configs.
7. hello_gpu.cu        - CUDA hello world. 

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
