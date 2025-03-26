// Simple experiment spawning children from a kernel function. 
// Known as dynamic parallelism.
#include <cstdio>
#include <cuda.h>

__global__ void child_kernel(int n) 
{
    std::printf("inside child kernel %d\n", threadIdx.x);
}

__global__ void parent_kernel(int n) 
{
    std::printf("inside parent kernel, about to create %d children\n", n);
    child_kernel<<<1,n>>>(n);
}

int main()
{
    std::printf("Host process running, about to spawn parent kernel\n");
    int n = 100;
    parent_kernel<<<1,1>>>(n);
    cudaDeviceSynchronize();
    std::printf("Host process after sync\n");
    return 0;
}
