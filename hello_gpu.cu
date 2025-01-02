#include <stdio.h>
#include <cuda.h>

__global__ void dkernel()
{
    printf("Hello from GPU.\n");
}

int main()
{
    // Run 1 block, 1 thread per block.
    dkernel<<<1,1>>>();

    cudaDeviceSynchronize();
    return 0;
}

