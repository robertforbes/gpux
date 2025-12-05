// Experiment with cooperative groups, based on NVIDIA sample code.
#include <cstdio>

#include <cooperative_groups.h>

using namespace cooperative_groups;

// CUDA kernel.
__global__ void groupKernel()
{
    // threadBlockGroup includes all threads in the block
    thread_block threadBlockGroup     = this_thread_block();
    int          threadBlockGroupSize = threadBlockGroup.size();
    std::printf("threadBlockGroupSize: %d\n", threadBlockGroupSize);

    threadBlockGroup.sync();
}

int main() {
    cudaError_t err;

    // Launch the kernel
    int blocksPerGrid   = 1;
    int threadsPerBlock = 64;

    std::printf("\nLaunching a single block with %d threads...\n\n", threadsPerBlock);

    // we use the optional third argument to specify the size
    // of shared memory required in the kernel
    groupKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>();
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    
    return 0;
}
