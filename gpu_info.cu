#include <cstdio>
#include <cuda.h>

static void displayDeviceProperties(cudaDeviceProp *devProps);

int main()
{
    int devCount;
    cudaGetDeviceCount(&devCount);

    std::printf("CUDA device count: %d\n", devCount);
    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp devProps;
        cudaGetDeviceProperties(&devProps, i);
        std::printf("CUDA device properties for device %d:\n", i);
        displayDeviceProperties(&devProps);
    }
    return 0;
}

static void displayDeviceProperties(cudaDeviceProp *devProps)
{
    std::printf("name                  : %s\n", devProps->name);
    std::printf("total global mem      : %zd\n", devProps->totalGlobalMem);
    std::printf("shared mem per block  : %zd\n", devProps->sharedMemPerBlock);
    std::printf("regs per block        : %d\n", devProps->regsPerBlock);
    std::printf("warp size             : %d\n", devProps->warpSize);
    std::printf("mem pitch             : %zd\n", devProps->memPitch);
    std::printf("max threads per block : %d\n", devProps->maxThreadsPerBlock);
    std::printf("max threads dim       : %d %d %d\n",
        devProps->maxThreadsDim[0],
        devProps->maxThreadsDim[1],
        devProps->maxThreadsDim[2]);
    std::printf("max grid size         : %d %d %d\n",
        devProps->maxGridSize[0],
        devProps->maxGridSize[1],
        devProps->maxGridSize[2]);
    std::printf("clock rate            : %d\n", devProps->clockRate);
    std::printf("total const mem       : %zd\n", devProps->totalConstMem);
    std::printf("major                 : %d\n", devProps->major);
    std::printf("minor                 : %d\n", devProps->minor);
}

