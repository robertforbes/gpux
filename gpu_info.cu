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
    std::printf("name: %s\n", devProps->name);
}
