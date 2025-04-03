#include <cstdio>
#include <cuda.h>

static void displayDeviceProperties(cudaDeviceProp *devProps);
static void displayMemInfo(void);

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
    displayMemInfo();
    return 0;
}

static void displayDeviceProperties(cudaDeviceProp *devProps)
{
    std::printf("name                           : %s\n", devProps->name);
    std::printf("total global mem               : %zd\n", devProps->totalGlobalMem);
    std::printf("shared mem per block           : %zd\n", devProps->sharedMemPerBlock);
    std::printf("regs per block                 : %d\n", devProps->regsPerBlock);
    std::printf("warp size                      : %d\n", devProps->warpSize);
    std::printf("mem pitch                      : %zd\n", devProps->memPitch);
    std::printf("max threads per block          : %d\n", devProps->maxThreadsPerBlock);
    std::printf("max threads per multiprocessor : %d\n", devProps->maxThreadsPerMultiProcessor);
    std::printf("max threads dim                : %d %d %d\n",
        devProps->maxThreadsDim[0],
        devProps->maxThreadsDim[1],
        devProps->maxThreadsDim[2]);
    std::printf("max grid size                  : %d %d %d\n",
        devProps->maxGridSize[0],
        devProps->maxGridSize[1],
        devProps->maxGridSize[2]);
    std::printf("clock rate                     : %d\n", devProps->clockRate);
    std::printf("total const mem                : %zd\n", devProps->totalConstMem);
    std::printf("major                          : %d\n", devProps->major);
    std::printf("minor                          : %d\n", devProps->minor);
    std::printf("texture alignment              : %zd\n", devProps->textureAlignment);
    std::printf("device overlap                 : %d\n", devProps->deviceOverlap);
    std::printf("multiprocessor count           : %d\n", devProps->multiProcessorCount);
    std::printf("kernel exec timeout enabled    : %d\n", devProps->kernelExecTimeoutEnabled);
    std::printf("integrated                     : %d\n", devProps->integrated);
    std::printf("can map host memory            : %d\n", devProps->canMapHostMemory);
    std::printf("compute mode                   : %d\n", devProps->computeMode);

    std::printf("surface alignment              : %zd\n", devProps->surfaceAlignment);
    std::printf("concurrent kernels             : %d\n", devProps->concurrentKernels);
    std::printf("ECC enabled                    : %d\n", devProps->ECCEnabled);
    std::printf("PCI bus ID                     : %d\n", devProps->pciBusID);
    std::printf("PCI device ID                  : %d\n", devProps->pciDeviceID);
    std::printf("PCI domain ID                  : %d\n", devProps->pciDomainID);
    std::printf("tccDriver                      : %d\n", devProps->tccDriver);
    std::printf("async engine count             : %d\n", devProps->asyncEngineCount);
    std::printf("unified addressing             : %d\n", devProps->unifiedAddressing);
    std::printf("memory clock rate              : %d\n", devProps->memoryClockRate);
    std::printf("memory bus width               : %d\n", devProps->memoryBusWidth);

    std::printf("registers per block            : %d\n", devProps->regsPerBlock);
    std::printf("registers per multiprocessor   : %d\n", devProps->regsPerMultiprocessor);

    std::printf("reserved shared mem per block  : %zd\n", devProps->reservedSharedMemPerBlock);
    std::printf("unified function pointers      : %d\n", devProps->unifiedAddressing);

    std::printf("tcc driver                     : %d\n", devProps->tccDriver);
    std::printf("texture pitch alignment        : %zd\n", devProps->texturePitchAlignment);
    std::printf("\n");
}

static void displayMemInfo(void)
{
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::printf("CUDA free memory               : %zd\n", freeMem);
    std::printf("CUDA total memory              : %zd\n", freeMem);
}
