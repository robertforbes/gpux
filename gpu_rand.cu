#include <cstdio>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void gpuRand(curandState* state, uint64_t seed, float* rndVals, int n)
{
    int tid = 0;
    curand_init(seed, tid, 0, state);
    for(int i = 0; i < n; ++i)
    {
        std::printf("curand_uniform %10.5f\n", curand_uniform(state));
    }
}

__global__ void gpuRandInt(
    curandState* state,
    uint64_t seed,
    int* rndVals,
    int n,
    int min,
    int max)
{
    int tid = 0;
    curand_init(seed, tid, 0, state);
    for(int i = 0; i < n; ++i)
    {
        float rndF = curand_uniform(state);
        std::printf("curand_uniform %10.5f\n", rndF);
        float scaled = (float)min + rndF * (float)(max - min);
        int rndI = (int)round(scaled); 
        std::printf("rand int %d\n", rndI);
    }
}

int main()
{
    curandState* state;
    float* devRndVals;
    int* devIntVals;
    // float* hostRndVals;
    int nThreads = 1;
    int N = 10;

    cudaMalloc(&state, nThreads * sizeof(curandState));
    cudaMalloc(&devRndVals, N * sizeof(float));
    cudaMalloc(&devIntVals, N * sizeof(int));

    gpuRand<<<1,1>>>(state, 123, devRndVals, N);
    gpuRandInt<<<1,1>>>(state, 123, devIntVals, N, 3, 7);

    cudaDeviceSynchronize();
    cudaFree(state);
    cudaFree(devRndVals);
    cudaFree(devIntVals);

    return 0;
}
