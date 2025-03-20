#include <cstdio>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void gpurand(curandState* state, uint64_t seed, float* rndVals, int n)
{
    int tid = 0;
    curand_init(seed, tid, 0, state);
    for(int i = 0; i < n; ++i)
    {
        std::printf("curand_uniform %10.5f\n", curand_uniform(state));
    }
}

int main()
{
    curandState* state;
    float* devRndVals;
    // float* hostRndVals;
    int nThreads = 1;
    int N = 10;

    cudaMalloc(&state, nThreads * sizeof(curandState));
    cudaMalloc(&devRndVals, N * sizeof(float));

    gpurand<<<1,1>>>(state, 123, devRndVals, N);

    cudaDeviceSynchronize();
    cudaFree(state);
    cudaFree(devRndVals);
    return 0;
}
