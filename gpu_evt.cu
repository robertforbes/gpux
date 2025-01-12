#include <cstdio>
#include <cuda.h>

const int N = 1000;

// GPU functions.
__global__ static void vec_add_gpu(float *out, float *x, float *y, int n);

static float x[N];
static float y[N];

int main()
{
    // Initialise test vectors.
    for(int i = 0; i < N; i++)
    {
        x[i] = 19.0f;
        y[i] = 23.0f;
    }

    float *d_x;
    float *d_y;
    float *d_out0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory.
    cudaMalloc((void**)&d_x, sizeof(float) * N);
    cudaMalloc((void**)&d_y, sizeof(float) * N);
    cudaMalloc((void**)&d_out0, sizeof(float) * N);

    // Copy CPU -> GPU.
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * N, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    // Run 1 block, 1 thread per block.
    vec_add_gpu<<<1,1>>>(d_out0, d_x, d_y, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::printf("start/stop event delta: %10.5fms\n", milliseconds);

    // Free device memory.
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out0);

    return 0;
}

__global__ static void vec_add_gpu(float *out, float *x, float *y, int n)
{
    auto block_len = n / gridDim.x;
    auto slice_len = n / (gridDim.x * blockDim.x);
    int start = blockIdx.x * block_len + threadIdx.x * slice_len;
    int end = start + slice_len;

    for(int i = start; i < end; i++)
    {
        out[i] = x[i] + y[i];
    }
}
