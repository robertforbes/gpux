#include <stdio.h>
#include <cuda.h>
#include <chrono>

#define N 1000000

// CPU functions.
static void vec_add(float *out, float *x, float *y, int n);
static void print_vec(float *vec, int n);
static void check_vec(float *vec, float *ref, float thresh, int n);

// GPU functions.
__global__ static void vec_add_gpu(float *out, float *x, float *y, int n);

using std::chrono::high_resolution_clock;
using std::chrono::duration;

static float x[N];
static float y[N];
static float cpu_out[N];
static float gpu_out0[N];
static float gpu_out1[N];
static float gpu_out2[N];

int main()
{
    duration<double, std::micro> us;

    for(int i = 0; i < N; i++)
    {
        x[i] = 19.0f;
        y[i] = 23.0f;
    }

    auto t0 = high_resolution_clock::now();
    vec_add(cpu_out, x, y, N);
    auto t1 = high_resolution_clock::now();
    us = t1 - t0;

    printf("vec_add CPU, %10.5fus, output:\n", us.count());
    print_vec(cpu_out, 10);

    float *d_x;
    float *d_y;
    float *d_out0;
    float *d_out1;
    float *d_out2;
    
    // Allocate device memory.
    cudaMalloc((void**)&d_x, sizeof(float) * N);
    cudaMalloc((void**)&d_y, sizeof(float) * N);
    cudaMalloc((void**)&d_out0, sizeof(float) * N);
    cudaMalloc((void**)&d_out1, sizeof(float) * N);
    cudaMalloc((void**)&d_out2, sizeof(float) * N);

    // Copy CPU -> GPU.
    t0 = high_resolution_clock::now();
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * N, cudaMemcpyHostToDevice);
    t1 = high_resolution_clock::now();

    // Run 1 block, 1 thread per block.
    vec_add_gpu<<<1,1>>>(d_out0, d_x, d_y, N);
    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();

    // Run 1 block, 100 threads per block.
    vec_add_gpu<<<1,100>>>(d_out1, d_x, d_y, N);
    cudaDeviceSynchronize();
    auto t3 = high_resolution_clock::now();

    // Run 100 blocks, 1000 threads per block.
    vec_add_gpu<<<100,1000>>>(d_out2, d_x, d_y, N);
    cudaDeviceSynchronize();
    auto t4 = high_resolution_clock::now();

    // Copy results GPU -> CPU.
    cudaMemcpy(gpu_out0, d_out0, sizeof(float) * N, cudaMemcpyDeviceToHost);
    auto t5 = high_resolution_clock::now();
    cudaMemcpy(gpu_out1, d_out1, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_out2, d_out2, sizeof(float) * N, cudaMemcpyDeviceToHost);
   
    // Calculate GPU durations.
    duration<double, std::micro> d1, d2, d3, d4, d5;
    d1 = t1 - t0;
    d2 = t2 - t1;
    d3 = t3 - t2;
    d4 = t4 - t3;
    d5 = t5 - t4;
    printf("vec_add GPU, cp1 %10.5fus, ex1 %10.5fus, ex2 %10.5fus, ex3 %10.5fus, cp2 %10.5fus, output:\n",
        d1.count(),
        d2.count(),
        d3.count(),
        d4.count(),
        d5.count());
    print_vec(gpu_out0, 10);
    check_vec(gpu_out0, cpu_out, 0.00001, N);
    print_vec(gpu_out1, 10);
    check_vec(gpu_out1, cpu_out, 0.00001, N);
    print_vec(gpu_out2, 10);
    check_vec(gpu_out2, cpu_out, 0.00001, N);

    // Free device memory.
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out0);
    cudaFree(d_out1);
    cudaFree(d_out2);

    return 0;
}

void vec_add(float *out, float *x, float *y, int n)
{
    for(int i = 0; i < n; i++)
    {
        out[i] = x[i] + y[i];
    }
}

__global__ static void vec_add_gpu(float *out, float *x, float *y, int n)
{
    // printf("gridDim x %d, y %d, z %d\n", gridDim.x, gridDim.y, gridDim.z);
    // printf("blockDim x %d, y %d, z %d\n", blockDim.x, blockDim.y, blockDim.z);
    // printf("blockIdx x %d, y %d, z %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    // printf("threadIdx x %d, y %d, z %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    auto block_len = n / gridDim.x;
    auto slice_len = n / (gridDim.x * blockDim.x);
    int start = blockIdx.x * block_len + threadIdx.x * slice_len;
    int end = start + slice_len;
    // printf("block_len %d, slice_len %d\n", block_len, slice_len);
    // printf("start %d, end %d\n", start, end);

    for(int i = start; i < end; i++)
    {
        out[i] = x[i] + y[i];
    }
}

static void print_vec(float *vec, int n)
{
    for(int i = 0; i < n; i++)
    {
        printf("%10.5f ", vec[i]);
    }
    printf("\n");
}

static void check_vec(float *vec, float *ref, float thresh, int n)
{
    for(int i = 0; i < n; i++)
    {
        float diff = fabs(vec[i] - ref[i]);
        if(diff > thresh)
        {
            printf("check_vec failed at index %d, vec %10.5f, ref %10.5f, diff %10.5f, thresh %10.5f\n",
                i,
                vec[i],
                ref[i],
                diff,
                thresh);
            return;
        }
    }
    printf("check_vec passed\n");
}
