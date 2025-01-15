#include <cstdio>
#include <chrono>
#include <vector>

#include <cuda.h>

#define N 1048576

// CPU functions.
static void vec_add(float *out, float *x, float *y, int n);
static void print_vec(float *vec, int n);
static void check_vec(float *vec, float *ref, float thresh, int n);

// GPU functions.
__global__ static void vec_add_gpu(float *out, float *x, float *y, int n);

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main()
{
    // Allocate CPU memory.
    std::vector<float> x(N);
    std::vector<float> y(N);
    std::vector<float> cpu_out(N);
    std::vector<float> gpu_out0(N);
    std::vector<float> gpu_out1(N);
    std::vector<float> gpu_out2(N);
    std::vector<float> gpu_out3(N);


    duration<double, std::micro> us;

    // Initialise test vectors.
    for(int i = 0; i < N; i++)
    {
        x[i] = 19.0f;
        y[i] = 23.0f;
    }

    // Run a reference CPU computation and time the execution.
    auto t0 = high_resolution_clock::now();
    vec_add(cpu_out.data(), x.data(), y.data(), N);
    auto t1 = high_resolution_clock::now();
    us = t1 - t0;

    std::printf("vec_add CPU, %10.5fus, output:\n", us.count());
    print_vec(cpu_out.data(), 10);

    float *d_x;
    float *d_y;
    float *d_out0;
    float *d_out1;
    float *d_out2;
    float *d_out3;
    
    // Allocate device memory.
    cudaMalloc((void**)&d_x, sizeof(float) * N);
    cudaMalloc((void**)&d_y, sizeof(float) * N);
    cudaMalloc((void**)&d_out0, sizeof(float) * N);
    cudaMalloc((void**)&d_out1, sizeof(float) * N);
    cudaMalloc((void**)&d_out2, sizeof(float) * N);
    cudaMalloc((void**)&d_out3, sizeof(float) * N);

    // Copy CPU -> GPU.
    t0 = high_resolution_clock::now();
    cudaMemcpy(d_x, x.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    t1 = high_resolution_clock::now();

    // Run 1 block, 1 thread per block.
    vec_add_gpu<<<1,1>>>(d_out0, d_x, d_y, N);
    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();

    // Run 1 block, 128 threads per block.
    vec_add_gpu<<<1,128>>>(d_out1, d_x, d_y, N);
    cudaDeviceSynchronize();
    auto t3 = high_resolution_clock::now();

    // Run 128 blocks, 1024 threads per block.
    vec_add_gpu<<<128,1024>>>(d_out2, d_x, d_y, N);
    cudaDeviceSynchronize();
    auto t4 = high_resolution_clock::now();

    // Run 4096 blocks, 256 threads per block.
    vec_add_gpu<<<4096,256>>>(d_out3, d_x, d_y, N);
    cudaDeviceSynchronize();
    auto t5 = high_resolution_clock::now();

    // Copy results GPU -> CPU.
    cudaMemcpy(gpu_out0.data(), d_out0, sizeof(float) * N, cudaMemcpyDeviceToHost);
    auto t6 = high_resolution_clock::now();
    cudaMemcpy(gpu_out1.data(), d_out1, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_out2.data(), d_out2, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_out3.data(), d_out3, sizeof(float) * N, cudaMemcpyDeviceToHost);
   
    // Calculate GPU durations.
    duration<double, std::micro> d1, d2, d3, d4, d5, d6;
    d1 = t1 - t0;
    d2 = t2 - t1;
    d3 = t3 - t2;
    d4 = t4 - t3;
    d5 = t5 - t4;
    d6 = t6 - t5;
    std::printf("vec_add GPU, cp1 %10.5fus, ex1 %10.5fus, ex2 %10.5fus, ex3 %10.5fus, ex4 %10.5fus, cp2 %10.5fus, output:\n",
        d1.count(),
        d2.count(),
        d3.count(),
        d4.count(),
        d5.count(),
        d6.count());

    // Dump out part of the vectors, and run a check against the reference.
    print_vec(gpu_out0.data(), 10);
    check_vec(gpu_out0.data(), cpu_out.data(), 0.00001, N);
    print_vec(gpu_out1.data(), 10);
    check_vec(gpu_out1.data(), cpu_out.data(), 0.00001, N);
    print_vec(gpu_out2.data(), 10);
    check_vec(gpu_out2.data(), cpu_out.data(), 0.00001, N);
    print_vec(gpu_out3.data(), 10);
    check_vec(gpu_out3.data(), cpu_out.data(), 0.00001, 1000);

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
    auto block_len = n / gridDim.x;
    auto slice_len = n / (gridDim.x * blockDim.x);
    int start = blockIdx.x * block_len + threadIdx.x * slice_len;
    int end = start + slice_len;
    if(end > n)
    {
        end = n;
    }

    for(int i = start; i < end; i++)
    {
        out[i] = x[i] + y[i];
    }
}

static void print_vec(float *vec, int n)
{
    for(int i = 0; i < n; i++)
    {
        std::printf("%10.5f ", vec[i]);
    }
    std::printf("\n");
}

static void check_vec(float *vec, float *ref, float thresh, int n)
{
    for(int i = 0; i < n; i++)
    {
        float diff = fabs(vec[i] - ref[i]);
        if(diff > thresh)
        {
            std::printf("check_vec failed at index %d, vec %10.5f, ref %10.5f, diff %10.5f, thresh %10.5f\n",
                i,
                vec[i],
                ref[i],
                diff,
                thresh);
            return;
        }
    }
    std::printf("check_vec passed\n");
}
