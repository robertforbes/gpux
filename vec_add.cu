#include <stdio.h>
#include <cuda.h>
#include <chrono>

#define N 10000

// CPU functions.
static void vec_add(float *out, float *x, float *y, int n);
static void print_vec(float *vec, int n);

// GPU functions.
__global__ static void vec_add_gpu(float *out, float *x, float *y, int n);

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main()
{
    float x[N];
    float y[N];
    float cpu_out[N];
    float gpu_out[N];
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
    float *d_out;
    
    // Allocate device memory.
    cudaMalloc((void**)&d_x, sizeof(float) * N);
    cudaMalloc((void**)&d_y, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Copy CPU -> GPU.
    t0 = high_resolution_clock::now();
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * N, cudaMemcpyHostToDevice);
    t1 = high_resolution_clock::now();

    // Run 1 block, 1 thread per block.
    vec_add_gpu<<<1,1>>>(d_out, d_x, d_y, N);
    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();

    // Copy result GPU -> CPU.
    cudaMemcpy(gpu_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    auto t3 = high_resolution_clock::now();
   
    // Calculate GPU durations.
    duration<double, std::micro> d1, d2, d3, total;
    d1 = t1 - t0;
    d2 = t2 - t1;
    d3 = t3 - t2;
    total = t3 - t0;
    printf("vec_add GPU, cp1 %10.5fus, ex %10.5fus, cp2 %10.5fus, total %10.5fus, output :\n",
        d1.count(),
        d2.count(),
        d3.count(),
        total.count());
    print_vec(gpu_out, 10);

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
    for(int i = 0; i < n; i++)
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
