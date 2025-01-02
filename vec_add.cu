#include <stdio.h>
#include <cuda.h>

#define N 10000

// CPU functions.
static void vec_add(float *out, float *x, float *y, int n);
static void print_vec(float *vec, int n);

// GPU functions.
__global__ static void vec_add_gpu(float *out, float *x, float *y, int n);

int main()
{
    float x[N];
    float y[N];
    float cpu_out[N];
    float gpu_out[N];

    for(int i = 0; i < N; i++)
    {
        x[i] = 19.0f;
        y[i] = 23.0f;
    }

    vec_add(cpu_out, x, y, N);
    printf("vec_add output CPU:\n");
    print_vec(cpu_out, 10);

    float *d_x;
    float *d_y;
    float *d_out;
    
    // Allocate device memory.
    cudaMalloc((void**)&d_x, sizeof(float) * N);
    cudaMalloc((void**)&d_y, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Copy CPU -> GPU.
    cudaMemcpy(d_x, x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Run 1 block, 1 thread per block.
    vec_add_gpu<<<1,1>>>(d_out, d_x, d_y, N);
    cudaDeviceSynchronize();

    // Copy result GPU -> CPU.
    cudaMemcpy(gpu_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    printf("vec_add output GPU:\n");
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
