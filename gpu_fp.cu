// Simple experiment with function pointers.
#include <cstdio>
#include <cuda.h>

typedef float (*op_fn) (float, float);

__device__ float addf(float a, float b)
{
    std::printf("inside addf\n");
    return a + b;
}

__device__ float mulf(float a, float b)
{
    std::printf("inside mulf\n");
    return a * b;
}

// Static pointers to device functions.
__device__ op_fn p_addf = addf;
__device__ op_fn p_mulf = mulf;

__global__ void applyOp(op_fn op, float a, float b)
{
    float res = op(a, b);
    std::printf("op %10.5f %10.5f = %10.5f\n", a, b, res);
}

int main()
{
    op_fn add_op;
    op_fn mul_op;
    cudaMemcpyFromSymbol(&add_op, p_addf, sizeof(op_fn));
    cudaMemcpyFromSymbol(&mul_op, p_mulf, sizeof(op_fn));
    std::printf("About to apply\n");
    applyOp<<<1,1>>>(add_op, 3.0, 5.0);
    cudaDeviceSynchronize();
    applyOp<<<1,1>>>(mul_op, 3.0, 5.0);
    cudaDeviceSynchronize();
    return 0;
}
