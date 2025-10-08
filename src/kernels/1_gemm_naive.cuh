#include <cuda_runtime.h>

// naive gemm
// A(M * K) * B(K * N) = C(M * N)

__global__ void gemm_naive_kernel(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N)
    {
        int val = 0;
        for (int i = 0; i < K; i++)
        {
            val += A[m * K + i] * B[i * N + n];
        }
        C[m * M + n] = val;
    }
}
