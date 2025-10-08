#include <cuda_runtime.h>

// -------- Kernel Launch Assumption -------
// Optimization: BlockDim need to changed. Kernel code remain the same.

// Matrix A, B, C is all row major.
// BlockDim is set to (N/32, M/32), which means that
//  n = blockIdx.x * blockDim.x + threadIdx.x
//  m = blockIdx.y * blockDim.y + threadIdx.y
//  In CUDA world, threadsIdx would be "linearized" x->y->z.
//  so, (0, 0) (0, 1) (0, 2) ... (0, 3) would be formed into a "warp".
//

// A(M * K) * B(K * N) = C(M * N)
__global__ void gemm_global_coalesce_kernel(const float *A, const float *B, float *C, size_t M, size_t N, size_t K)
{
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    size_t m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N)
    {
        int ss = 0;
        for (int i = 0; i < K; i++)
        {
            ss += A[m * K + i] * B[i * N + n];
        }
        C[m * M + n] = ss;
    }
}