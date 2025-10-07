#include <cuda_runtime.h>

// naive gemm
// A(M * K) * B(K * N) = C(M * N)

__global__ gemm_naive_kernel(float* A, float* B, float* C, size_t M, size_t N, size_t K){
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        int val = 0;
        for (int i = 0; i < K; i ++) {
            val += A[m * K + i] * B[i * N + n];
        }
        C[m * M + n] = val;
    } 
}


void launch_gemm_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    
    gemm_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
