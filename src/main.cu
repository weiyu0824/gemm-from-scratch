#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "kernels/1_gemm_naive.cuh"

// Macro
#define CEIL_DIV(numerator, denominator) (((numerator) + (denominator) - 1) / (denominator))

// Compute GFLOPS utility
inline float compute_gflops(size_t M, size_t N, size_t K, float ms) {
    double flops = 2.0 * M * N * K;  // 2 ops per FMA
    return static_cast<float>(flops / (ms * 1e6));  // GFLOPS
}

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    
    gemm_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}


int main() {
    std::vector<size_t> sizes = {128, 256, 512, 1024, 2048, 4096};

    for (auto s : sizes) {
        size_t M = s, N = s, K = s;

        float *A, *B, *C;
        cudaMallocManaged(&A, M * K * sizeof(float));
        cudaMallocManaged(&B, K * N * sizeof(float));
        cudaMallocManaged(&C, M * N * sizeof(float));

        for (size_t i = 0; i < M * K; i++) A[i] = 1.0f;
        for (size_t i = 0; i < K * N; i++) B[i] = 1.0f;

        // Warm-up
        launch_gemm_naive(A, B, C, M, N, K);
        cudaDeviceSynchronize();

        // Timing using CUDA events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        launch_gemm_naive(A, B, C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        float gflops = compute_gflops(M, N, K, ms);

        std::cout << "Naive GEMM | Size: " << s
                  << " | Time: " << ms << " ms"
                  << " | " << gflops << " GFLOPS"
                  << std::endl;

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
