#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <stdio.h>
#include "kernels/1_gemm_naive.cuh"
#include "kernels/2_gemm_global_coalesce.cuh"
#include "kernels/3_gemm_block_tiling.cuh"
#include "kernels/4_gemm_thread_tiling_1d.cuh"
#include "kernels/5_gemm_thread_tiling_2d.cuh"
#include "kernels/5.1_gemm_thread_tiling_2d_memopt.cuh"
#include "kernels/6_gemm_coalesce_store.cuh"
#include "utils.h"



// Compute GFLOPS utility
inline float compute_gflops(size_t M, size_t N, size_t K, float ms)
{
    double flops = 2.0 * M * N * K;                // 2 ops per FMA
    return static_cast<float>(flops / (ms * 1e6)); // GFLOPS
}

void launch_gemm_naive(const float *A, const float *B, float *C, int M, int K, int N)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));

    gemm_naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}

void launch_gemm_global_coalesce(const float *A, const float *B, float *C, int M, int K, int N)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));

    gemm_global_coalesce_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}

void launch_gemm_block_tilling(const float *A, const float *B, float *C, int M, int K, int N)
{   
    const int BS = 32; 
    dim3 threadsPerBlock(BS, BS);
    dim3 blocksPerGrid(CEIL_DIV(N, BS), CEIL_DIV(M, BS));
    gemm_block_tiling_kernel<BS><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}


void launch_gemm_thread_tiling_1d(const float* A, const float* B, float* C, int M, int K, int N) {
    // we set block_tile = A(BM * BK) & B(BK * BN)
    // we set thread tile = TM * 1 
    // we want each thread to load 1 data from B and 1 from A. 
    // # threads. 

    const int BM = 64, BN = 64, BK = 8, TM = 8;

    dim3 threadsPerBlock(BM * BN / TM); // (64 * 64 / 8) = 512
    dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    gemm_thread_tiling_1d_kernel<BM, BN, BK, TM><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
void launch_gemm_thread_tiling_2d(const float* A, const float* B, float* C, int M, int K, int N) {
    // we set block_tile = A(BM * BK) & B(BK * BN)
    // we set thread tile = TM * 1 
    // we want each thread to load 1 data from B and 1 from A. 
    // # threads. 
    // printf("dim: %d, %d\n", CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    dim3 threadsPerBlock(BN/TN, BM/TM); // (64 * 64 / 8) = 512
    dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    gemm_thread_tiling_2d_kernel<BM, BN, BK, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}

void launch_gemm_thread_tiling_2d_memopt(const float* A, const float* B, float* C, int M, int K, int N) {
    // we set block_tile = A(BM * BK) & B(BK * BN)
    // we set thread tile = TM * 1 
    // we want each thread to load 1 data from B and 1 from A. 
    // # threads. 
    // printf("dim: %d, %d\n", CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    dim3 threadsPerBlock(BN/TN, BM/TM); // (64 * 64 / 8) = 512
    dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    gemm_thread_tiling_2d_kernel_memopt<BM, BN, BK, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
void launch_gemm_global_coalesce_store(const float* A, const float* B, float* C, int M, int K, int N) {
    // we set block_tile = A(BM * BK) & B(BK * BN)
    // we set thread tile = TM * 1 
    // we want each thread to load 1 data from B and 1 from A. 
    // # threads. 
    // printf("dim: %d, %d\n", CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    dim3 threadsPerBlock(BN/TN, BM/TM); // (64 * 64 / 8) = 512
    dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    gemm_global_coalesce_store<BM, BN, BK, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}


void print_benchmark_result(const char* kernel_name, size_t size, float time_ms, float gflops)
{
    printf("%-25s | Size: %-5zu | Time: %10.4f ms | GFLOPS: %8.4f\n",
           kernel_name, size, time_ms, gflops);
}

void warmup() {
    size_t M = 256, N = 256, K = 256;
    float *A, *B, *C;
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));
    launch_gemm_naive(A, B, C, M, N, K);
    cudaDeviceSynchronize();
    cudaFree(A); cudaFree(B); cudaFree(C);
}
int main(int argc, char **argv)
{
    std::vector<size_t> sizes;
    
    if (argc > 1) {
        // size from command-line args
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(std::atoi(argv[i]));
        }
    } else {
        // default sizes
        sizes = {256, 512, 1024, 2048, 4096, 8192};
    }

    warmup();

    // stats
    float time_ms, gflops;
    

    // benchmark
    for (auto s : sizes)
    {
        size_t M = s, N = s, K = s;

        float *A, *B, *C;
        
        // init data
        cudaMallocManaged(&A, M * K * sizeof(float));
        cudaMallocManaged(&B, K * N * sizeof(float));
        cudaMallocManaged(&C, M * N * sizeof(float));

        for (size_t i = 0; i < M * K; i++)
            A[i] = 1.0f;
        for (size_t i = 0; i < K * N; i++)
            B[i] = 1.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // -- GEMM NAIVE --
        cudaEventRecord(start);
        launch_gemm_naive(A, B, C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time_ms, start, stop);
        gflops = compute_gflops(M, N, K, time_ms);
        print_benchmark_result("Naive", s, time_ms, gflops);

        // -- GEMM Global Coalesce --
        cudaEventRecord(start);
        launch_gemm_global_coalesce(A, B, C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time_ms, start, stop);
        gflops = compute_gflops(M, N, K, time_ms);
        print_benchmark_result("GlobalCoalesce", s, time_ms, gflops);

        // -- GEMM Block Tiling --
        cudaEventRecord(start);
        launch_gemm_block_tilling(A, B, C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time_ms, start, stop);
        gflops = compute_gflops(M, N, K, time_ms);
        print_benchmark_result("BlockTiling", s, time_ms, gflops);

        // -- GEMM Thread Tiling 1d --
        cudaEventRecord(start);
        launch_gemm_thread_tiling_1d(A, B, C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time_ms, start, stop);
        gflops = compute_gflops(M, N, K, time_ms);
        print_benchmark_result("ThreadTiling1D", s, time_ms, gflops);

        // // -- GEMM Thread Tiling 2d --
        cudaEventRecord(start);
        launch_gemm_thread_tiling_2d(A, B, C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time_ms, start, stop);
        gflops = compute_gflops(M, N, K, time_ms);
        print_benchmark_result("ThreadTiling2D", s, time_ms, gflops);

         // // -- GEMM Thread Tiling 2d --
        cudaEventRecord(start);
        launch_gemm_thread_tiling_2d_memopt(A, B, C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time_ms, start, stop);
        gflops = compute_gflops(M, N, K, time_ms);
        print_benchmark_result("ThreadTiling2DMemopt", s, time_ms, gflops);

        // // -- GEMM Thread Tiling 2d Global coalesce store --
        cudaEventRecord(start);
        launch_gemm_global_coalesce_store(A, B, C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time_ms, start, stop);
        gflops = compute_gflops(M, N, K, time_ms);
        print_benchmark_result("ThreadTiling2DGlobalCalesceStore", s, time_ms, gflops);

        printf("\n");
        // Free data
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    

    return 0;
}
