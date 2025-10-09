#include <cuda_runtime.h>
#define CEIL_DIV(numerator, denominator) (((numerator) + (denominator) - 1) / (denominator))
#define INDEX_2D(row, col, width) ((row) * (width) + col)


// Assumption: BM = BN, 
//             BK = BM / TM (each threads only load 1 element from both A and B)

#include <stdio.h>

template<const int BM,const int  BN, const int  BK, const int  TM>
__global__ void gemm_thread_tiling_1d_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    size_t n_offset = blockIdx.x * BN;
    size_t m_offset = blockIdx.y * BM;

    size_t tid = threadIdx.x;

    // load position & inner thread-tile index
    size_t a_row = tid / BK; 
    size_t a_col = tid % BK;
    size_t b_row = tid / BN;
    size_t b_col = tid % BN;
    size_t out_row_offset = (tid / BN) * TM;
    size_t out_col = tid % BN;

    // shared_mem
    __shared__ float shared_A[BM][BK];
    __shared__ float shared_B[BK][BN];

    // register
    float results[TM + 1] = {0.0};

    for (int k_offset = 0; k_offset < K; k_offset += BK) {
        // load 1 element from A
        if (m_offset + a_row < M && k_offset + a_col < K){
            shared_A[a_row][a_col] = A[INDEX_2D(m_offset + a_row, k_offset + a_col, K)];
            // printf("a: %f\n", shared_A[a_row][a_col]);
        } else {
            shared_A[a_row][a_col] = 0;
        }

        // load 1 element from B
        if (k_offset + b_row < K && n_offset + b_col < N){
            shared_B[b_row][b_col] = B[INDEX_2D(k_offset + b_row, n_offset + b_col, N)];
            // printf("b: %f\n", shared_B[b_row][b_col]);
        } else {
            shared_B[b_row][b_col] = 0;
        }

        __syncthreads();

        // calculate inner product for multiple result.
        // each thread calculate TM results.
        for (int j = 0; j < BK; j += 1) {  // Note: TM=BK
            float btmp = shared_B[j][out_col];
            for (int i = 0; i < TM; i += 1) {
                results[i] += shared_A[out_row_offset + i][j] * btmp;
            }
        }
        
        __syncthreads();

    }

    // Each thread write TM results
    size_t c_col = n_offset + out_col;
    for (int i = 0; i < TM; i += 1) {
        size_t c_row = m_offset + out_row_offset + i;
        if (c_row < M && c_col < N) {
            C[INDEX_2D(c_row, c_col, N)] = results[i];
        }
    }
    
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
// (M, K) (K, N) (M, N)
