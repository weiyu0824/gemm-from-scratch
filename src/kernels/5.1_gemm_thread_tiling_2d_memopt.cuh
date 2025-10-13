#include <cuda_runtime.h>
#include "../utils.h"

//#define BM 128
//#define BN 128
//#define BK 8
//#define TM 8
//#define TN 8 
// num_thread = 128*128/(8*8) = 16*16
// num_load = 128 * 8 / (16*16) = 4 
// Load=4 times, Compute=64 times.


// Assumption: BM = BN, 
//             BK = BM / TM (each threads only load 1 element from both A and B)

#include <stdio.h>

template<const int  BM, const int  BN,const int  BK,const int  TM,const int  TN>
__global__ void gemm_thread_tiling_2d_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int n_offset = blockIdx.x * BN;
    int m_offset = blockIdx.y * BM;

    int tn = threadIdx.x;
    int tm = threadIdx.y;

    int tid = tm * blockDim.x + tn;
    int num_threads = blockDim.x * blockDim.y;

    // inner thread-tile index
    int out_row_offset = tm * TM;
    int out_col_offset = tn * TN;

    // shared_mem
    // Note: adding padding to avoid bank conflicts.
    // __shared__ float shared_A[BM][BK + 1]; // method 1

    // Note
    __shared__ float shared_A[BK][BM];
    __shared__ float shared_B[BK][BN];

    // register
    float atmp[TM] = {0.0};
    float btmp[TN] = {0.0};
    float results[TM][TN] = {0.0};

    for (int k_offset = 0; k_offset < K; k_offset += BK) {
        // load num_load_a element from A
        for (int la = tid; la < BM * BK; la += num_threads) {
            int a_row = la / BK;
            int a_col = la % BK;
            if (m_offset + a_row < M && k_offset + a_col < K){
                shared_A[a_col][a_row] = A[INDEX_2D(m_offset + a_row, k_offset + a_col, K)];
                // printf("A: %d - %f in la=%d,r=%d,c=%d\n", tid, shared_A[a_row][a_col], la, a_row, a_col);
            } else {
                shared_A[a_col][a_row] = 0;
            }
        }
        

        // load num_load_b element from B
        for (int lb = tid; lb < BK * BN; lb += num_threads) {
            int b_row = lb / BN;
            int b_col = lb % BN;
            if (k_offset + b_row < K && n_offset + b_col < N){
                shared_B[b_row][b_col] = B[INDEX_2D(k_offset + b_row, n_offset + b_col, N)];
                // printf("B: %d - %f in lb=%d,r=%d,c=%d\n", tid, shared_B[b_row][b_col], lb, b_row, b_col);

            } else {
                shared_B[b_row][b_col] = 0;
            }
        }
        

        __syncthreads();

        // calculate inner product for multiple result.
        // each thread calculate TM results.
        for (int j = 0; j < BK; j += 1) {  // Note: TM=BK
            for (int i = 0; i < TM; i += 1) {
                atmp[i] = shared_A[j][out_row_offset + i];
            }
            for (int i = 0; i < TN; i += 1) {
                btmp[i] = shared_B[j][out_col_offset + i];
            }
            for (int r = 0; r < TM; r += 1) {
                for (int c = 0; c < TN; c += 1) {
                    results[r][c] +=  atmp[r] * btmp[c];
                    // if (tid == 0 && r < 2 && c < 2){
                    //     printf("%f += %f * %f\n", results[r][c], atmp[r], btmp[c]);
                    // }
                }
            }
        }
        
        __syncthreads();

    }

    // Each thread write TM * TN results
    for (int r = 0; r < TM; r += 1) {
        size_t c_row = m_offset + out_row_offset + r;
        for (int c = 0; c < TN; c += 1) {
            size_t c_col = n_offset + out_col_offset + c;
            if (c_row < M && c_col < N) {
                
                C[INDEX_2D(c_row, c_col, N)] = results[r][c];
            }
        }
    }
    
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
// (M, K) (K, N) (M, N)
