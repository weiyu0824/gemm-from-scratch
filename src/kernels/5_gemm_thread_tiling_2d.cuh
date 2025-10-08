#include <cuda_runtime.h>

#define INDEX_2D(row, col, width) (row * (width) + col) // row-major matrix
#define BLOCK_TILE 64
#define THREAD_TILE_H 8 // each thread compute 8 results.

// -------- Kernel Launch Assumption -------

// A(M * K) * B(K * N) = C(M * N)

// block tiling gemm
// number of thread in a block =

// ~100kb float=16bits=2bytes.
// Load all: 32 * K * 2bytes * 2(A&B) -> bounds by K.
// Load tile: 32 * 32 * 2bytes * 2(A&B) = 4 kb

__global__ void gemm_thread_tiling_2d_kernel(const float *A, const float *B, float *C, int M, int K, int N)
{
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    size_t m = blockIdx.y * blockDim.y + threadIdx.y;

    // inner tile index
    size_t tn = threadIdx.x;
    size_t tm = threadIdx.y;

    __shared__ float tile_A[BLOCK_TILE][THREAD_TILE_H];
    __shared__ float tile_B[THREAD_TILE_H][BLOCK_TILE];

    for (int ti = 0; ti < CEIL_DIV(K, ); ti++)
    {
        tile_A[tm][tn] = 0;
        tile_B[tm][tn] = 0;

        if (m < M && ti * BLOCK_TILE_SIZE + tn < K)
        {
            tile_A[tm][tn] = A[INDEX_2D(m, (ti * BLOCK_TILE_SIZE + tn), K)];
        }

        if (ti * BLOCK_TILE_SIZE + tm < K && n < N)
        {
            tile_B[tm][tn] = B[INDEX_2D((ti * BLOCK_TILE_SIZE + tm), n, N)];
        }

        __syncthreads();

        // inner product inside tile.
        for (int i = 0; i < BLOCK_TILE_SIZE; i++)
        {
            val += tile_A[tm][i] * tile_B[i][tn];
        }
        __syncthreads();
        // if (m == 0) {
        //     printf("%d, %d, %f, %f\n", m, n, tile_A[0][0], tile_B[0][0]);
        // }
    }
    for (int i = 0;)
        if (m < M && n < N)
            C[INDEX_2D(m, n, N)] = val;
}
