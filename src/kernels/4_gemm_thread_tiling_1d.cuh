#include <cuda_runtime.h>

#define TILE_SIZE 32
#define INDEX_2D(row, col, width) (row * (width) + col) // row-major matrix
#define THREAD_TILE_SIZE 4                              // compute 4*4=16 output

// -------- Kernel Launch Assumption -------

// A(M * K) * B(K * N) = C(M * N)

// block tiling gemm
// number of thread in a block = TILE_SIZE * TILE_SIZE

// ~100kb float=16bits=2bytes.
// Load all: 32 * K * 2bytes * 2(A&B) -> bounds by K.
// Load tile: 32 * 32 * 2bytes * 2(A&B) = 4 kb

__global__ void gemm_thread_tiling_1d_kernel(const float *A, const float *B, float *C, int M, int K, int N)
{
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    size_t m = blockIdx.y * blockDim.y + threadIdx.y;

    // inner tile index
    size_t tn = threadIdx.x;
    size_t tm = threadIdx.y;

    __shared__ float tile_A[BLOCK_TILE_SIZE][BLOCK_TILE_SIZE];
    __shared__ float tile_B[BLOCK_TILE_SIZE][BLOCK_TILE_SIZE];

    float val = 0;
  
    for (int ti = 0; ti < CEIL_DIV(K, BLOCK_TILE_SIZE); ti++)
    {
        tile_A[tm][tn] = 0;
        tile_B[tm][tn] = 0;

        // load tile of data from gmem to smem.
        for (int i = 0; i < THREAD_TILE_SIZE; i += 1)
        {
            tn = threadIdx.x * THREAD_TILE_SIZE + i;
            for (int j = 0; j < THREAD_TILE_SIZE; j += 1)
            {
                tm = threadIdx.x * THREAD_TILE_SIZE + j;

                if (m < M && ti * BLOCK_TILE_SIZE + tn < K)
                {
                    tile_A[tm][tn] = A[INDEX_2D(m, (ti * BLOCK_TILE_SIZE + tn), K)];
                    // printf("A %d, %d, %f\n", m, n, tile_A[tm][tn]);
                }

                if (ti * BLOCK_TILE_SIZE + tm < K && n < N)
                {
                    tile_B[tm][tn] = B[INDEX_2D((ti * BLOCK_TILE_SIZE + tm), n, N)];
                    // printf("B %d, %d, %f\n", m, n, tile_B[tm][tn]);
                }
            }
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
    if (m < M && n < N)
        C[INDEX_2D(m, n, N)] = val;
}
