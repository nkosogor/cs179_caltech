#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // TODO: do not modify code, just comment on suboptimal accesses

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // TODO: Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.

    // __shared__ float data[???];
    __shared__ float tile[64][65]; // Using padding to avoid bank conflicts

    int x = blockIdx.x * 64 + threadIdx.x;
    int y = blockIdx.y * 64 + threadIdx.y;

    // Load input block into shared memory tile, coalescing global memory reads
    for (int i = 0; i < 64; i += 16) {
        tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * n + x];
    }
    __syncthreads();

    // Transpose block in shared memory
    x = blockIdx.y * 64 + threadIdx.x; // Transposed block's x coordinate
    y = blockIdx.x * 64 + threadIdx.y; // Transposed block's y coordinate

    // Store from shared memory tile to output, coalescing global memory writes
    for (int i = 0; i < 64; i += 16) {
        output[(y + i) * n + x] = tile[threadIdx.x][threadIdx.y + i];
    }

}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling.
    __shared__ float tile[64][65];

    int x = blockIdx.x * 64 + threadIdx.x;
    int y = blockIdx.y * 64 + (8 * threadIdx.y);  // Each thread handles 8 rows

    // Load data from global memory to shared memory, unroll loop explicitly
    if (x < n) {
        if (y < n && threadIdx.y * 8 < 64) {
            tile[threadIdx.y * 8][threadIdx.x] = input[y * n + x];
        }
        if (y + 1 < n && threadIdx.y * 8 + 1 < 64) {
            tile[threadIdx.y * 8 + 1][threadIdx.x] = input[(y + 1) * n + x];
        }
        if (y + 2 < n && threadIdx.y * 8 + 2 < 64) {
            tile[threadIdx.y * 8 + 2][threadIdx.x] = input[(y + 2) * n + x];
        }
        if (y + 3 < n && threadIdx.y * 8 + 3 < 64) {
            tile[threadIdx.y * 8 + 3][threadIdx.x] = input[(y + 3) * n + x];
        }
        if (y + 4 < n && threadIdx.y * 8 + 4 < 64) {
            tile[threadIdx.y * 8 + 4][threadIdx.x] = input[(y + 4) * n + x];
        }
        if (y + 5 < n && threadIdx.y * 8 + 5 < 64) {
            tile[threadIdx.y * 8 + 5][threadIdx.x] = input[(y + 5) * n + x];
        }
        if (y + 6 < n && threadIdx.y * 8 + 6 < 64) {
            tile[threadIdx.y * 8 + 6][threadIdx.x] = input[(y + 6) * n + x];
        }
        if (y + 7 < n && threadIdx.y * 8 + 7 < 64) {
            tile[threadIdx.y * 8 + 7][threadIdx.x] = input[(y + 7) * n + x];
        }
    }

    __syncthreads();

    // Calculate transposed output indices
    int trans_x = blockIdx.y * 64 + threadIdx.x;
    int trans_y = blockIdx.x * 64 + (8 * threadIdx.y);

    // Store data from shared memory to output, unroll loop explicitly
    if (trans_x < n) {
        if (trans_y < n && threadIdx.y * 8 < 64) {
            output[trans_y * n + trans_x] = tile[threadIdx.x][threadIdx.y * 8];
        }
        if (trans_y + 1 < n && threadIdx.y * 8 + 1 < 64) {
            output[(trans_y + 1) * n + trans_x] = tile[threadIdx.x][threadIdx.y * 8 + 1];
        }
        if (trans_y + 2 < n && threadIdx.y * 8 + 2 < 64) {
            output[(trans_y + 2) * n + trans_x] = tile[threadIdx.x][threadIdx.y * 8 + 2];
        }
        if (trans_y + 3 < n && threadIdx.y * 8 + 3 < 64) {
            output[(trans_y + 3) * n + trans_x] = tile[threadIdx.x][threadIdx.y * 8 + 3];
        }
        if (trans_y + 4 < n && threadIdx.y * 8 + 4 < 64) {
            output[(trans_y + 4) * n + trans_x] = tile[threadIdx.x][threadIdx.y * 8 + 4];
        }
        if (trans_y + 5 < n && threadIdx.y * 8 + 5 < 64) {
            output[(trans_y + 5) * n + trans_x] = tile[threadIdx.x][threadIdx.y * 8 + 5];
        }
        if (trans_y + 6 < n && threadIdx.y * 8 + 6 < 64) {
            output[(trans_y + 6) * n + trans_x] = tile[threadIdx.x][threadIdx.y * 8 + 6];
        }
        if (trans_y + 7 < n && threadIdx.y * 8 + 7 < 64) {
            output[(trans_y + 7) * n + trans_x] = tile[threadIdx.x][threadIdx.y * 8 + 7];
        }
    }
}


void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
