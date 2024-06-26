/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {


    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < padded_length) {
        cufftComplex a = raw_data[idx];
        cufftComplex b = impulse_v[idx];
        cufftComplex result;

        // Complex multiplication: (ac - bd) + (ad + bc)i
        result.x = a.x * b.x - a.y * b.y;
        result.y = a.x * b.y + a.y * b.x;

        // Scale the result by the padded length
        out_data[idx].x = result.x / padded_length;
        out_data[idx].y = result.y / padded_length;

        idx += blockDim.x * gridDim.x;
    }

}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding.

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */
    
    
    // Memory Allocation and Indexing: Load real part's absolute value or set to zero if out of bounds.
    // Parallel Reduction: Reduce in shared memory by halving the comparison range in each iteration.
    // Atomic Maximum Operation: Write block's maximum to global memory safely using atomicMax.

    

    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float max_val = 0;

    while (i < padded_length) {
        max_val = fmaxf(max_val, fabsf(out_data[i].x));
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = max_val;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Only thread 0 writes back to global mem
    if (tid == 0) {
        atomicMax(max_abs_val, sdata[0]);
    }


}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < padded_length && *max_abs_val != 0) {
        out_data[idx].x /= *max_abs_val;
        out_data[idx].y /= *max_abs_val;
        idx += blockDim.x * gridDim.x;
    }

}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* TODO: Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v, out_data, padded_length);

}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* TODO 2: Call the max-finding kernel. */
    cudaMaximumKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(out_data, max_abs_val, padded_length);


}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2: Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);

}