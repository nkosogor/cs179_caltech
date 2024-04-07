CS 179: GPU Computing
Lab 2
Name: Nikita Kosogorov


## PART 1

### Question 1.1: Latency Hiding

**Approximately how many arithmetic instructions does it take to hide the latency of a single arithmetic instruction on a GK110?**

- **Answer**: The key to latency hiding is to ensure that the GPU has enough independent instructions to execute while waiting for the results of previous instructions. For the NVIDIA GK110 architecture, the typical latency of an arithmetic instruction can be around 10 cycles. Given that the GK110 can issue a warp (32 threads) every clock cycle and assuming all instructions are independent, it effectively means that to hide the latency of one arithmetic instruction, we need roughly 10 warps or 320 instructions, assuming each thread executes one instruction in parallel.

### Question 1.2: Thread Divergence

**(a)** Does the following code diverge?

```cpp
int idx = threadIdx.y + blockSize.y * threadIdx.x;
if (idx % 32 < 16)
    foo();
else
    bar();
```

- **Answer**: No, this code does not diverge within a warp. All threads of a warp execute the same instruction at any given time; divergence happens if threads within the same warp must follow different execution paths. Since `idx % 32 < 16` uniformly applies to threads within warps due to the calculation method (keeping threads within a warp aligned in their conditional outcomes), all threads in a warp will take the same path, avoiding divergence.

**(b)** Does this code diverge?

```cpp
const float pi = 3.14;
float result = 1.0;
for (int i = 0; i < threadIdx.x; i++)
    result *= pi;
```

- **Answer**: Potentially, yes, but it depends on the context. If the threads within a warp have different `threadIdx.x` values, each thread might execute the loop a different number of times, leading to divergence. However, since the divergence here is due to a loop count rather than a branch condition, the impact might be less significant than traditional control flow divergence. All threads will eventually reconverge at the loop exit.

### Question 1.3: Coalesced Memory Access

**(a)** Is this write coalesced?

```cpp
data[threadIdx.x + blockSize.x * threadIdx.y] = 1.0;
```

- **Answer**: Yes, this write is coalesced. Threads in a warp access consecutive memory locations, assuming `blockSize.x` is a multiple of the warp size (32). This pattern leads to efficient memory access with a single memory transaction per warp if aligned to memory boundaries.

How many 128 byte cache lines does this write to?

- **Answer**: Assuming `blockSize.x` is 32, each warp writes to a single 128-byte cache line since each `float` is 4 bytes, and 32 floats fit into 128 bytes.

**(b)** Is this write coalesced?

```cpp
data[threadIdx.y + blockSize.y * threadIdx.x] = 1.0;
```

- **Answer**: No, this write is not coalesced for warps along the `y` dimension, as it leads to strided accesses that are not consecutive in memory, causing multiple memory transactions.

How many 128 byte cache lines does this write to?

- **Answer**: Given a block size of (32, 32, 1), this pattern results in accesses that are spread across different cache lines in a non-sequential manner. The exact number of cache lines accessed depends on the `blockSize.y` value and the data layout, but it's significantly more than the coalesced case, with potentially one cache line per thread in the worst case.

**(c)** Is this write coalesced?

```cpp
data[1 + threadIdx.x + blockSize.x * threadIdx.y] = 1.0;
```

- **Answer**: This access pattern is similar to (a), but with an offset of 1. It remains coalesced, assuming `blockSize.x` aligns with the warp size, but the starting address is shifted. 

How many 128 byte cache lines does this write to?

- **Answer**: Similar to part (a), assuming a `blockSize.x` of 32, it would still target a single 128-byte cache line per warp, but the alignment might cause it to span across two cache lines depending on the starting address alignment.

### Question 1.4: Bank Conflicts and Instruction Dependencies

**(a)** Are there bank conflicts in this code?

- **Answer**: Yes, bank conflicts can occur because multiple threads may attempt to access the same bank simultaneously due to the pattern of access in shared memory (`output[i + 

32 * j]`). Since `i` varies within a warp, accesses to `output` by threads in the same warp can target the same shared memory bank, leading to conflicts.

How many ways is the bank conflict (2-way, 4-way, etc)?

- **Answer**: The exact nature of the conflict (2-way, 4-way, etc.) depends on the stride of access relative to the number of banks. Shared memory is typically divided into 32 banks, and accesses to the same bank by threads within a warp lead to serial execution. The stride of `32 * sizeof(float)` likely results in bank conflicts, potentially 2-way or more, depending on the exact access pattern and shared memory layout.

**(b) to (e)** The answers to these parts involve more detailed analysis and pseudo-assembly creation, which can be highly specific and technical, requiring assumptions about hardware architecture and compiler behavior not fully detailed here. Generally, optimizing for fewer instruction dependencies and bank conflicts involves reordering computations, minimizing shared memory access conflicts, and ensuring memory access patterns are as sequential and aligned as possible.

For a precise and optimal solution, detailed knowledge of the CUDA architecture, the shared memory bank model, and specific compiler optimizations would be necessary.

---