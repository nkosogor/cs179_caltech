CS 179: GPU Computing
Lab 2
Name: Nikita Kosogorov


## PART 1

### Question 1.1: Latency Hiding

**Approximately how many arithmetic instructions does it take to hide the latency of a single arithmetic instruction on a GK110?**

- **Answer**: The key to latency hiding is to ensure that the GPU has enough independent instructions to execute while waiting for the results of previous instructions. For the NVIDIA GK110 architecture, the typical latency of an arithmetic instruction can be around 10 cycles. Given that the GK110 can issue a warp (32 threads) every clock cycle and assuming all instructions are independent, it effectively means that to hide the latency of one arithmetic instruction, we need roughly 10 warps or 320 instructions, assuming each thread executes one instruction in parallel.

### Question 1.2: Thread Divergence

Let the block shape be (32, 32, 1).

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

- **Answer**: Yes. If the threads within a warp have different `threadIdx.x` values, each thread might execute the loop a different number of times, leading to divergence. However, since the divergence here is due to a loop count rather than a branch condition, the impact might be less significant. All threads will eventually reconverge at the loop exit.

### Question 1.3: Coalesced Memory Access
Let the block shape be (32, 32, 1). Let data be a (float *) pointing to global
memory and let data be 128 byte aligned (so data % 128 == 0).

**(a)** Is this write coalesced?

```cpp
data[threadIdx.x + blockSize.x * threadIdx.y] = 1.0;
```

- **Answer**: Yes, this write is coalesced. Threads in a warp access consecutive memory locations. This pattern leads to efficient memory access with a single memory transaction per warp if aligned to memory boundaries.

How many 128 byte cache lines does this write to?

- **Answer**: Number of cache lines = Total bytes written / Cache line size 
Number of cache lines = 4096 bytes / 128 bytes/cache line = 32 cache lines (Total bytes written = 1024 threads * 4 bytes/thread = 4096 bytes)

**(b)** Is this write coalesced?

```cpp
data[threadIdx.y + blockSize.y * threadIdx.x] = 1.0;
```

- **Answer**: No, this write is not coalesced for warps along the `y` dimension, as it leads to strided accesses that are not consecutive in memory, causing multiple memory transactions.

How many 128 byte cache lines does this write to?

- **Answer**: Number of cache lines: 32 * 32 = 1024.

**(c)** Is this write coalesced?

```cpp
data[1 + threadIdx.x + blockSize.x * threadIdx.y] = 1.0;
```

- **Answer**: This access pattern is similar to (a), but with an offset of 1. It remains partially coalesced, but the starting address is shifted. 

How many 128 byte cache lines does this write to?

- **Answer**: Due to the 4-byte shift, each row might end up straddling two cache lines, effectively leading to additional cache line usage. In practice, this could mean up to 64 cache lines are accessed

### Question 1.4: Bank Conflicts and Instruction Dependencies

Let's consider multiplying a 32 x 128 matrix with a 128 x 32 element matrix.
This outputs a 32 x 32 matrix. We'll use 32 ** 2 = 1024 threads and each thread
will compute 1 output element. Although its not optimal, for the sake of
simplicity let's use a single block, so grid shape = (1, 1, 1),
block shape = (32, 32, 1).

For the sake of this problem, let's assume both the left and right matrices have
already been stored in shared memory are in column major format. This means the
element in the ith row and jth column is accessible at lhs[i + 32 * j] for the
left hand side and rhs[i + 128 * j] for the right hand side.

This kernel will write to a variable called output stored in shared memory.

Consider the following kernel code:

```cpp
int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
    output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
    output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
}
```

**(a)** Are there bank conflicts in this code?

- **Answer**: Yes, bank conflicts can occur because multiple threads may attempt to access the same bank simultaneously due to the pattern of access in shared memory (`output[i + 32 * j]`). Since `i` varies within a warp, accesses to `output` by threads in the same warp can target the same shared memory bank, leading to conflicts.

How many ways is the bank conflict (2-way, 4-way, etc)?

- **Answer**: Type of Conflict: Given this configuration, conflicts are likely to be 32-way since all threads in a single column of the 32x32 thread block (corresponding to different rows) are accessing the same bank simultaneously in a warp operation

**(b)** 
Expand the inner part of the loop (below)
```cpp
output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
```
into "psuedo-assembly" as was done in the coordinate addition example in lecture 4.

There's no need to expand the indexing math, only to expand the loads, stores,
and math. Notably, the operation a += b * c can be computed by a single
instruction called a fused multiply add (FMA), so this can be a single
instruction in your "psuedo-assembly".

Hint: Each line should expand to 5 instructions.

- **Answer**:
#### For the First Operation:
1. **LD**: Load `lhs[i + 32 * k]` into a register (`Rlhs`).
   - `LD Rlhs, [lhs + offset1]`
2. **LD**: Load `rhs[k + 128 * j]` into another register (`Rrhs`).
   - `LD Rrhs, [rhs + offset2]`
3. **LD**: Load `output[i + 32 * j]` into a register (`Rout`).
   - `LD Rout, [output + offset3]`
4. **FMA**: Execute the fused multiply-add instruction.
   - `FMA Rout, Rlhs, Rrhs, Rout`
5. **ST**: Store the result back to `output[i + 32 * j]`.
   - `ST [output + offset3], Rout`

#### For the Second Operation:
1. **LD**: Load `lhs[i + 32 * (k + 1)]` into a register (`Rlhs1`).
   - `LD Rlhs1, [lhs + offset4]`
2. **LD**: Load `rhs[(k + 1) + 128 * j]` into another register (`Rrhs1`).
   - `LD Rrhs1, [rhs + offset5]`
3. **LD**: Since `output[i + 32 * j]` is already in `Rout` and modified, we don't need to reload it.
4. **FMA**: Execute the fused multiply-add instruction with the new values.
   - `FMA Rout, Rlhs1, Rrhs1, Rout`
5. **ST**: Store the updated result back to `output[i + 32 * j]`.
   - `ST [output + offset3], Rout`

**offset1**, **offset2**, **offset3**, **offset4**, and **offset5** are placeholders for the computed memory addresses based on the indices i, j, k, and the respective matrix dimensions. These should be calculated based on the layout of your matrices in memory.


**(c)**
Identify pairs of dependent instructions in your answer to part b.

- **Answer**:

**For the First Operation:**
- **(LD Rout, [output + offset3]) and (FMA Rout, Rlhs, Rrhs, Rout)**: The FMA instruction depends on the value of `Rout` loaded from memory. This ensures that the addition in FMA uses the correct initial value of `output`.
- **(FMA Rout, Rlhs, Rrhs, Rout) and (ST [output + offset3], Rout)**: The store operation depends on the `Rout` register, which holds the result of the FMA operation. The result must be computed before it can be stored back into memory.

**For the Second Operation:**
- **(LD Rlhs1, [lhs + offset4]) and (FMA Rout, Rlhs1, Rrhs1, Rout)**: The FMA instruction depends on the `Rlhs1` value being correctly loaded from the `lhs` matrix.
- **(LD Rrhs1, [rhs + offset5]) and (FMA Rout, Rlhs1, Rrhs1, Rout)**: Similarly, the FMA instruction depends on the `Rrhs1` value being loaded from the `rhs` matrix.
- **(FMA Rout, Rlhs1, Rrhs1, Rout) and (ST [output + offset3], Rout)**: Again, the result of the FMA operation must be completed before it can be stored back into the memory.



**(d)**
Rewrite the code given at the beginning of this problem to minimize instruction
dependencies. You can add or delete instructions (deleting an instruction is a
valid way to get rid of a dependency!) but each iteration of the loop must still
process 2 values of k.

```cpp
int i = threadIdx.x;
int j = threadIdx.y;
float temp_output = 0.0;  // Local variable for accumulation

for (int k = 0; k < 128; k += 2) {
    // Accumulate results in a local variable to minimize global memory writes
    temp_output += lhs[i + 32 * k] * rhs[k + 128 * j];
    temp_output += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
}

// Write the accumulated result to the output array once after the loop
output[i + 32 * j] += temp_output;
```

**(e)**
Can you think of any other anything else you can do that might make this code
run faster?

 Reduce loop overhead by manually unrolling the loop. For example, processing more than 2 values of `k` within each iteration might be beneficial if it aligns well with the memory access patterns and computational capacity of the CUDA cores.

Utilize asynchronous memory copy (using asynchronous copy engines on newer GPUs) to prefetch data blocks into shared memory while computation is ongoing.

Experiment with different block sizes to find the optimal configuration for your specific GPU architecture. 

Use lower precision arithmetic (e.g., `float16` or mixed precision) where it does not affect the outcome significantly. 

## PART 2  - Matrix transpose optimization

Running:
```c 
./transpose
 ```
 Results in:
```c 
Index of the GPU with the lowest temperature: 1 (0 C)
Time limit for this program set to 10 seconds
Size 512 naive CPU: 1.117920 ms
Size 512 GPU memcpy: 0.025408 ms
Size 512 naive GPU: 0.047616 ms
Size 512 shmem GPU: 0.011424 ms
Size 512 optimal GPU: 0.012160 ms

Size 1024 naive CPU: 5.245472 ms
Size 1024 GPU memcpy: 0.048256 ms
Size 1024 naive GPU: 0.111968 ms
Size 1024 shmem GPU: 0.036416 ms
Size 1024 optimal GPU: 0.037248 ms

Size 2048 naive CPU: 49.426529 ms
Size 2048 GPU memcpy: 0.159040 ms
Size 2048 naive GPU: 0.382400 ms
Size 2048 shmem GPU: 0.136288 ms
Size 2048 optimal GPU: 0.137760 ms

Size 4096 naive CPU: 200.695679 ms
Size 4096 GPU memcpy: 0.548448 ms
Size 4096 naive GPU: 1.658016 ms
Size 4096 shmem GPU: 0.557760 ms
Size 4096 optimal GPU: 0.561728 ms
 ```

The performance of various matrix transpose implementations across different sizes shows that GPU-based methods significantly outperform the naive CPU implementation. The shared memory GPU approach shows a marked improvement over the naive GPU method due to optimized memory access patterns. However, the "optimal" GPU implementation, despite employing advanced techniques like loop unrolling and increased work per thread, does not demonstrate a significant performance gain over the shared memory version, suggesting that additional optimizations yield diminishing returns or are limited by other factors such as execution configuration or hardware constraints.


## Feedback

1.1) Need 80 instructions because there are 4 warps and 2 instructions per warp. This results in 8 instructions per clock cycle, with an instruction requiring 10 clocks. Therefore, we need 8 * 10 = 80 to hide the latency (refer to Lecture 5, slide 19).

1.3) Provides correct per block cache line writes but does not explicitly state whether the number provided is per warp or per block.

1.4.a) There is no bank conflict. Bank conflicts occur when multiple threads in a warp try to access different elements in the same bank. The different `i` in a warp for `output[i + 32 * j]` means that they have different banks.

1.4.b) During the 2nd operation, output gets reloaded.

1.4.c) FMA of the first operation depends on loading of lhs and rhs. Load of output in the 2nd operation depends on store in the 1st.

