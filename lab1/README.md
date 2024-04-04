CS 179: GPU Computing
Lab 1: Introduction to CUDA
Name: Nikita Kosogorov

# Question 1

## 1.1 Incorrect Pointer Initialization

**Original Code:**
```c
void test1() {
    int *a = 3;
    *a = *a + 2;
    printf("%d\n", *a);
}
```

**Issue:** The pointer `a` is directly assigned an integer value of 3, which is incorrect. Pointers should be assigned the address of a memory location.

**Correction:**
```c
void test1() {
    int val = 3;
    int *a = &val;
    *a = *a + 2;
    printf("%d\n", *a);
}
```

## 1.2 Pointer and Variable Declaration Error

**Original Code:**
```c
void test2() {
    int *a, b;
    a = (int *) malloc(sizeof(int));
    b = (int *) malloc(sizeof(int));

    if (!(a && b)) {
        printf("Out of memory\n");
        exit(-1);
    }
    *a = 2;
    *b = 3;
}
```

**Issue:** `b` is declared as an `int` but is incorrectly treated as a pointer with `malloc`.

**Correction:**
```c
void test2() {
    int *a, *b;
    a = (int *) malloc(sizeof(int));
    b = (int *) malloc(sizeof(int));

    if (!(a && b)) {
        printf("Out of memory\n");
        exit(-1);
    }
    *a = 2;
    *b = 3;
    // Remember to free allocated memory
    free(a);
    free(b);
}
```

## 1.3 Incorrect Memory Allocation for an Array

**Original Code:**
```c
void test3() {
    int i, *a = (int *) malloc(1000);

    if (!a) {
        printf("Out of memory\n");
        exit(-1);
    }
    for (i = 0; i < 1000; i++)
        *(i + a) = i;
}
```

**Issue:** Memory allocation size is incorrect. It allocates 1000 bytes instead of 1000 integers.

**Correction:**
```c
void test3() {
    int i, *a = (int *) malloc(1000 * sizeof(int));

    if (!a) {
        printf("Out of memory\n");
        exit(-1);
    }
    for (i = 0; i < 1000; i++)
        a[i] = i;
    // Free allocated memory
    free(a);
}
```

## 1.4 Setting a 2D Array Element

**Original Code:**
```c
void test4() {
    int **a = (int **) malloc(3 * sizeof(int *));
    a[1][1] = 5;
}
```

**Issue:** Memory is allocated for pointers to `int` but not for the `int` values themselves.

**Correction:**
```c
void test4() {
    int **a = (int **) malloc(3 * sizeof(int *));
    for (int i = 0; i < 3; i++) {
        a[i] = (int *) malloc(100 * sizeof(int));
    }
    a[1][1] = 5;
    // Free allocated memory
    for (int i = 0; i < 3; i++) {
        free(a[i]);
    }
    free(a);
}
```

## 1.5 Misplaced Null Pointer Check

**Original Code:**
```c
void test5() {
    int *a = (int *) malloc(sizeof(int));
    scanf("%d", a);
    if (!a)
        printf("Value is 0\n");
}
```

**Issue:** The null pointer check is done after dereferencing `a`, which is not effective for checking if the input value is 0.

**Correction:**
```c
void test5() {
    int *a = (int *) malloc(sizeof(int));
    if (a) {
        scanf("%d", a);
        if (*a == 0)
            printf("Value is 0\n");
        // Free allocated memory
        free(a);
    } else {
        printf("Out of memory\n");
        exit(-1);
    }
}
```


# Question 2

## 2.1 GPU Implementation Comparison

For the given difference equations:

- \(y_1[n] = x[n - 1] + x[n] + x[n + 1]\)
- \(y_2[n] = y_2[n - 2] + y_2[n - 1] + x[n]\)

**Expected Easier and Faster Implementation on the GPU:**

The calculation for \(y_1[n]\) is expected to have an easier and faster implementation on the GPU. The primary reason is that \(y_1[n]\)'s computation for any \(n\) is independent of its other outputs, relying solely on the input signal \(x\). This characteristic allows for a highly parallel computation, where each thread in the GPU can independently calculate \(y_1[n]\) for different values of \(n\) without needing to synchronize or wait for other calculations to complete.

In contrast, \(y_2[n]\) depends on its previous two outputs (\(y_2[n - 2]\) and \(y_2[n - 1]\)). This dependency creates a sequential dependency chain that hampers parallelization because the calculation of \(y_2[n]\) cannot proceed without first computing \(y_2[n - 1]\) and \(y_2[n - 2]\), making it less suited for GPU's parallel processing capabilities.

## 2.2 Parallelizing Exponential Moving Average (EMA)

Given the Exponential Moving Average (EMA) formula:
- \(y[n] = c \cdot x[n] + (1 - c) \cdot y[n - 1]\)

Where \(c\) is close to 1, implying that \(1 - c\) is close to 0. When expanding the EMA's recurrence relation, the contribution of \(y[n - k]\) terms to \(y[n]\) decreases exponentially as \(k\) increases. Therefore, for a sufficiently large \(k\), \(y[n - k]\)'s impact on \(y[n]\) becomes negligible, allowing us to approximate \(y[n]\) without considering all previous \(y\) values.

**Approximation Method:**

To approximate \(y[n]\) in a way that is parallelizable, one can limit the recursion to a fixed number of steps, say \(m\), where \(m\) is chosen based on how close \(c\) is to 1 and the acceptable error margin of the approximation. This approach essentially truncates the exponential decay of the contribution from older \(y\) values, under the assumption that beyond \(m\) steps, the contributions are minimal.

**Pseudocode/Equation:**
```
1. Choose an appropriate m based on the desired accuracy.
2. For each n, calculate y[n] using only the last m values of y and x, according to:
   y[n] ≈ c * x[n] + (1 - c) * c * x[n-1] + ... + (1 - c)^(m-1) * c * x[n-m+1] + (1 - c)^m * y[n-m]
3. For parallelization, each thread calculates y[n] for different n independently, using the above approximation.
```

This method allows the approximation of \(y[n]\) to be calculated in parallel for different \(n\) values since the dependence on past values is limited to a fixed number \(m\), and these values can be precomputed or fetched efficiently in a parallel manner.




# Question 3: Small-Kernel Convolution 

See the code in `blur.cu`:

Running:
```c 
./audio-blur 256 1000 resources/example_test.wav output.wav
 ```
 Results in:
```c 
Normalized by factor of: 0.964579
gaussian[0] = 0.0111947
gaussian[1] = 0.0163699
gaussian[2] = 0.0229988
gaussian[3] = 0.0310452
gaussian[4] = 0.0402634
gaussian[5] = 0.0501713
gaussian[6] = 0.0600659
gaussian[7] = 0.0690923
gaussian[8] = 0.0763588
gaussian[9] = 0.0810805
gaussian[10] = 0.0827185
gaussian[11] = 0.0810805
gaussian[12] = 0.0763588
gaussian[13] = 0.0690923
gaussian[14] = 0.0600659
gaussian[15] = 0.0501713
gaussian[16] = 0.0402634
gaussian[17] = 0.0310452
gaussian[18] = 0.0229988
gaussian[19] = 0.0163699
gaussian[20] = 0.0111947
CPU blurring...
GPU blurring...
No kernel error detected
Comparing...

Successful output

CPU time: 77.2547 milliseconds
GPU time: 4.95018 milliseconds

Speedup factor: 15.6064

CPU blurring...
GPU blurring...
No kernel error detected
Comparing...

Successful output

CPU time: 72.7711 milliseconds
GPU time: 2.75318 milliseconds

Speedup factor: 26.4316
 ```