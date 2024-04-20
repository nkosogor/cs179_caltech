CS 179: GPU Computing
Lab 3
Name: Nikita Kosogorov

## Overview


### Large-Kernel Convolution
Efficient convolution of audio signals using the Fast Fourier Transform (FFT), particularly suitable for large impulse responses. Utilizing FFT and the cuFFT library optimized for NVIDIA GPUs, this method significantly enhances computational efficiency:

X = FFT(x), H = FFT(h), Y = X * H, y = IFFT(Y)


### Normalization
Following convolution, the output signal may exceed standard amplitude levels, risking clipping and distortion. Normalization adjusts the signal within an acceptable range:

y_normalized = y / max(|y|)

This step involves a reduction operation to find the maximum amplitude and scales the entire signal accordingly.

### Results


Running:
```c 
./noaudio-fft 256 64
 ```

Results in:
 ```c 
 Index of the GPU with the lowest temperature: 1 (0 C)
Time limit for this program set to 180 seconds

N (number of samples per channel):    10000000

Impulse length (number of samples per channel):    2001

CPU convolution...
GPU convolution...
No kernel error detected
Comparing...

Successful output

CPU time (convolve): 60331.8 milliseconds
GPU time (convolve): 216.971 milliseconds

Speedup factor (convolution): 278.064


CPU normalization...
GPU normalization...
No kernel error detected
No kernel error detected

CPU normalization constant: 0.504522
GPU normalization constant: 0.504522

CPU time (normalization): 61.2293 milliseconds
GPU time (normalization): 1.15427 milliseconds

Speedup factor (normalization): 53.0458




CPU convolution...
GPU convolution...
No kernel error detected
Comparing...

Successful output

CPU time (convolve): 60276.4 milliseconds
GPU time (convolve): 160.268 milliseconds

Speedup factor (convolution): 376.098


CPU normalization...
GPU normalization...
No kernel error detected
No kernel error detected

CPU normalization constant: 0.502063
GPU normalization constant: 0.502063

CPU time (normalization): 61.4292 milliseconds
GPU time (normalization): 1.16538 milliseconds

Speedup factor (normalization): 52.712
 ```


```c
./audio-fft 256 32 resources/example_testfile.wav resources/silo_small.wav outputs/output1.wav
```

```c 
Index of the GPU with the lowest temperature: 1 (0 C)
Time limit for this program set to 180 seconds

N (number of samples per channel):    1375413

Impulse length (number of samples per channel):    1014


CPU convolution...
GPU convolution...
No kernel error detected
Comparing...

Successful output

CPU time (convolve): 4272.23 milliseconds
GPU time (convolve): 54.0685 milliseconds

Speedup factor (convolution): 79.0151


CPU normalization...
GPU normalization...
No kernel error detected
No kernel error detected

CPU normalization constant: 6.75525
GPU normalization constant: 6.75525

CPU time (normalization): 8.82518 milliseconds
GPU time (normalization): 0.323232 milliseconds

Speedup factor (normalization): 27.3029




CPU convolution...
GPU convolution...
No kernel error detected
Comparing...

Successful output

CPU time (convolve): 4207.89 milliseconds
GPU time (convolve): 28.1465 milliseconds

Speedup factor (convolution): 149.499


CPU normalization...
GPU normalization...
No kernel error detected
No kernel error detected

CPU normalization constant: 7.91163
GPU normalization constant: 7.91163

CPU time (normalization): 8.87706 milliseconds
GPU time (normalization): 0.30272 milliseconds

Speedup factor (normalization): 29.3243
```

```c 
./audio-fft 256 32 resources/example_testfile.wav resources/silo_small.wav outputs/output2.wav
```

```c 
Index of the GPU with the lowest temperature: 1 (0 C)
Time limit for this program set to 180 seconds

N (number of samples per channel):    1375413

Impulse length (number of samples per channel):    1014

CPU convolution...
GPU convolution...
No kernel error detected
Comparing...

Successful output

CPU time (convolve): 4289.6 milliseconds
GPU time (convolve): 54.4632 milliseconds

Speedup factor (convolution): 78.7614


CPU normalization...
GPU normalization...
No kernel error detected
No kernel error detected

CPU normalization constant: 6.75525
GPU normalization constant: 6.75525

CPU time (normalization): 8.65331 milliseconds
GPU time (normalization): 0.302304 milliseconds

Speedup factor (normalization): 28.6245




CPU convolution...
GPU convolution...
No kernel error detected
Comparing...

Successful output

CPU time (convolve): 4195.86 milliseconds
GPU time (convolve): 28.0481 milliseconds

Speedup factor (convolution): 149.595


CPU normalization...
GPU normalization...
No kernel error detected
No kernel error detected

CPU normalization constant: 7.91163
GPU normalization constant: 7.91163

CPU time (normalization): 8.66506 milliseconds
GPU time (normalization): 0.302144 milliseconds

Speedup factor (normalization): 28.6786
```