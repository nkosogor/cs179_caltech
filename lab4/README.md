CS 179: GPU Computing
Lab 4
Name: Nikita Kosogorov

## Overview



###  Point-Cloud Alignment Project Overview

This lab aligns two 3D point clouds by computing a transformation matrix using CUDA technologies, specifically leveraging cuBLAS and cuSolver. The solution processes OBJ files to extract vertex data, applies matrix operations to determine the optimal alignment, and outputs the transformed 3D model for verification. The implementation enhances performance and accuracy in handling large datasets typical in 3D graphics processing.

### Results

```c 
./point_alignment resources/bunny.obj resources/noisy-bunny.obj outputs/noisy_bunny_output.obj
```
```c 
Aligning resources/bunny.obj with resources/noisy-bunny.obj
Reading resources/bunny.obj, which has 14290 vertices
Reading resources/noisy-bunny.obj, which has 14290 vertices
0.0085758 0.0185024 0.051359 0.912375 
0.0356903 0.078674 0.0786958 1.75671 
0.0140075 -0.0031318 -0.00200421 3.09388 
0 0 0 1 
```


```c
./point_alignment resources/bunny2.obj resources/bunny2_trans.obj outputs/bunny2_output.obj
```
```c
Aligning resources/bunny2.obj with resources/bunny2_trans.obj
Reading resources/bunny2.obj, which has 14290 vertices
Reading resources/bunny2_trans.obj, which has 14290 vertices
0.000398147 -6.75226e-11 -0.5 2.22572e-08 
3.69351e-08 0.5 1.00468e-08 1.50906e-08 
0.5 1.97539e-08 0.000398134 9.65071e-09 
0 0 0 1 
``` 
```c
./point_alignment resources/cube.obj resources/cube2.obj outputs/cube_output.obj
```
```c
Aligning resources/cube.obj with resources/cube2.obj
Reading resources/cube.obj, which has 8 vertices
Reading resources/cube2.obj, which has 8 vertices
2 0 0 0 
0 2 2.22045e-15 -4.47035e-08 
0 0 2 -3.97364e-08 
0 0 0 1 
```
```c
./point_alignment resources/tetrahedron2.obj resources/tetrahedron2_trans.obj outputs/tetrahedron_output.obj
```

```c
Aligning resources/tetrahedron2.obj with resources/tetrahedron2_trans.obj
Reading resources/tetrahedron2.obj, which has 4 vertices
Reading resources/tetrahedron2_trans.obj, which has 4 vertices
1 0 0 -2 
0 0.73169 -0.681639 3.57628e-07 
0 0.68164 0.731689 -3.17891e-07 
0 0 0 1 
```