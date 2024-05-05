CS 179: GPU Computing
Lab 5 and 6
Name: Nikita Kosogorov

## Overview

### Neural Network Library with CUDA


This lab comprises a neural network library designed for constructing and operating fully connected (dense) and convolutional neural networks for classification tasks. Utilizing cuBLAS and cuDNN, the library provides efficient execution of both forward and backward passes. Labs 5 and 6 focus on implementing this library to handle the MNIST dataset of handwritten digits. The network supports operations such as adding layers, training, predicting, and evaluating model performance across both types of network architectures.

## Compilation
Compile the project using the provided Makefile:
```bash
make clean all
```
This will generate two executables: `bin/dense-neuralnet` for Lab 5 and `bin/conv-neuralnet` for Lab 6.

## Results
```bash
bin/dense-neuralnet --dir /srv/cs179_mnist
```

```
Image Magic        :803                            2051
Image Count        :EA60                           60000
Image Rows         :1C                              28
Image Columns      :1C                              28
Label Magic        :801                            2049
Label Count        :EA60                           60000
Loaded training set.
Predicting on 10 classes.
Epoch 1
--------------------------------------------------------------
Loss: 1.51776,  Accuracy: 0.539583
...

Epoch 25
--------------------------------------------------------------
Loss: 0.262031, Accuracy: 0.926316

Image Magic        :803                            2051
Image Count        :2710                           10000
Image Rows         :1C                              28
Image Columns      :1C                              28
Label Magic        :801                            2049
Label Count        :2710                           10000
Loaded test set.
Validation
----------------------------------------------------
Loss: 0.262224, Accuracy: 0.9253
```

```bash
 bin/conv-neuralnet --dir /srv/cs179_mnist
```

```
Image Magic        :803                            2051
Image Count        :EA60                           60000
Image Rows         :1C                              28
Image Columns      :1C                              28
Label Magic        :801                            2049
Label Count        :EA60                           60000
Loaded training set.
Predicting on 10 classes.
Epoch 1
--------------------------------------------------------------
Loss: 1.12989,  Accuracy: 0.6529

...

Epoch 25
--------------------------------------------------------------
Loss: 0.0749583,        Accuracy: 0.977417

Image Magic        :803                            2051
Image Count        :2710                           10000
Image Rows         :1C                              28
Image Columns      :1C                              28
Label Magic        :801                            2049
Label Count        :2710                           10000
Loaded test set.
Validation
----------------------------------------------------
Loss: 0.0750352,        Accuracy: 0.9766
```

