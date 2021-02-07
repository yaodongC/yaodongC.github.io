---
layout: post
title: "Pytorch Tensor Basics"
subtitle:   A Beginner's Guid to Pytorch 01
date:       2019-06-15
author:     Yaodong Cui
header-img: img/pytorch-logo-dark.svg
header-mask: 0.5
catalog: true
tags:
    - pytorch
---


# What is Tensors

A mathematical term for n-dimensional arrrys (arrays, vectors, matrices, scalars).

## Rank, Shape and Axes

- Rank:  Indicates the number of dimensions/Axes of the array containded in the Tensor object.
- Shape: Indicates the number of values in each dimensions/Axes.
- Axes:  An axis of a tensor is a specific dimension of a tensor.

Note: scalars have rank [None], 1D vectors have rank [1], etc.

<br> In PyTorch use the '.size()' and '.shape' to check tensor's size, 'len(t.shape)' to check rank and '.numel()' to check number of elements.
```python
data = np.array([1,2,3])  create a numpy array, dtype=int32;
t    = torch.tensor(data) use the class constructor, dtype=torch.float32;
t.size()                  use size method to check tensor size;
t.shape                   use shape attribute to check tensor size;
len(t.shape)              check tensor's Rank;
t.numel()                 check number of elements;
```
<!-- ```python
data = np.array([1,2,3])  # create a numpy array, dtype=int32;
t    = torch.tensor(data) # use the class constructor, dtype=torch.float32;
t.size()                  # use size method to check tensor size;
t.shape                   # use shape attribute to check tensor size;
len(t.shape)             # check tensor's Rank;
t.numel()                # check number of elements;
``` -->
# Tensor in PyTorch

<br> PyTorch tensors are instances of the torch.Tensor Python class. A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.

## [Create Tensors with data](https://pytorch.org/docs/stable/torch.html#tensor-creation-ops)

Best Options:
- **'torch.tensor()'**: The go-to function to create new tensor.
- **'torch.as_tensor()'**: The go-to function to create new tensor with no copying and memory sharing.

### Create tensor by copying

<!-- ```python
data = np.array([1,2,3])    # create a numpy array, dtype=int32
t1 = torch.Tensor(data)     # use the class constructor, dtype=torch.float32
t2 = torch.tensor(data)     # use a factory function, dtype=torch.int32
t3 = torch.as_tensor(data)  # Convert the data into a torch.Tensor, no copy
t4 = torch.from_numpy(data) # Convert the numpy data into a torch.Tensor, no copy
``` -->
```python
data = np.array([1,2,3])    create a numpy array, dtype=int32
t1 = torch.Tensor(data)     use the class constructor, dtype=torch.float32
t2 = torch.tensor(data)     use a factory function, dtype=torch.int32
t3 = torch.as_tensor(data)  Convert the data into a torch.Tensor, no copy
t4 = torch.from_numpy(data) Convert the numpy data into a torch.Tensor, no copy
```

<br> Using the 'torch.Tensor()' class constructor create tensors with 'dtype=torch.float32', which could be overkill for certain cases. This is bevause the default dtype of torch.Tensor objects is 'float32'. Just avoid this method.

<br> `torch.tensor(data)` is a [factory function](https://en.wikipedia.org/wiki/Factory_(object-oriented_programming) that constructs `torch.Tensor` objects and returns them to the caller.  `torch.tensor(data)` choose a dtype based on the incoming data. This is called type inference. The 'dtype' is inferred based on the incoming data. Note that the 'dtype' and 'device' can also be explicitly set for these calls by specifying the 'dtype' as an argument:
```python
torch.tensor(data, dtype=torch.float32,device=torch.device('cuda:0'))
torch.as_tensor(data, dtype=torch.float32,device=torch.device('cpu'))
```
<br> A tensor can be created with 'requires_grad=True'so that 'torch.autograd' records operations on them for automatic differentiation.
```python
x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
```
<br> If you already have a **Tensor data** and want to avoid a copy, use 'torch.Tensor.requires_grad_()' or 'torch.Tensor.detach()'.
<br>'torch.Tensor.requires_grad_()'sets the tensor’s requires_grad attribute in-place.
```python
x = torch.tensor([[1., -1.], [1., 1.]])
x.requires_grad_()
```
<br>'torch.Tensor.detach()' detaches the Tensor from the graph that created it, making it a leaf.

### Create tensor by memory sharing

| Share Data          | Copy Data      |
| ------------------- | -------------- |
|  |
| torch.as\_tensor()  | torch.tensor() |
| torch.from\_numpy() | torch.Tensor() |

<br> Pytorch create a new `torch.Tensor` object without copying the data. In Pytorch C++ implementation, it just use the same memory pointer of the numpy data. This also means **the actual data of these numpy and torch objects in memory exists in a single place**. **And if you change the numpy data, the content (array) of the `torch.Tensor` object also changes**.

<br> Sharing data is **more efficient** and **uses less memory** than copying data because the data is not written to two locations in memory. However, it could be dangerous as data can be changed by accident.

<br> The 'torch.from_numpy()' function only accepts numpy.ndarrays, while the 'torch.as_tensor()' function accepts a wide variety of array-like objects including other PyTorch tensors. For this reason, 'torch.as_tensor()' is the winning choice in the memory sharing game.


## Tensor dtype


<br> Torch defines 10 tensor types with CPU and GPU variants. One thing to keep in mind about tensor data types is that tensor operations between tensors must happen between tensors with the same type of data.

| Data type                                                                  | dtype                                 | CPU tensor                                                                                            | GPU tensor                  |
| -------------------------------------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------- |
| 32-bit floating point                                                      | `torch.float32` or `torch.float`      | `torch.FloatTensor`                                                                                   | `torch.cuda.FloatTensor`    |
| 64-bit floating point                                                      | `torch.float64` or `torch.double`     | `torch.DoubleTensor`                                                                                  | `torch.cuda.DoubleTensor`   |
| 16-bit floating point [1](https://pytorch.org/docs/1.7.1/tensors.html#id3) | `torch.float16` or `torch.half`       | `torch.HalfTensor`                                                                                    | `torch.cuda.HalfTensor`     |
| 16-bit floating point [2](https://pytorch.org/docs/1.7.1/tensors.html#id4) | `torch.bfloat16`                      | `torch.BFloat16Tensor`                                                                                | `torch.cuda.BFloat16Tensor` |
| 32-bit complex                                                             | `torch.complex32`                     |                                                                                                       |                             |
| 64-bit complex                                                             | `torch.complex64`                     |                                                                                                       |                             |
| 128-bit complex                                                            | `torch.complex128` or `torch.cdouble` |                                                                                                       |                             |
| 8-bit integer (unsigned)                                                   | `torch.uint8`                         | `torch.ByteTensor`                                                                                    | `torch.cuda.ByteTensor`     |
| 8-bit integer (signed)                                                     | `torch.int8`                          | `torch.CharTensor`                                                                                    | `torch.cuda.CharTensor`     |
| 16-bit integer (signed)                                                    | `torch.int16` or `torch.short`        | `torch.ShortTensor`                                                                                   | `torch.cuda.ShortTensor`    |
| 32-bit integer (signed)                                                    | `torch.int32` or `torch.int`          | `torch.IntTensor`                                                                                     | `torch.cuda.IntTensor`      |
| 64-bit integer (signed)                                                    | `torch.int64` or `torch.long`         | `torch.LongTensor`                                                                                    | `torch.cuda.LongTensor`     |
| Boolean                                                                    | `torch.bool`                          | [`torch.BoolTensor`](https://pytorch.org/docs/1.7.1/tensors.html#torch.BoolTensor "torch.BoolTensor") | `torch.cuda.BoolTensor`     |

## Tensor operations

The high-level categories of tensor operations:

- Reshaping operations
- Element-wise operations
- Reduction operations
- Access operations

### Reshaping operations

'.reshape()'

'.squeeze()' and '.unsqueeze()'
