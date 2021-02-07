---
layout: post
title: "Pytorch  Object Oriented Programming"
subtitle:   A Beginner's Guid to Pytorch 02
date:       2019-07-20
author:     Yaodong Cui
header-img: img/pytorch-logo-dark.svg
header-mask: 0.5
catalog: true
tags:
    - pytorch
---

# Pytorch :

* [Prepare the data](#Prepare the data)
* [Build the model](#Build the model)
* [Train the model](#Train the model)
* [Analyze the model's results](#Analyze the model's results)


| Package                | Description                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
|  |
| torch                  | The top-level PyTorch package and tensor library.                                                                               |
| torch.nn               | A subpackage that contains modules and extensible classes for building neural networks.                                         |
| torch.optim            | A subpackage that contains standard optimization operations like SGD and Adam.                                                  |
| torch.nn.functional    | A functional interface that contains typical operations used for building neural networks like loss functions and convolutions. |
| torchvision            | A package that provides access to popular datasets, model architectures, and image transformations for computer vision.         |
| torchvision.transforms | An interface that contains common transforms for image processing.                                                              |

## Prepare the data

- Extract – Get the data from the source.
- Transform – Put our data into tensor form.
- Load – Put our data into an object to make it easily accessible.

For these purposes, PyTorch provides us with two classes:

| Class                       | Description                                                           |
| --------------------------- | --------------------------------------------------------------------- |
|  |
| torch.utils.data.Dataset    | An abstract class for representing a (Python iterable over) dataset.  |
| torch.utils.data.DataLoader | Wraps a dataset and provides access to the underlying data.           |

An abstract class is a Python class that has methods we must implement, so we can create a custom dataset by creating a subclass that extends the functionality of the Dataset class.

To create a custom dataset using PyTorch, we extend the Dataset class by creating a subclass that implements these required methods. Upon doing this, our new subclass can then be passed to the a PyTorch DataLoader object.

### Dataset Types
The most important argument of DataLoader constructor is dataset, which indicates a dataset object to load data from. PyTorch supports two different types of datasets:
- [map-style datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets)
- [iterable-style datasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets)

#### Map-style datasets: ['torch.utils.data.Dataset'](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

A map-style dataset is one that implements the '__getitem__()' and '__len__()' protocols, and represents a map from (possibly non-integral) indices/keys to data samples.

For example, such a dataset, when accessed with 'dataset[idx]', could read the 'idx-th' image and its corresponding label from a folder on the disk.

#### Iterable-style datasets: ['torch.utils.data.IterableDataset'](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)

An iterable-style dataset is an instance of a subclass of 'IterableDataset' that implements the' __iter__()' protocol, and represents an iterable over data samples. This type of datasets is particularly suitable for cases where **random reads are expensive or even improbable (large datasets)**, and where the batch size depends on the fetched data.

For example, such a dataset, when called 'iter(dataset)', could return a stream of data reading from a database, a remote server, or even logs generated in real time.



## Build the model



- Create a neural network class that extends the nn.Module base class.
- In the class constructor, define the network’s layers as class attributes using pre-built layers from torch.nn.
- Use the network’s layer attributes as well as operations from the nn.functional API to define the network’s forward pass.


### The basic building block: 'torch.nn' package

'torch.nn.Module' is b the base class for all neural network modules. Your models/layers should also subclass this class.

```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
```

```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)
        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

      return t`
```

#






## Container classes
This is an abstract Module class which declares methods defined in all containers. It reimplements many of the Module methods such that calls are propagated to the contained modules.


| [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")                             | Base class for all neural network modules. |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| [`Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential "torch.nn.Sequential")             | A sequential container.                    |
| [`ModuleList`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#torch.nn.ModuleList "torch.nn.ModuleList")             | Holds submodules in a list.                |
| [`ModuleDict`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict "torch.nn.ModuleDict")             | Holds submodules in a dictionary.          |
| [`ParameterList`](https://pytorch.org/docs/stable/generated/torch.nn.ParameterList.html#torch.nn.ParameterList "torch.nn.ParameterList") | Holds parameters in a list.                |
| [`ParameterDict`](https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html#torch.nn.ParameterDict "torch.nn.ParameterDict") | Holds parameters in a dictionary.          |


<br> An example of 'torch.nn.Sequential()' class


```python
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
```

Modules will be added to it in the order they are passed in the constructor.

## Convolution Layers

| [`nn.Conv1d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d "torch.nn.Conv1d")                                     | Applies a 1D convolution over an input signal composed of several input planes.                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d "torch.nn.Conv2d")                                     | Applies a 2D convolution over an input signal composed of several input planes.                    |
| [`nn.Conv3d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d "torch.nn.Conv3d")                                     | Applies a 3D convolution over an input signal composed of several input planes.                    |
| [`nn.ConvTranspose1d`](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html#torch.nn.ConvTranspose1d "torch.nn.ConvTranspose1d") | Applies a 1D transposed convolution operator over an input image composed of several input planes. |
| [`nn.ConvTranspose2d`](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d "torch.nn.ConvTranspose2d") | Applies a 2D transposed convolution operator over an input image composed of several input planes. |
| [`nn.ConvTranspose3d`](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html#torch.nn.ConvTranspose3d "torch.nn.ConvTranspose3d") | Applies a 3D transposed convolution operator over an input image composed of several input planes. |
| [`nn.Unfold`](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold "torch.nn.Unfold")                                     | Extracts sliding local blocks from a batched input tensor.                                         |
| [`nn.Fold`](https://pytorch.org/docs/stable/generated/torch.nn.Fold.html#torch.nn.Fold "torch.nn.Fold")                                             | Combines an array of sliding local blocks into a large containing tensor.                          |
