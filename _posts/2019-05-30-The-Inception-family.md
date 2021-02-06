---
layout:     post
title:      The Inception Family
subtitle:   Building a deep neural network from Inception
date:       2019-05-30
author:     Yaodong Cui
header-img: img/post-bg-balls.jpg
header-mask: 0.5
catalog: true
tags:
    - Deep learning
    - Image Processing
    - Inception
    - Batch Normalization
    - Convolutional Neural Network
---


# Background

In 2014, a team in Google proposed a convolutional neural network architecture, called GoogLeNet or  [Inception](https://arxiv.org/abs/1409.4842) , which increase the depth and width of the network while maintaining the computational expense. 


# Inception V1：Go Deeper

The network achieves this with several techniques including 1 X  1 convolution, Inception modules and global average pooling. 

The 1 X 1 convolution is used to reduce computation bottleneck by dimension reduction. The Inception modules, as shown in Fig.1, concatenates the results 1 X 1 Convolution, 3 X 3 Convolution,5 X 5 Convolution and X 3 max pooling, while using 1 X 1 Convolution for dimensionality reduction.
Instead of the fully connected layers, the global average pooling is used at the end of the network to reduce the number of weights.

<br>
<div  align="center"> 
<img src="https://raw.githubusercontent.com/yaodongC/yaodongC.github.io/master/post_img/190530/InceptionV101.png"
   width = "300" height = "300"> </div>
 <div align="center">Fig.1  Inception module with 1 X 1 Convolution</div>
<br>


The GoogLeNet also employs auxiliary classifier during the training process. Therefore, the total loss function is a weighted sum of the final loss and auxiliary loss. This methodology can prevent vanishing gradient problem and provide regularization. 



# The Inception V2 and V3: Do It With Less Computation

The Inception V2 and V3 models take a deeper dive in improving computational efficiency while achieving better accuracy.


The General design principles employed by Inception V2 are :

 - Avoid representational bottlenecks. The intuition is that the drastic reduction of representational size will cause information loss, namely the representational bottlenecks. 
 
- Factorizing large kernels into small ones. A 5 X 5 kernel has 25 weights, which is 2.78 times more expensive than a 3 X 3 kernel. Therefore, stacking two 3 X 3 kernel leads to better performance, as shown in Fig.2.

- Increase the dimensions of representations within the network, which allows for more disentangled features.

- Perform spatial aggregation/dimension reduction over lower dimensional embedding can avoid serious adverse effects. For instance, employ 1 X 1 convolution to reduce the input dimension before feeding it into a spread out convolution(e.g. 3X3 or 5X5).

- Balance the width and depth of the network.
 <br>
<div  align="center"> 
    <img 
    src="https://raw.githubusercontent.com/yaodongC/yaodongC.github.io/master/post_img/190530/InceptionV2.png"
    width = "300" height = "300"> </div>
 <div align="center">Fig.2   Inception V2 module with 5 X 5 convolution replaced by two 3 X 3 convolution.</div>
 <br>

##spatial factorization into asymmetric convolutions
The author takes factorization a step further into asymmetric convolutions, namely use n X 1 convolution. For instance, implement 3 X 1 convolution followed by a 1 X 3 convolution is equivalent to a 3 X 3 convolution, while saving 33% computational expense. This idea is very similar to the concept of depthwise separable convolution proposed by MobileNet in 2017.
<br>
<div  align="center"> 
    <img 
    src="https://raw.githubusercontent.com/yaodongC/yaodongC.github.io/master/post_img/190530/InceptionV2factorization.png"
    width = "300" height = "400"></div>
 <div align="center">Fig.3  Inception V2 module with asymmetric convolutionsn.</div>
<br>

##why go deeper when you can go wider
The Inception V2 also attempts to eliminate the information bottleneck with going wider, as shown in Fig.4. The notion is that if the model is made to be deeper, it would need to reduce data dimension, which could lead to information loss. The author proposes to expand the filter banks to avoid the representational bottleneck.
<br>
<div  align="center"> 
    <img 
    src="https://raw.githubusercontent.com/yaodongC/yaodongC.github.io/master/post_img/190530/InceptionV2highrepresentation.png"
    width = "300" height = "300"></div>
 <div align="center">Fig.4   Inception V2 module with expanded filter bank outputs.</div>
<br>  

# Conclusion

