# ResNet (Residual Network)
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result


---

## 简介
ResNet（残差网络）是由何凯明、张祥雨、任少卿等人于2015年提出的划时代深度卷积神经网络，成果发表于《Deep Residual Learning for Image Recognition》，一举夺得ILSVRC 2015图像分类竞赛冠军。它打破了深度网络层数的增长瓶颈，首次提出**残差块**与**短路连接（Shortcut Connection）**核心设计，通过残差学习策略彻底解决了传统深度网络的**性能退化**难题，同时有效缓解梯度消失与梯度爆炸。ResNet成功构建出152层超深网络，将ImageNet分类错误率大幅降低至3.57%，结构简洁、泛化能力极强，成为深度学习史上应用最广泛的基础骨架网络，为后续目标检测、语义分割、Transformer视觉模型、轻量化网络等各类任务提供了核心支撑。

## 架构
ResNet核心架构为**残差块堆叠式深度神经网络**，整体由**初始卷积池化层**、**多组串联残差模块**、**全局平均池化**与**全连接分类头**组成，摒弃了传统网络单纯堆叠卷积层的固有模式。原论文标准输入为224×224分辨率的3通道RGB图像，适配任意深度层级（ResNet18/34/50/101/152），具体核心结构与设计如下：
- **核心特征单元（残差块）**：分为基础残差块（浅层网络）与瓶颈残差块（深层网络）。主分支通过多层卷积完成特征映射与提取，支路采用恒等映射或1×1卷积下采样实现维度匹配，双分支结果逐元素相加融合输出残差特征。
- **残差学习机制（短路连接）**：放弃直接拟合底层到高层的完整映射，转而学习残差映射，让深层网络拟合恒等映射变得简单可行。梯度可通过短路支路直接反向传播，极大保留梯度信息流，从根源缓解深层网络梯度消失。
- **多尺度层级设计与分类输出**：网络按阶段逐步下采样、提升通道数，逐级提取浅层纹理、中层局部特征、高层全局语义特征；末端使用全局平均池化压缩特征，搭配单层全连接层完成分类，精简参数量并抑制过拟合。

该架构以极简的旁路设计突破网络深度限制，兼顾高性能、易训练与强泛化性，**残差学习**思想奠定了现代深层视觉网络的设计范式，是计算机视觉领域里程碑式的创新。




**注意**：本文实验基于CIFAR-10十分类数据集开展，不同于原论文的ImageNet大规模数据集。因CIFAR-10图像尺寸仅为32×32，远小于原版224×224输入，因此对原生ResNet结构进行轻量化适配：调整首层卷积核尺寸、步长，删减冗余下采样层，适配小尺寸图像特征提取，全程完整保留**残差块+短路连接**核心机制，保证模型原理一致性。

## 数据集
我们使用的是数据集CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10是由Hinton的学生Alex Krizhevsky和Ilya Sutskever整理的一个用于识别普适物体的小型数据集。一共包含10个类别的RGB彩色图片：飞机（airplane）、汽车（automobile）、鸟类（bird）、猫（cat）、鹿（deer）、狗（dog）、蛙类（frog）、马（horse）、船（ship）和卡车（truck）。每个图片的尺寸为32×32，每个类别有6000个图像，数据集中一共有50000张训练图片和10000张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

它不同于我们常见的图片存储格式，而是用二进制优化了储存，当然我们也可以将其复刻出来为PNG等图片格式，但那会很大，我们的目标是神经网络，这里不做细致解析数据集，如果你想了解该数据集请观看链接：https://cloud.tencent.com/developer/article/2150614

---

## Introduction
ResNet (Residual Network) is a landmark deep convolutional neural network proposed by Kaiming He, Xiangyu Zhang, Shaoqing Ren and other researchers in 2015. 2015. The research was published in *Deep Residual Learning for Image Recognition* and won the championship of the ILSVRC 2015 Image Classification Competition. It broke through the depth limitation of traditional convolutional networks and firstly proposed two core designs: **residual block** and **shortcut connection**. By adopting the residual learning strategy, it thoroughly solved the performance degradation problem of ultra-deep networks, and effectively relieved gradient vanishing and gradient explosion.

ResNet successfully constructed a 152-layer ultra-deep network, reducing the ImageNet classification error rate to 3.57%. With concise structure and strong generalization ability, it has become the most widely used backbone network in deep learning. It provides fundamental technical support for downstream tasks such as object detection, semantic segmentation, vision Transformer and lightweight model design.

## Architecture
The core architecture of ResNet is a **residual block stacked deep neural network**. The overall structure consists of an **initial convolution and pooling layer**, **multi-stage stacked residual modules**, **global average pooling** and a **fully connected classification head**, which abandons the simple layer-stacking mode of traditional CNNs.

The standard input of the original paper is 224×224 RGB images with 3 channels, covering multiple network specifications including ResNet18/34/50/101/152. The core design is as follows:
- **Core Feature Extraction (Residual Block)**: Divided into basic residual blocks for shallow networks and bottleneck residual blocks for deep networks. The main branch completes feature extraction through stacked convolutions, while the shortcut branch uses identity mapping or 1×1 convolution downsampling to unify channel and spatial dimensions, and the two branches are fused by element-wise addition.
- **Residual Learning Strategy (Shortcut Connection)**: Instead of fitting the direct mapping from shallow to deep layers, the network learns residual functions. It makes it easy for ultra-deep models to fit identity mapping, and gradients can be backpropagated directly through shortcut branches to ensure effective information transmission.
- **Hierarchical Feature Fusion & Classification Head**: The network gradually downsamples and increases channel dimensions stage by stage to extract low-level texture, mid-level local features and high-level semantic features. Global average pooling is adopted before classification to replace redundant fully connected layers, reducing parameters and overfitting risks.

With the simple bypass connection design, ResNet breaks the depth bottleneck of convolutional networks. Its innovative residual learning paradigm has become the essential design principle of modern deep visual models.



**Note:** This experiment adopts the 10-class CIFAR-10 dataset. Considering the 32×32 low-resolution input (much smaller than the 224×224 input in the original paper), we have made targeted lightweight adaptation for ResNet: adjusting the kernel size and stride of the first convolutional layer, removing redundant early pooling and downsampling layers, and adapting the dimension of fully connected layers. The core residual mechanism including residual blocks and shortcut connections is completely retained.

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Hhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

It differs from common image storage formats, using binary-optimized storage. While we could recreate it as PNG or other image formats, that would result in a very large file size. Our focus is on neural networks, so we won't delve into a detailed analysis of the dataset here. If you'd like to learn more about this dataset, please see the link: https://cloud.tencent.com/developer/article/2150614

---
## 原文章 | Original article
He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
