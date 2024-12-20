---
title: 'PyTorch 高频使用代码'
date: 2024-07-25
permalink: /posts/2024/07/Pytorchcoding/
tags:
  - PyThon
  - PyTorch
  - CUDA
---

PyTorch常用代码段合集，涵盖基本配置、张量处理、模型定义与操作、数据处理、模型训练与测试等5个方面，还给出了多个值得注意的Tips，内容非常全面。

PyTorch最好的资料是官方文档。本文是PyTorch常用代码段，在参考资料(张皓：PyTorch Cookbook)的基础上做了一些修补，方便使用时查阅。

## 📑【目录】
- [基本配置](#1-基本配置)
    - [导入包和版本查询](#11-导入包和版本查询)
    - [可复现性](#12-可复现性)
    - [显卡设置](#13-显卡设置)
    - [清除显存](#14-清除显存)
- [tensor张量的处理](#2-tensor张量的处理)
    - [张量的数据类型](#21-张量的数据类型)
    - [张量基本信息](#22-张量基本信息)
    - [命名张量](#23-命名张量)
    - [数据类型转换](#24-数据类型转换)
    - [torchtensor与npndarray转换](#25-torchtensor与npndarray转换)
    - [torchtensor与pilimage转换](#26-torchtensor与pilimage转换)
    - [npndarray与pilimage的转换](#27-npndarray与pilimage的转换)
    - [从只包含一个元素的张量中提取值](#28-从只包含一个元素的张量中提取值)
    - [张量形变](#29-张量形变)
    - [打乱顺序](#210-打乱顺序)
    - [水平翻转](#211-水平翻转)
    - [复制张量](#212-复制张量)
    - [张量拼接](#213-张量拼接)
    - [将整数标签转为one-hot编码](#214-将整数标签转为one-hot编码)
    - [得到非零元素](#215-得到非零元素)
    - [判断两个张量相等](#216-判断两个张量相等)
    - [张量扩展](#217-张量扩展)
    - [矩阵乘法](#218-矩阵乘法)
    - [计算两组数据之间的两两欧式距离](#219-计算两组数据之间的两两欧式距离)
    - [张量求和](#220-张量求和)

- [模型定义和操作](#3-模型定义和操作)
    - [一个简单两层卷积网络的示例](#31-一个简单两层卷积网络的示例)
    - [双线性池化操作bilinear-pooling](#32-双线性池化操作bilinear-pooling)
    - [多卡同步-bnbatch-normalization](#33-多卡同步-bnbatch-normalization)
    - [将已有网络的所有bn层改为同步bn层](#34-将已有网络的所有bn层改为同步bn层)
    - [类似-bn-滑动平均](#35-类似-bn-滑动平均)
    - [计算模型整体参数量](#36-计算模型整体参数量)
    - [查看网络中的参数](#37-查看网络中的参数)
    - [模型可视化使用pytorchviz](#38-模型可视化使用pytorchviz)
    - [类似-keras-的-modelsummary-输出模型信息使用pytorch-summary](#39-类似-keras-的-modelsummary-输出模型信息使用pytorch-summary)
- [模型权重初始化](#4-模型权重初始化)
    - [提取模型中的某一层](#41-提取模型中的某一层)
    - [部分层使用预训练模型](#42-部分层使用预训练模型)
    - [将在-gpu-保存的模型加载到-cpu](#43-将在-gpu-保存的模型加载到-cpu)
    - [导入另一个模型的相同部分到新的模型](#44-导入另一个模型的相同部分到新的模型)
- [数据处理](#5-数据处理)
    - [计算数据集的均值和标准差](#51-计算数据集的均值和标准差)
    - [得到视频数据基本信息](#52-得到视频数据基本信息)
    - [tsn-每段segment采样一帧视频](#53-tsn-每段segment采样一帧视频)
    - [常用训练和验证数据预处理](#54-常用训练和验证数据预处理)
- [模型训练和测试](#6-模型训练和测试)
    - [分类模型训练代码](#61-分类模型训练代码)
    - [分类模型测试代码](#62-分类模型测试代码)
    - [自定义loss](#63-自定义loss)
    - [标签平滑label-smoothing](#64-标签平滑label-smoothing)
    - [mixup训练](#65-mixup训练)
    - [l1-正则化](#66-l1-正则化)
    - [不对偏置项进行权重衰减weight-decay](#67-不对偏置项进行权重衰减weight-decay)
    - [梯度裁剪gradient-clipping](#68-梯度裁剪gradient-clipping)
    - [得到当前学习率](#69-得到当前学习率)
    - [学习率衰减](#610-学习率衰减)
    - [优化器链式更新](#611-优化器链式更新)
    - [模型训练可视化](#612-模型训练可视化)
    - [保存与加载断点](#613-保存与加载断点)
    - [提取-imagenet-预训练模型某层的卷积特征](#614-提取-imagenet-预训练模型某层的卷积特征)
    - [提取-imagenet-预训练模型多层的卷积特征](#615-提取-imagenet-预训练模型多层的卷积特征)
    - [微调全连接层](#616-微调全连接层)
    - [以较大学习率微调全连接层较小学习率微调卷积层](#617-以较大学习率微调全连接层较小学习率微调卷积层)
- [其他注意](#7-其他注意)
---

### 1. 基本配置

#### 1.1 导入包和版本查询

~~~python
import torch
import torch.nn as nn
import torchvision
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))
~~~

#### 1.2 可复现性
在硬件设备（CPU、GPU）不同时，完全的可复现性无法保证，即使随机种子相同。但是，在同一个设备上，应该保证可复现性。具体做法是，在程序开始的时候固定torch的随机种子，同时也把numpy的随机种子固定。
~~~python
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
~~~

#### 1.3 显卡设置
如果只需要一张显卡
~~~python
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
~~~

如果需要指定多张显卡，比如0，1号显卡。
~~~python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
~~~

也可以在命令行运行代码时设置显卡：

~~~python
CUDA_VISIBLE_DEVICES=0,1 python train.py
~~~

#### 1.4 清除显存
~~~python
torch.cuda.empty_cache()
~~~
也可以使用在命令行重置GPU的指令
~~~python
nvidia-smi --gpu-reset -i [gpu_id]
~~~

### 2. Tensor张量的处理

#### 2.1 张量的数据类型

PyTorch有9种CPU张量类型和9种GPU张量类型，如下图所示：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b73c6926d29b24007cd928da7f8c0bf1.png#pic_center)


#### 2.2 张量基本信息

~~~python
tensor = torch.randn(3,4,5)
print(tensor.type())  
# 数据类型
print(tensor.size())  
# 张量的shape，是个元组
print(tensor.dim())   
# 维度的数量
~~~

#### 2.3 命名张量

张量命名是一个非常有用的方法，这样可以方便地使用维度的名字来做索引或其他操作，大大提高了可读性、易用性，防止出错。

~~~python
# 在PyTorch 1.3之前，需要使用注释
# Tensor[N, C, H, W]
images = torch.randn(32, 3, 56, 56)
images.sum(dim=1)
images.select(dim=1, index=0)

# PyTorch 1.3之后
NCHW = [‘N’, ‘C’, ‘H’, ‘W’]
images = torch.randn(32, 3, 56, 56, names=NCHW)
images.sum('C')
images.select('C', index=0)
# 也可以这么设置
tensor = torch.rand(3,4,1,2,names=('C', 'N', 'H', 'W'))
# 使用align_to可以对维度方便地排序
tensor = tensor.align_to('N', 'C', 'H', 'W')
~~~

#### 2.4 数据类型转换
~~~python
# 设置默认类型，pytorch中的FloatTensor远远快于DoubleTensor
torch.set_default_tensor_type(torch.FloatTensor)

# 类型转换
tensor = tensor.cuda()
tensor = tensor.cpu()
tensor = tensor.float()
tensor = tensor.long()
~~~

#### 2.5 Torch.Tensor与Np.Ndarray转换

除了CharTensor，其他所有CPU上的张量都支持转换为numpy格式然后再转换回来。

~~~python
ndarray = tensor.cpu().numpy()
tensor = torch.from_numpy(ndarray).float()
tensor = torch.from_numpy(ndarray.copy()).float() 
# If ndarray has negative stride.
~~~

#### 2.6 Torch.Tensor与PIL.Image转换

~~~python
# pytorch中的张量默认采用[N, C, H, W]的顺序，并且数据范围在[0,1]，需要进行转置和规范化
# torch.Tensor -> PIL.Image
image = PIL.Image.fromarray(torch.clamp(tensor*255, min=0, max=255).byte().permute(1,2,0).cpu().numpy())
# 同样的转换形式下面可以替换，上下等价
image = torchvision.transforms.functional.to_pil_image(tensor)  

# PIL.Image -> torch.Tensor
path = r'./figure.jpg'
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2,0,1).float() / 255
# Equivalently way
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path))
~~~

#### 2.7 Np.Ndarray与PIL.Image的转换
~~~python
image = PIL.Image.fromarray(ndarray.astype(np.uint8))
ndarray = np.asarray(PIL.Image.open(path))
~~~

#### 2.8 从只包含一个元素的张量中提取值
~~~python
value = torch.rand(1).item()
~~~

#### 2.9 张量形变
~~~python
# 在将卷积层输入全连接层的情况下通常需要对张量做形变处理，
# 相比torch.view，torch.reshape可以自动处理输入张量不连续的情况

tensor = torch.rand(2,3,4)
shape = (6, 4)
tensor = torch.reshape(tensor, shape)
~~~

#### 2.10 打乱顺序

~~~python
# 打乱第一个维度
tensor = tensor[torch.randperm(tensor.size(0))]
~~~

#### 2.11 水平翻转

~~~python
# pytorch不支持tensor[::-1]这样的负步长操作，水平翻转可以通过张量索引实现
# 假设张量的维度为[N, D, H, W].

tensor = tensor[:,:,:,torch.arange(tensor.size(3) - 1, -1, -1).long()]
~~~

#### 2.12 复制张量

~~~python
# Operation                 |  New/Shared memory | Still in computation graph |
tensor.clone()            # |        New         |          Yes               |
tensor.detach()           # |      Shared        |          No                |
tensor.detach.clone()()   # |        New         |          No                |
~~~

#### 2.13 张量拼接

~~~python
'''
注意torch.cat和torch.stack的区别在于torch.cat沿着给定的维度拼接，
而torch.stack会新增一维。例如当参数是3个10x5的张量，torch.cat的结果是30x5的张量，
而torch.stack的结果是3x10x5的张量。
'''
tensor = torch.cat(list_of_tensors, dim=0)
tensor = torch.stack(list_of_tensors, dim=0)
~~~

#### 2.14 将整数标签转为One-Hot编码
首先是经常使用的torch的形式：
pytorch自带的将标签转换成独热编码的方法：
~~~python
torch.nn.funtional.one_hot(tensor,num_classes=-1)->LongTensor
~~~
也就是num_class控制的是独热编码的维度，默认按照前面tensor大小设置，前面如果最大是6，则num_class设置成0-6也就是7；前面最大2，则num_class为3。Num_class也可以设置成大于等于前面数字中的任意值，那么就生成对应的维度。
~~~python
# pytorch的标记默认从0开始
tensor = torch.tensor([0, 2, 1, 3])
N = tensor.size(0)
num_classes = 4
one_hot = torch.zeros(N, num_classes).long()
one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())
~~~

#### 2.15 得到非零元素

~~~python
# index of non-zero elements
torch.nonzero(tensor) 
# index of zero elements              
torch.nonzero(tensor==0)  
# number of non-zero elements          
torch.nonzero(tensor).size(0)
# number of zero elements       
torch.nonzero(tensor == 0).size(0) 
~~~

#### 2.16 判断两个张量相等

~~~python
# float tensor
torch.allclose(tensor1, tensor2)  
# int tensor
torch.equal(tensor1, tensor2)     
~~~

#### 2.17 张量扩展

~~~python
# Expand tensor of shape 64*512 to shape 64*512*7*7.
tensor = torch.rand(64,512)
torch.reshape(tensor, (64, 512, 1, 1)).expand(64, 512, 7, 7)
~~~

#### 2.18 矩阵乘法
单纯的乘积就是torch.mm
忽略前面的batch维度的乘法就是torch.bmm
逐元素相乘的点积运算则是*直接进行

~~~python
# Matrix multiplcation: (m*n) * (n*p) * -> (m*p).
result = torch.mm(tensor1, tensor2)

# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p)
result = torch.bmm(tensor1, tensor2)

# Element-wise multiplication.
result = tensor1 * tensor2
~~~

#### 2.19 计算两组数据之间的两两欧式距离

利用广播机制
~~~python
dist = torch.sqrt(torch.sum((X1[:,None,:] - X2) ** 2, dim=2))
~~~

#### 2.20 张量求和
使用torch.einsum()
下面给出一个实例：
~~~python
# trace(迹)
>>> torch.einsum('ii', torch.randn(4, 4))
tensor(-1.4157)

# diagonal（对角线）
>>> torch.einsum('ii->i', torch.randn(4, 4))
tensor([ 0.0266,  2.4750, -1.0881, -1.3075])

# outer product（外积）
>>> x = torch.randn(5)
tensor([-0.3550, -0.6059, -1.3375, -1.5649,  0.2675])
>>> y = torch.randn(4)
tensor([-0.2202, -1.5290, -2.0062,  0.9600])
>>> torch.einsum('i,j->ij', x, y)
tensor([[ 0.0782,  0.5428,  0.7122, -0.3408],
        [ 0.1334,  0.9264,  1.2156, -0.5817],
        [ 0.2945,  2.0451,  2.6834, -1.2840],
        [ 0.3445,  2.3927,  3.1396, -1.5023],
        [-0.0589, -0.4089, -0.5366,  0.2568]])

# batch matrix multiplication(批量矩阵乘法)
>>> As = torch.randn(3,2,5)
tensor([[[-0.0306,  0.8251,  0.0157, -0.4563,  0.5550],
         [-1.4550,  0.0762,  0.9258,  0.1198, -1.1737]],

        [[-0.4460, -0.7224,  0.7260,  0.7552,  0.0326],
         [-0.3904, -1.2392,  0.4848, -0.4756,  0.2301]],

        [[ 1.5307,  0.7668, -1.9426,  1.7473, -0.6258],
         [ 0.6758,  1.8240, -0.2053,  0.0973, -0.6118]]])

>>> Bs = torch.randn(3,5,4)
tensor([[[-0.7054, -0.2155, -1.5458, -0.8236],
         [-1.4957, -2.2604,  0.6897, -1.0360],
         [ 1.2924,  0.2798,  1.0544,  0.3656],
         [-0.3993, -1.2463, -0.6601,  0.2706],
         [ 1.0727,  0.5418, -0.2516, -0.1133]],

        [[ 0.4215,  1.5712, -0.2351,  1.3741],
         [ 1.6418,  0.9806, -1.0259, -1.1297],
         [ 0.7326,  0.4989,  0.4404,  0.2975],
         [-0.6866,  0.5696, -0.8942,  0.6815],
         [ 1.7486,  0.5344,  0.0538,  0.5258]],

        [[ 1.6280, -1.3989, -0.2900,  0.0936],
         [-0.9436, -0.1766,  0.6780,  0.3152],
         [ 0.9645, -0.1199, -1.1644, -1.0290],
         [-0.2791, -0.8086,  0.2161,  0.7901],
         [ 1.3222, -1.4023, -2.4181, -1.2875]]])

>>> torch.einsum('bij,bjk->bik', As, Bs)
tensor([[[-0.4147, -0.9847,  0.7946, -1.0103],
         [ 0.8020, -0.3849,  3.4942,  1.6233]],
        
        [[-1.3035, -0.5993,  0.4922,  0.9511],
         [-1.1150, -1.7346,  2.0142,  0.8047]],
        
        [[-1.4202, -2.5790,  4.2288,  4.5702],
         [-1.6549, -0.4636,  2.7802,  1.7141]]])


# with sublist format and ellipsis（带有子列表格式和省略号）
>>> torch.einsum(As, [..., 0, 1], Bs, [..., 1, 2], [..., 0, 2])
tensor([[[-0.4147, -0.9847,  0.7946, -1.0103],
         [ 0.8020, -0.3849,  3.4942,  1.6233]],
        
        [[-1.3035, -0.5993,  0.4922,  0.9511],
         [-1.1150, -1.7346,  2.0142,  0.8047]],
        
        [[-1.4202, -2.5790,  4.2288,  4.5702],
         [-1.6549, -0.4636,  2.7802,  1.7141]]])


# batch permute（批量交换）
>>> A = torch.randn(2, 3, 4, 5)
>>> torch.einsum('...ij->...ji', A).shape
torch.Size([2, 3, 5, 4])


# equivalent to torch.nn.functional.bilinear（等价于torch.nn.functional.bilinear）
>>> A = torch.randn(3,5,4)
>>> l = torch.randn(2,5)
>>> r = torch.randn(2,4)
>>> torch.einsum('bn,anm,bm->ba', l, A, r)
tensor([[-0.3430, -5.2405,  0.4494],
        [ 0.3311,  5.5201, -3.0356]])



### 3. 模型定义和操作

#### 3.1 一个简单两层卷积网络的示例

~~~python
# convolutional neural network (2 convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)
~~~

<!-- #### 卷积层的计算和展示可以用这个网站辅助。 -->

#### 3.2 双线性池化操作（Bilinear Pooling）
bilinear pooling主要用于特征融合, 对于从同一个样本提取出来的特征 $x$ 和特征 $y$, 通过bilinear pooling得到两个特征融合后的向量, 进 而用来分类。
如果特征 $x$ 和特征 $y$ 来自两个特征提取器, 则被称为多模双线性池化 (MBP, Multimodal Bilinear Pooling)
如果特征 $x=$ 特征 $y$, 则被称为同源双线性池化 $\mathrm{Q}$ （HBP, Homogeneous Bilinear Pooling）或者二阶池化（Second-order Pooling）
~~~python
# Assume X has shape N*D*H*W
X = torch.reshape(N, D, H * W)  
# Bilinear pooling                      
X = torch.bmm(X, torch.transpose(X, 1, 2)) / (H * W)  
assert X.size() == (N, D, D)
X = torch.reshape(X, (N, D * D))
# Signed-sqrt normalization
X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
# L2 normalization   
X = torch.nn.functional.normalize(X) 
~~~

#### 3.3 多卡同步 BN（Batch Normalization）

当使用 torch.nn.DataParallel 将代码运行在多张 GPU 卡上时，PyTorch 的 BN 层默认操作是各卡上数据独立地计算均值和标准差，同步 BN 使用所有卡上的数据一起计算 BN 层的均值和标准差，缓解了当批量大小（batch size）比较小时对均值和标准差估计不准的情况，是在目标检测等任务中一个有效的提升性能的技巧。
~~~python
sync_bn = torch.nn.SyncBatchNorm(num_features, 
                                 eps=1e-05, 
                                 momentum=0.1, 
                                 affine=True, 
                                 track_running_stats=True)
~~~

#### 3.4 将已有网络的所有BN层改为同步BN层

~~~python
def convertBNtoSyncBN(module, process_group=None):
    '''Recursively replace all BN layers to SyncBN layer.

    Args:
        module[torch.nn.Module]. Network
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        sync_bn = torch.nn.SyncBatchNorm(module.num_features, module.eps, module.momentum, 
                                         module.affine, module.track_running_stats, process_group)
        sync_bn.running_mean = module.running_mean
        sync_bn.running_var = module.running_var
        if module.affine:
            sync_bn.weight = module.weight.clone().detach()
            sync_bn.bias = module.bias.clone().detach()
        return sync_bn
    else:
        for name, child_module in module.named_children():
            setattr(module, name) = convert_syncbn_model(child_module, process_group=process_group))
        return module
~~~

#### 3.5 类似 BN 滑动平均

如果要实现类似 BN 滑动平均的操作，在 forward 函数中要使用原地（inplace）操作给滑动平均赋值。
~~~python
class BN(torch.nn.Module)
    def __init__(self):
        ...
        self.register_buffer('running_mean', torch.zeros(num_features))

    def forward(self, X):
        ...
        self.running_mean += momentum * (current - self.running_mean)
~~~

#### 3.6 计算模型整体参数量

~~~python
num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
~~~

#### 3.7 查看网络中的参数

可以通过model.state_dict()或者model.named_parameters()函数查看现在的全部可训练参数（包括通过继承得到的父类中的参数）
~~~python
params = list(model.named_parameters())
(name, param) = params[28]
print(name)
print(param.grad)
print('-------------------------------------------------')
(name2, param2) = params[29]
print(name2)
print(param2.grad)
print('----------------------------------------------------')
(name1, param1) = params[30]
print(name1)
print(param1.grad)
~~~

#### 3.8 模型可视化（使用Pytorchviz）
~~~
szagoruyko/pytorchvizgithub.com
~~~

#### 3.9 类似 Keras 的 model.summary() 输出模型信息，使用pytorch-summary
~~~
sksq96/pytorch-summarygithub.com
~~~

### 4. 模型权重初始化

注意 model.modules() 和 model.children() 的区别：model.modules() 会迭代地遍历模型的所有子层，而 model.children() 只会遍历模型下的一层。
~~~python
# Common practise for initialization.
for layer in model.modules():
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                      nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

# Initialization with given tensor.
layer.weight = torch.nn.Parameter(tensor)
~~~

#### 4.1 提取模型中的某一层

modules()会返回模型中所有模块的迭代器，它能够访问到最内层，比如self.layer1.conv1这个模块，还有一个与它们相对应的是name_children()属性以及named_modules(),这两个不仅会返回模块的迭代器，还会返回网络层的名字。
~~~python
# 取模型中的前两层
new_model = nn.Sequential(*list(model.children())[:2] 
# 如果希望提取出模型中的所有卷积层，可以像下面这样操作：
for layer in model.named_modules():
    if isinstance(layer[1],nn.Conv2d):
         conv_model.add_module(layer[0],layer[1])
~~~

#### 4.2 部分层使用预训练模型

注意如果保存的模型是 torch.nn.DataParallel，则当前的模型也需要是

~~~python
model.load_state_dict(torch.load('model.pth'), strict=False)
~~~

#### 4.3 将在 GPU 保存的模型加载到 CPU
~~~python
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
~~~

#### 4.4 导入另一个模型的相同部分到新的模型

模型导入参数时，如果两个模型结构不一致，则直接导入参数会报错。用下面方法可以把另一个模型的相同的部分导入到新的模型中。

~~~python
# model_new代表新的模型
# model_saved代表其他模型，比如用torch.load导入的已保存的模型
model_new_dict = model_new.state_dict()
model_common_dict = {k:v for k, v in model_saved.items() if k in model_new_dict.keys()}
model_new_dict.update(model_common_dict)
model_new.load_state_dict(model_new_dict)
~~~

### 5. 数据处理

#### 5.1 计算数据集的均值和标准差
~~~python
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

def compute_mean_and_std(dataset):
    # 输入PyTorch的dataset，输出均值和标准差
    mean_r = 0
    mean_g = 0
    mean_b = 0

    for img, _ in dataset:
        img = np.asarray(img) # change PIL Image to numpy array
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0

    for img, _ in dataset:
        img = np.asarray(img)

        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
    return mean, std
~~~

#### 5.2 得到视频数据基本信息
~~~python
import cv2
video = cv2.VideoCapture(mp4_path)
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
video.release()
~~~

#### 5.3 TSN 每段（Segment）采样一帧视频
~~~python
K = self._num_segments
if is_train:
    if num_frames > K:
        # Random index for each segment.
        frame_indices = torch.randint(
            high=num_frames // K, size=(K,), dtype=torch.long)
        frame_indices += num_frames // K * torch.arange(K)
    else:
        frame_indices = torch.randint(
            high=num_frames, size=(K - num_frames,), dtype=torch.long)
        frame_indices = torch.sort(torch.cat((
            torch.arange(num_frames), frame_indices)))[0]
else:
    if num_frames > K:
        # Middle index for each segment.
        frame_indices = num_frames / K // 2
        frame_indices += num_frames // K * torch.arange(K)
    else:
        frame_indices = torch.sort(torch.cat((                              
            torch.arange(num_frames), torch.arange(K - num_frames))))[0]
assert frame_indices.size() == (K,)
return [frame_indices[i] for i in range(K)]
~~~

#### 5.4 常用训练和验证数据预处理

其中 ToTensor 操作会将 PIL.Image 或形状为 H×W×D，数值范围为 [0, 255] 的 np.ndarray 转换为形状为 D×H×W，数值范围为 [0.0, 1.0] 的 torch.Tensor。
~~~python
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(size=224,
                                             scale=(0.08, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
 ])
 val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
])
~~~

### 6。 模型训练和测试

#### 6.1 分类模型训练代码

~~~python
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i ,(images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
~~~

#### 6.2 分类模型测试代码

~~~python
# Test the model
# eval mode(batch norm uses moving mean/variance 
#instead of mini-batch mean/variance)
model.eval()  
              
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test accuracy of the model on the 10000 test images: {} %'
          .format(100 * correct / total))
~~~

#### 6.3 自定义Loss
继承torch.nn.Module类写自己的loss。
~~~python
class MyLoss(torch.nn.Moudle):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean((x - y) ** 2)
        return loss
~~~

#### 6.4 标签平滑（Label Smoothing）
写一个label_smoothing.py的文件，然后在训练代码里引用，用LSR代替交叉熵损失即可。label_smoothing.py内容如下：
~~~python
import torch
import torch.nn as nn


class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        #labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / (length - 1)

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                    .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                    .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                    .format(x.size()))


        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)

        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')
~~~

或者直接在训练文件里做label smoothing
~~~python
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    N = labels.size(0)
    # C is the number of classes.
    smoothed_labels = torch.full(size=(N, C), fill_value=0.1 / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=0.9)

    score = model(images)
    log_prob = torch.nn.functional.log_softmax(score, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
~~~

#### 6.5 Mixup训练
~~~python
beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    # Mixup images and labels.
    lambda_ = beta_distribution.sample([]).item()
    index = torch.randperm(images.size(0)).cuda()
    mixed_images = lambda_ * images + (1 - lambda_) * images[index, :]
    label_a, label_b = labels, labels[index]

    # Mixup loss.
    scores = model(mixed_images)
    loss = (lambda_ * loss_function(scores, label_a)
            + (1 - lambda_) * loss_function(scores, label_b))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
~~~

#### 6.6 L1 正则化
~~~python
l1_regularization = torch.nn.L1Loss(reduction='sum')
loss = ...  # Standard cross-entropy loss

for param in model.parameters():
    loss += torch.sum(torch.abs(param))
loss.backward()
~~~

#### 6.7 不对偏置项进行权重衰减（Weight Decay)
pytorch里的weight decay相当于l2正则
~~~python
bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'parameters': bias_list, 'weight_decay': 0},                
              {'parameters': others_list}]
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
~~~

#### 6.8 梯度裁剪（Gradient Clipping）
~~~python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
~~~

#### 6.9 得到当前学习率
~~~python
# If there is one global learning rate (which is the common case).
lr = next(iter(optimizer.param_groups))['lr']

# If there are multiple learning rates for different layers.
all_lr = []
for param_group in optimizer.param_groups:
    all_lr.append(param_group['lr'])
~~~

另一种方法，在一个batch训练代码里，当前的lr是optimizer.param_groups[0]['lr']

#### 6.10 学习率衰减
~~~python
# Reduce learning rate when validation accuarcy plateau.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
for t in range(0, 80):
    train(...)
    val(...)
    scheduler.step(val_acc)

# Cosine annealing learning rate.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
# Reduce learning rate by 10 at given epochs.
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
for t in range(0, 80):
    scheduler.step()    
    train(...)
    val(...)

# Learning rate warmup by 10 epochs.
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: t / 10)
for t in range(0, 10):
    scheduler.step()
    train(...)
    val(...)
~~~

#### 6.11 优化器链式更新

从1.4版本开始，torch.optim.lr_scheduler 支持链式更新（chaining），即用户可以定义两个 schedulers，并交替在训练中使用。
~~~python
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, StepLR
model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = StepLR(optimizer, step_size=3, gamma=0.1)
for epoch in range(4):
    print(epoch, scheduler2.get_last_lr()[0])
    optimizer.step()
    scheduler1.step()
    scheduler2.step()
~~~

#### 6.12 模型训练可视化
PyTorch可以使用tensorboard来可视化训练过程。

安装和运行TensorBoard。
~~~python
pip install tensorboard
tensorboard --logdir=runs
~~~

使用SummaryWriter类来收集和可视化相应的数据，放了方便查看，可以使用不同的文件夹，比如'Loss/train'和'Loss/test'。
~~~python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
~~~

#### 6.13 保存与加载断点
注意为了能够恢复训练，我们需要同时保存模型和优化器的状态，以及当前的训练轮数。
~~~python
start_epoch = 0
# Load checkpoint.
if resume: # resume为参数，第一次训练时设为0，中断再训练时设为1
    model_path = os.path.join('model', 'best_checkpoint.pth.tar')
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Load checkpoint at epoch {}.'.format(start_epoch))
    print('Best accuracy so far {}.'.format(best_acc))

# Train the model
for epoch in range(start_epoch, num_epochs): 
    ... 

    # Test the model
    ...

    # save checkpoint
    is_best = current_acc > best_acc
    best_acc = max(current_acc, best_acc)
    checkpoint = {
        'best_acc': best_acc,
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    model_path = os.path.join('model', 'checkpoint.pth.tar')
    best_model_path = os.path.join('model', 'best_checkpoint.pth.tar')
    torch.save(checkpoint, model_path)
    if is_best:
        shutil.copy(model_path, best_model_path)
~~~

#### 6.14 提取 ImageNet 预训练模型某层的卷积特征
~~~python
# VGG-16 relu5-3 feature.
model = torchvision.models.vgg16(pretrained=True).features[:-1]
# VGG-16 pool5 feature.
model = torchvision.models.vgg16(pretrained=True).features
# VGG-16 fc7 feature.
model = torchvision.models.vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
# ResNet GAP feature.
model = torchvision.models.resnet18(pretrained=True)
model = torch.nn.Sequential(collections.OrderedDict(
    list(model.named_children())[:-1]))

with torch.no_grad():
    model.eval()
    conv_representation = model(image)
~~~

#### 6.15 提取 ImageNet 预训练模型多层的卷积特征
~~~python
class FeatureExtractor(torch.nn.Module):
    """Helper class to extract several convolution features from the given
    pre-trained model.

    Attributes:
        _model, torch.nn.Module.
        _layers_to_extract, list<str> or set<str>

    Example:
        >>> model = torchvision.models.resnet152(pretrained=True)
        >>> model = torch.nn.Sequential(collections.OrderedDict(
                list(model.named_children())[:-1]))
        >>> conv_representation = FeatureExtractor(
                pretrained_model=model,
                layers_to_extract={'layer1', 'layer2', 'layer3', 'layer4'})(image)
    """
    def __init__(self, pretrained_model, layers_to_extract):
        torch.nn.Module.__init__(self)
        self._model = pretrained_model
        self._model.eval()
        self._layers_to_extract = set(layers_to_extract)

    def forward(self, x):
        with torch.no_grad():
            conv_representation = []
            for name, layer in self._model.named_children():
                x = layer(x)
                if name in self._layers_to_extract:
                    conv_representation.append(x)
            return conv_representation
~~~

#### 6.16 微调全连接层
~~~python
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 100)  # Replace the last fc layer
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
~~~

#### 6.17 以较大学习率微调全连接层，较小学习率微调卷积层
~~~python
model = torchvision.models.resnet18(pretrained=True)
finetuned_parameters = list(map(id, model.fc.parameters()))
conv_parameters = (p for p in model.parameters() if id(p) not in finetuned_parameters)
parameters = [{'params': conv_parameters, 'lr': 1e-3}, 
              {'params': model.fc.parameters()}]
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
~~~

### 7. 其他注意
- 不要使用太大的线性层。因为nn.Linear(m,n)使用的是的内存，线性层太大很容易超出现有显存。

- 不要在太长的序列上使用RNN。因为RNN反向传播使用的是BPTT算法，其需要的内存和输入序列的长度呈线性关系。
- model(x) 前用 model.train() 和 model.eval() 切换网络状态。

- 不需要计算梯度的代码块用 with torch.no_grad() 包含起来。
model.eval() 和 torch.no_grad() 的区别在于，model.eval() 是将网络切换为测试状态，例如 BN 和dropout在训练和测试阶段使用不同的计算方法。torch.no_grad() 是关闭 PyTorch 张量的自动求导机制，以减少存储使用和加速计算，得到的结果无法进行 loss.backward()。

- model.zero_grad()会把整个模型的参数的梯度都归零, 而optimizer.zero_grad()只会把传入其中的参数的梯度归零.
- torch.nn.CrossEntropyLoss 的输入不需要经过 Softmax。
- torch.nn.CrossEntropyLoss 等价于 torch.nn.functional.log_softmax + torch.nn.NLLLoss。

- loss.backward() 前用 optimizer.zero_grad() 清除累积梯度。

- torch.utils.data.DataLoader 中尽量设置 pin_memory=True，对特别小的数据集如 MNIST 设置 pin_memory=False 反而更快一些。num_workers 的设置需要在实验中找到最快的取值。

- 用 del 及时删除不用的中间变量，节约 GPU 存储。使用 inplace 操作可节约 GPU 存储，如：
x = torch.nn.functional.relu(x, inplace=True)
- 减少 CPU 和 GPU 之间的数据传输。例如如果你想知道一个 epoch 中每个 mini-batch 的 loss 和准确率，先将它们累积在 GPU 中等一个 epoch 结束之后一起传输回 CPU 会比每个 mini-batch 都进行一次 GPU 到 CPU 的传输更快。

- 使用半精度浮点数 half() 会有一定的速度提升，具体效率依赖于 GPU 型号。需要小心数值精度过低带来的稳定性问题。

- 时常使用 assert tensor.size() == (N, D, H, W) 作为调试手段，确保张量维度和你设想中一致。

- 除了标记 y 外，尽量少使用一维张量，使用 n*1 的二维张量代替，可以避免一些意想不到的一维张量计算结果。

- 统计代码各部分耗时：
~~~python
# 或者在命令行运行python -m torch.utils.bottleneck main.py
with torch.autograd.profiler.profile(enabled=True, use_cuda=False) as profile:    
  ...print(profile)
~~~

- 使用TorchSnooper来调试PyTorch代码，程序在执行的时候，就会自动 print 出来每一行的执行结果的 tensor 的形状、数据类型、设备、是否需要梯度的信息。
~~~python
# pip install torchsnooper
import torchsnooper# 对于函数，使用修饰器@torchsnooper.snoop()

# 如果不是函数，使用 with 语句来激活 TorchSnooper，把训练的那个循环装进 with 语句中去。
with torchsnooper.snoop():    
  原本的代码
https://github.com/zasdfgbnm/TorchSnoopergithub.com
~~~

模型可解释性，使用captum库：https://captum.ai/captum.ai

# CONTINUE...