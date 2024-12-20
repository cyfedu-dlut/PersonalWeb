---
title: '【一日速成编程系列】几行代码学会Pytorch'
date: 2024-09-05
permalink: /posts/2024/09/pytorchprogramming/
tags:
  - PyTorch
  - Python
  - Programming
---

顾名思义，几行代码的意思就是凝练浓缩重要高频率代码，快速掌握PyTorch

# Pytorch
## 导入库以及计算形状（查看形状太高频了，操作tensor必须的操作）
~~~python
# Import the pytorch library
import torch

# Create a 2D tensor with random values
x1 = torch.rand(3,4)    # The arguments indicate the shape
print(x1)
x2 = torch.rand(2,3,4)    # The arguments indicate the shape
print(x2)
# You can check the shape of the tensor
print("x2's shape is {}".format(x2.shape))
~~~
输出结果：
~~~
x1: 
tensor([[0.1312, 0.9761, 0.5368, 0.7348],
        [0.0394, 0.8787, 0.1282, 0.2441],
        [0.4475, 0.3453, 0.4261, 0.5283]])

x2: 
tensor([[[0.9598, 0.9903, 0.4824, 0.1071],
         [0.2781, 0.2530, 0.3437, 0.0645],
         [0.6265, 0.6392, 0.1205, 0.4414]],

        [[0.0745, 0.1636, 0.5996, 0.5957],
         [0.7711, 0.0973, 0.0545, 0.9022],
         [0.8294, 0.1489, 0.4127, 0.4979]]])
         
torch.Size([2, 3, 4])
~~~

## 创建tensors，重中之重
~~~python
# Can be any number of dimensions
x1 = torch.rand(4,2,6,3,8)

# Tensor with standard Normal distributed values
x2 = torch.randn(2,3)
print(x2)

# Tensor with zeros
x3 = torch.zeros(2,3)
print(x3)

# Tensor with ones
x4 = torch.ones(2,3)
print(x4)
~~~

输出结果：
~~~
tensor([[ 0.0778, -0.0176, -0.0550],
        [-0.3680, -0.7664, -0.8207]])
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
~~~
我们接着看一个事例：
~~~python
# Tensor with increasing values
x5 = torch.arange(6)
print(x5)

# Create a tensor with your favourite values
my_tensor = torch.Tensor([1.41, 5.1, 3.1415])
print(my_tensor)
~~~
输出就变成了：
~~~
tensor([0, 1, 2, 3, 4, 5])
tensor([1.4100, 5.1000, 3.1415])
~~~

## Tensors类型（重中之重！！！）
**不同的tensor类型将会对算法的性能以及运算产生不同程度的影响，这个必须清楚**：
在PyTorch中，Tensor（张量）是基本的数据结构，用于存储和操作数据。Tensor类似于NumPy的ndarray，但可以在GPU上进行加速计算。PyTorch支持多种Tensor类型，主要区别在于它们的存储位置（CPU或GPU）、数据类型（如整型、浮点型等）以及是否需要梯度（对于自动微分而言）。下面是一些主要的Tensor类型及其作用，以及在PyTorch中的体现方式。

### 存储位置

- **CPU Tensor**：默认创建的Tensor都存储在CPU上。它们适用于不涉及GPU加速的计算任务。
  ```python
  import torch
  x = torch.tensor([1.0, 2.0, 3.0])  # 默认在CPU上
  ```

- **GPU Tensor**：通过将Tensor移动到GPU上，可以利用GPU的并行计算能力来加速计算。这在进行大规模数值计算时非常有用。
  ```python
  if torch.cuda.is_available():
      x = x.cuda()  # 将Tensor移动到GPU上
  ```

### 2. 数据类型

Tensor可以存储多种数据类型，包括但不限于：

- **torch.float32** 或 **torch.float**：32位浮点数，这是最常用的数据类型，用于存储实数，也是标准的pytorch默认的存储格式。
- **torch.float64**：64位浮点数，用于需要更高精度的计算。
- **torch.int32** 或 **torch.int**：32位整数。
- **torch.int64**：64位整数。
- **torch.bool**：布尔类型，用于存储真或假。

在创建Tensor时，可以指定数据类型，使用dtype的属性指定。
```python
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
y = torch.tensor([1, 2, 3], dtype=torch.int32)
```

### 3. 是否需要梯度

- **需要梯度的Tensor**：在构建神经网络时，通常需要计算梯度以更新网络参数。PyTorch通过`torch.Tensor`的`requires_grad`属性来控制是否需要计算梯度。
  ```python
  x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
  # 现在x的梯度将被跟踪，可以用于反向传播
  ```

- **不需要梯度的Tensor**：对于不参与训练过程的Tensor（如输入数据），可以设置为不需要梯度，以减少内存消耗和提高计算效率。默认的tensor都是不带梯度的。
  ```python
  y = torch.tensor([4.0, 5.0, 6.0])  # 默认requires_grad=False
  ```

### 4. 稀疏Tensor

对于某些应用，数据可能非常稀疏（即大多数元素为0）。PyTorch提供了`torch.sparse`模块来高效存储和操作稀疏Tensor。

```python
indices = torch.tensor([[0, 1, 1],
                        [2, 0, 2]], dtype=torch.long)
values = torch.tensor([1, 2, 3], dtype=torch.float32)
shape = [2, 3]
sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
```
现在，我们来具体实操一下：

~~~python
x = torch.rand(3)
print(x.dtype)   # Print the type
~~~
输出结果：
~~~
torch.float32
~~~
接着：
~~~python
# Create a double precision tensor
x = torch.rand(3, dtype=torch.float64)
print(x.dtype)
print(x)
~~~

输出结果：
~~~
torch.float64
tensor([0.1363, 0.8705, 0.3424], dtype=torch.float64)
~~~
整型的我们也实操看看：
~~~python
# Create a tensor with integers
x = torch.LongTensor([3, 6, -11])
print(x.dtype)
print(x)
~~~

输出结果：
~~~
torch.int64
tensor([  3,   6, -11])
~~~

接着我们这么操作数据：
~~~python
# Try dividing the elements with 2
x / 2
~~~

输出结果：
~~~
tensor([ 1.5000,  3.0000, -5.5000])
~~~

~~~python
# That only works for float tensors. Use integer division instead
x // 2
~~~

输出结果：
~~~
tensor([ 1,  3, -6])
~~~
这里，是不是有些奇怪最后为何不是-5，而是-6，这里我给出具体解释：在Python的PyTorch库中，当你对Tensor进行整数除法运算（如使用`//`运算符）时，结果会遵循Python的整数除法规则，即向下取整（也称为向零取整或截断除法）。这意味着，不论被除数是正数还是负数，结果都会向更小的整数方向取整。

在例子中，`tensor([  3,   6, -11]) // 2`：

- 对于`3 // 2`，结果是`1`，因为3除以2的商是1余1，整数除法向下取整。
- 对于`6 // 2`，结果是`3`，这是显而易见的。
- 对于`-11 // 2`，结果是`-6`，而不是`-5`。这是因为-11除以2的商是-5余-1，但在整数除法中，结果会向更小的整数（即更远离0的方向）取整，因此结果是-6。

如果你想要得到商的绝对值最接近的整数（即“四舍五入”到最近的整数，但对于负数来说仍然向零取整），你可能需要使用其他方法，比如结合使用除法、四舍五入和条件逻辑。但是，请注意，标准的四舍五入对于负数来说并不是简单地“向零取整”，而是需要考虑小数部分来决定是向上还是向下取整。不过，对于整数除法来说，我们通常只关注向下取整的行为。

如果你的目标是得到`-5`而不是`-6`，那么按照下面的操作即可：

~~~python
torch.round(x // 2)
~~~

输出结果：
~~~
tensor([ 1,  3, -6])
~~~
下面我们看一个类似实际中操作图像的tensor事例，通常ByteTensor用来表示最高范围到255：

~~~python
# Images often comes as unsigned 8-bit tensors (or Byte tensors)
x = torch.ByteTensor([3, 0, 100, 255])
print(x)
print(x + 1)  # Note that 255 + 1 = 0  (i.e. modulo 256)
~~~

输出结果：
~~~
tensor([  3,   0, 100, 255], dtype=torch.uint8)
tensor([  4,   1, 101,   0], dtype=torch.uint8)
~~~
`torch.ByteTensor`是用于存储8位无符号整数的张量，其值范围从0到255（包含0和255）。这是因为8位无符号整数（通常称为字节）可以表示的最大值是$2^8 - 1 = 255$。图像数据经常以无符号8位张量（或称为字节张量）的形式出现。这是因为在数字图像处理中，颜色通道（如RGB）的像素值通常被量化为0到255之间的整数，以节省存储空间和简化处理。代码尝试将张量`x`中的每个元素加1。然而，由于`x`是无符号8位整数张量，当任何元素达到其最大值255并尝试再加1时，会发生溢出。在无符号整数运算中，这种溢出是通过模256运算来处理的（尽管在大多数现代编程环境中，这种溢出通常不会导致显式的模运算，而是简单地因为二进制表示的翻转而发生）。然而，在这个特定的上下文中，重要的是理解当255加1时，结果“回绕”到0，因为256超出了8位无符号整数的表示范围。因此，`x + 1`的结果将是`tensor([  4,   1, 101,   0], dtype=torch.uint8)`。这里，`255 + 1`确实“等于”0，从数学上讲，这是因为我们在模256的环境下工作，但更直观地说，这是因为我们处理的是固定大小（8位）的无符号整数，并且超出了这个范围的值会“回绕”到最小可能值（即0）。

## 仔细理解上面的代码，理解我所表述的意思，不要嫌我烦和啰嗦，非常有用！！！！！
下面的案例可能很多人在阅读之前，可能会对python中的一个切片操作有疑惑，作为预备知识，试着理解它：
~~~python
# 创建一个形状为(2, 2)的Tensor  
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  
print("A in original status:\n", A)
A_slice1 = A[:, 0]  # 选择所有行（:），第一列（0）  
print("Selecting all rows and the first column:\n", A_slice1)
A_slice2 = A[0, 1]  # 选择第一行（0），第二列（1）  
print("Selecting the first row and the second column:\n", A_slice2)
A_slice3 = A[:, :1]  # 选择所有行（:），但每行只选择第一个元素到第一个元素（即第一列）  
print("Selecting all rows but only the first element of each row:\n", A_slice3)
A_slice4 = A[[0, 1], 1]  # 使用列表来选择第一行和第二行（[0, 1]），然后选择每行的第二个元素（1）  
# 注意：这里实际上等同于A[:, 1]，但为了展示索引的灵活性  
print("Selecting the second element of the first and second rows:\n", A_slice4)
A_slice5 = A[0:1, 1:]  # 选择从第一行开始的第一行（不包括第二行），并选择从第二列开始的所有列  
print("Selecting a submatrix:\n", A_slice5)
~~~

输出结果如下：
~~~
A in original status:
 tensor([[1., 2.],
        [3., 4.]])
Selecting all rows and the first column:
 tensor([1., 3.])
Selecting the first row and the second column:
 tensor(2.)
Selecting all rows but only the first element of each row:
 tensor([[1.],
        [3.]])
Selecting the second element of the first and second rows:
 tensor([2., 4.])
Selecting a submatrix:
 tensor([[2.]])
~~~

接下来，我们再看下面的代码，仔细看它们的dtype：
~~~python
# We can convert from one tensor to another
x_float = 10 * torch.rand(3)
x_long = x_float.long()
x_float2 = x_long.float()

print('x_float =', x_float)
print('x_long =', x_long)
print('x_float2 =', x_float2)
~~~

输出结果：
~~~
x_float = tensor([0.5570, 6.0437, 1.2859])
x_long = tensor([0, 6, 1])
x_float2 = tensor([0., 6., 1.])
~~~
下面是我们经常用到的逐点相乘，这个老能用呢～～～看事例！

~~~python
x = torch.rand(3,4)
y = torch.rand(3,4)
print("x=:{}".format(x))
print("y=:{}".format(y))
# You can play around with what you can do.
# These operations are pointwise
# Add print statements if you want to see the outputs
z = x + y
print("z=x+y=:{}".format(z))
z = x / y
print("z=x/y=:{}".format(z))
z = x**y
print("z=x**y=:{}".format(z))
z = torch.exp(x)
print("z=torch.exp(x)=:{}".format(z))
z = torch.sin(x)
print("z=torch.sin(x)=:{}".format(z))
z = x.exp()
print("z=x.exp()=:{}".format(z))
z = x.round()
print("z=x.round()=:{}".format(z))
z = (x - 0.5).abs()
print("z=(x-0.5).abs()=:{}".format(z))

~~~

输出结果：
~~~
x=:tensor([[0.8647, 0.3250, 0.2807, 0.5767],
        [0.8255, 0.4842, 0.5593, 0.2843],
        [0.1633, 0.1761, 0.5472, 0.1603]])
y=:tensor([[0.0069, 0.5209, 0.9833, 0.9811],
        [0.0078, 0.8744, 0.5664, 0.5293],
        [0.3205, 0.8031, 0.7313, 0.6624]])
z=x+y=:tensor([[0.8716, 0.8460, 1.2640, 1.5578],
        [0.8333, 1.3586, 1.1257, 0.8137],
        [0.4838, 0.9792, 1.2785, 0.8227]])
z=x/y=:tensor([[124.5051,   0.6239,   0.2855,   0.5878],
        [105.4736,   0.5538,   0.9874,   0.5371],
        [  0.5094,   0.2193,   0.7482,   0.2420]])
z=x**y=:tensor([[0.9990, 0.5569, 0.2867, 0.5827],
        [0.9985, 0.5304, 0.7195, 0.5139],
        [0.5594, 0.2479, 0.6434, 0.2974]])
z=torch.exp(x)=:tensor([[2.3743, 1.3841, 1.3241, 1.7802],
        [2.2830, 1.6229, 1.7495, 1.3289],
        [1.1774, 1.1926, 1.7284, 1.1738]])
z=torch.sin(x)=:tensor([[0.7609, 0.3193, 0.2770, 0.5453],
        [0.7349, 0.4655, 0.5306, 0.2805],
        [0.1625, 0.1752, 0.5203, 0.1596]])
z=x.exp()=:tensor([[2.3743, 1.3841, 1.3241, 1.7802],
        [2.2830, 1.6229, 1.7495, 1.3289],
        [1.1774, 1.1926, 1.7284, 1.1738]])
z=x.round()=:tensor([[1., 0., 0., 1.],
        [1., 0., 1., 0.],
        [0., 0., 1., 0.]])
z=(x-0.5).abs()=:tensor([[0.3647, 0.1750, 0.2193, 0.0767],
        [0.3255, 0.0158, 0.0593, 0.2157],
        [0.3367, 0.3239, 0.0472, 0.3397]])
~~~
这里特别特别说明，如果维度是正确的，那么自然而然无任何疑惑，但是如果维度不一致，自然有很多人疑惑？为什么有什么可以借助广播机制进行操作不报错？有时候却不可以？先看一个案例。我们带着疑问解答：
运行下面的代码：
~~~python
x = torch.rand(3,4)
y = torch.rand(3,5)  
print("x=:{}".format(x))
print("y=:{}".format(y))
z = x + y
print("z=x+y=:{}".format(z))
~~~
会产生下面的报错：
~~~
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-22-85dc46862634> in <cell line: 28>()
     26 print("x=:{}".format(x))
     27 print("y=:{}".format(y))
---> 28 z = x + y
     29 print("z=x+y=:{}".format(z))
     30 x = torch.rand(3,4)

RuntimeError: The size of tensor a (4) must match the size of tensor b (5) at non-singleton dimension 1
~~~
但是若修改为这样：
~~~python
x = torch.rand(3,1)
y = torch.rand(3,5)  
print("x=:{}".format(x))
print("y=:{}".format(y))
z = x + y
print("z=x+y=:{}".format(z))
~~~
结果正常输出
~~~
x=:tensor([[0.3661],
        [0.2491],
        [0.3155]])
y=:tensor([[0.8391, 0.5509, 0.3412, 0.1940, 0.2338],
        [0.5364, 0.5692, 0.3935, 0.2058, 0.6261],
        [0.5569, 0.6386, 0.0479, 0.4786, 0.4489]])
z=x+y=:tensor([[1.2052, 0.9170, 0.7073, 0.5601, 0.5999],
        [0.7855, 0.8183, 0.6426, 0.4549, 0.8751],
        [0.8724, 0.9541, 0.3635, 0.7941, 0.7644]])
~~~
再看一个案例：
~~~python
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 形状为 (2, 2)，但这里为了演示，我们改为(2, 1)  
print("A in original status:{}".format(A))
A = A[:, :1]  # 现在A的形状是 (2, 1)  
print("A:{}".format(A))
B = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 形状为 (2, 3)  
print("B:{}".format(B))
# 由于A的形状是(2, 1)，而B的形状是(2, 3)，PyTorch会在A的第二个维度上自动扩展（广播）A，  
# 使其形状变为(2, 3)，以便与B进行元素对元素的加法。  
# 执行加法操作  
C = A + B  
print(C)  
~~~

输出结果：
~~~
A in original status:tensor([[1., 2.],
        [3., 4.]])
A:tensor([[1.],
        [3.]])
B:tensor([[1., 2., 3.],
        [4., 5., 6.]])
tensor([[2., 3., 4.],
        [7., 8., 9.]])
~~~
这几个案例看，为何一个是x=torch.rand(3,4)不能和y=torch.rand(3,5)进行相加或其他操作，但是x=torch.rand(3,1)可以和y=torch.rand(3,5)进行相加或其他操作，同样上面例子形状（2，1）可以和（2，3）进行相加或其他操作，原因是什么呢？
下面进行解释：
在PyTorch中，两个维度不同的tensor能否通过广播（Broadcasting）进行相加，主要取决于它们的维度和在这些维度上的尺寸是否满足广播的规则。广播规则允许tensor在某些维度上自动扩展（或称为“广播”）以匹配另一个tensor的形状，但这种扩展是有条件的。
### 可以广播的情况

两个tensor可以在以下情况下通过广播进行相加：

1. **维度数量不同**：形状较小的tensor会在前面补充1，直到它们的维度数量相同。然后，检查每个维度上的尺寸是否兼容。

2. **维度尺寸相同或其中一个为1**：对于两个tensor的每个维度，如果它们的尺寸相同，或者其中一个tensor的尺寸为1，则这两个tensor在这一维度上兼容。

然而，在给出的例子中：

```python
x = torch.rand(3, 4)
y = torch.rand(3, 5)
```

这两个tensor的维度数量是相同的（都是2维），但在第二个维度上它们的尺寸不同（4和5）。由于没有一个维度上的尺寸是1，且所有对应维度的尺寸都不相同，因此这两个tensor**无法通过广播进行相加**。

### 无法操作的情况

- 当两个tensor在对应维度上的尺寸都不为1且不相等时，无法进行广播。
- 如果尝试对这样的tensor进行算术运算（如加法），PyTorch会抛出一个错误，指出尺寸不匹配。
在PyTorch中，两个维度不同的tensor能否通过广播进行相加，主要取决于它们的维度和在这些维度上的尺寸是否满足广播的规则。下面我会详细解释您给出的两个例子，并再举几个额外的案例来帮助您理解。

### 第一个例子：x = torch.rand(3,4) 和 y = torch.rand(3,5)

**无法广播相加的情况**：
- 在这个例子中，`x` 和 `y` 都是二维tensor，但它们的第二个维度（即列数）不同（`x` 有4列，`y` 有5列）。由于两个tensor在对应维度上的尺寸都不为1且不相等，因此无法通过广播来相加。

### 第二个例子：形状为（2，1）和（2，3）的tensor

**可以广播相加的情况**：
- 假设有两个tensor `A` 和 `B`，其中 `A` 的形状是 `(2, 1)`，`B` 的形状是 `(2, 3)`。
- 在这种情况下，`A` 的第二个维度（列数）为1，而 `B` 的第二个维度为3。根据广播规则，`A` 的第二个维度会在广播过程中自动“扩展”以匹配 `B` 的第二个维度。
- 因此，`A` 会被视为一个 `(2, 3)` 的tensor，其中每一行的第二个和第三个元素都是第一个元素的副本，然后这个“扩展”后的 `A` 会与 `B` 进行元素对元素的相加。

### 额外案例

**案例1：形状为（1，2）和（3，2）的tensor**
- 假设有tensor `C` 形状为 `(1, 2)` 和tensor `D` 形状为 `(3, 2)`。
- `C` 的第一个维度（行数）为1，而 `D` 的第一个维度为3。根据广播规则，`C` 的第一个维度会在广播过程中自动“扩展”到3，以匹配 `D` 的第一个维度。
- 因此，`C` 会被视为一个 `(3, 2)` 的tensor，其中每一行都是原始 `C` 的行的副本，然后这个“扩展”后的 `C` 会与 `D` 进行元素对元素的相加。

**案例2：形状为（2，）和（2，3）的tensor（注意：第一个tensor是一维的）**
- 假设有tensor `E` 形状为 `(2,)`（一维，两个元素）和tensor `F` 形状为 `(2, 3)`。
- 在这种情况下，`E` 会被视为一个 `(2, 1)` 的tensor（在其前面添加一个维度，该维度的大小为1），以便与 `F` 的维度数量相匹配。
- 然后，就像第二个例子一样，`E` 的第二个维度（现在是1）会在广播过程中自动“扩展”到3，以匹配 `F` 的第二个维度。
- 最终，`E` 会被视为一个 `(2, 3)` 的tensor，其中每一行的所有元素都是原始 `E` 的元素的副本，然后这个“扩展”后的 `E` 会与 `F` 进行元素对元素的相加。

通过这些案例，您应该能够更清楚地理解PyTorch中的广播机制以及它如何影响不同形状的tensor之间的操作。
### Reducing Operations【一定要弄懂哦！】
下面看看这个案例：
~~~python
x = torch.rand(3,4)
print("x:{}".format(x))
# Sum all elements
print("torch.sum(x):{}".format(torch.sum(x)))
print("x.sum():{}".format(x.sum()))

print()

# Sum along a dimension
print(x.sum(dim=0))
print(x.sum(dim=1))

print()

# Keep dimensions after summing
print(x.shape)
print(x.sum(dim=0).shape)
print(x.sum(dim=0, keepdim=True).shape)
~~~

输出结果：
~~~
x:tensor([[0.9599, 0.7650, 0.4798, 0.6208],
        [0.4043, 0.5076, 0.7281, 0.3456],
        [0.6186, 0.1179, 0.6605, 0.7092]])
torch.sum(x):6.917286396026611
x.sum():6.917286396026611

tensor([1.9827, 1.3905, 1.8684, 1.6757])
tensor([2.8254, 1.9856, 2.1063])

torch.Size([3, 4])
torch.Size([4])
torch.Size([1, 4])
~~~
下面再看一个示例：
~~~python
x = torch.rand(3,4)
# You can calso do a mean
print('Mean: ', x.mean())
# Max and min works in a similar way
print('Min: ', x.min())
print('Max: ', x.max())
# You can use .item() to convert a scalar tensor to just a number
s = x.sum()
print(s)
print(s.item())
# With max/min along a dimension you also get the argmax/argmin
max_val, arg_max = x.max(dim=0)
print(max_val)
print(arg_max)
~~~

输出结果：
~~~
Mean:  tensor(0.6576)
Min:  tensor(0.2126)
Max:  tensor(0.9726)
tensor(7.8912)
7.891239643096924
tensor([0.9124, 0.7825, 0.9726, 0.9187])
tensor([2, 1, 0, 1])
~~~
上面的代码中可以看到最后的一行已经出现了索引，就是返回了最大值以及对应的索引，这个在深度学习中将成为最频繁的操作。
## 交换维度【超高频率的代码操作】
下面我们看一段代码，并仔细查看输出结果：
~~~python
# Next we will try to swap (transpose) dimensions

x1 = torch.rand(3,4)

# The transpose function swaps the indicated dimensions
# In this case, this corresponds to standard matrix transpose

x2 = x1.transpose(0,1)

# Lets compare with the effect of reshaping
x3 = x1.view(4,3)

print(x1)
print()
print(x2)
print()
print(x3)
~~~

输出结果：
~~~
tensor([[0.3338, 0.2846, 0.5354, 0.0095],
        [0.4081, 0.0094, 0.5169, 0.5800],
        [0.0808, 0.3443, 0.5486, 0.3742]])

tensor([[0.3338, 0.4081, 0.0808],
        [0.2846, 0.0094, 0.3443],
        [0.5354, 0.5169, 0.5486],
        [0.0095, 0.5800, 0.3742]])

tensor([[0.3338, 0.2846, 0.5354],
        [0.0095, 0.4081, 0.0094],
        [0.5169, 0.5800, 0.0808],
        [0.3443, 0.5486, 0.3742]])
~~~
发现什么区别没？x2和x3一样吗？其实，我们经常用到的是permute函数进行维度的交换：
~~~python
# The permute function is more general and can swap multiple dimensions at the same time.

x1 = torch.rand(2,3,4)
print("x1:{}".format(x1))
x2 = x1.permute(2,1,0)   # Input the new order of dimensions
print("x2 permute after:{}".format(x2))
print(x1.shape)
print(x2.shape)
~~~

输出结果：
~~~
[18]
0 秒
# The permute function is more general and can swap multiple dimensions at the same time.

x1 = torch.rand(2,3,4)
print("x1:{}".format(x1))
x2 = x1.permute(2,1,0)   # Input the new order of dimensions
print("x2 permute after:{}".format(x2))
print(x1.shape)
print(x2.shape)
x1:tensor([[[0.3413, 0.6809, 0.5446, 0.9499],
         [0.1116, 0.4533, 0.2707, 0.3171],
         [0.5107, 0.0284, 0.6996, 0.9481]],

        [[0.1904, 0.3109, 0.0377, 0.9311],
         [0.1485, 0.2117, 0.0373, 0.4536],
         [0.8732, 0.5178, 0.9100, 0.9925]]])
x2 permute after:tensor([[[0.3413, 0.1904],
         [0.1116, 0.1485],
         [0.5107, 0.8732]],

        [[0.6809, 0.3109],
         [0.4533, 0.2117],
         [0.0284, 0.5178]],

        [[0.5446, 0.0377],
         [0.2707, 0.0373],
         [0.6996, 0.9100]],

        [[0.9499, 0.9311],
         [0.3171, 0.4536],
         [0.9481, 0.9925]]])
torch.Size([2, 3, 4])
torch.Size([4, 3, 2])
~~~
仔细揣摩这个案例。
### 索引
我们上面玩儿到一个索引的案例，下面我们接着看相关的案例，看看你到底掌握没有？
~~~python
x = torch.rand(3,4,5)
print("x:{}".format(x))
# Index a single value
print(x[2,0,1])

# Slice a dimension. : means 'everything'
print(x[2,0,:])

# or multiple
print(x[:,0,:])

# Slice with a range
print(x[2, 1:, 2:-1])

# Tripple dot ... means 'all remaining dimensions'
print(x[1, ...])
~~~

输出结果：
~~~
x:tensor([[[0.2317, 0.0064, 0.0212, 0.8653, 0.1282],
         [0.8069, 0.9631, 0.6445, 0.8236, 0.1167],
         [0.9232, 0.8366, 0.5666, 0.3414, 0.7852],
         [0.9931, 0.4877, 0.0647, 0.6039, 0.7916]],

        [[0.8424, 0.4851, 0.3090, 0.8354, 0.9139],
         [0.5210, 0.1660, 0.8940, 0.6825, 0.0226],
         [0.5048, 0.5746, 0.1512, 0.6918, 0.2154],
         [0.6631, 0.3200, 0.1501, 0.5838, 0.9456]],

        [[0.0773, 0.8339, 0.3463, 0.9860, 0.6884],
         [0.7520, 0.6714, 0.8818, 0.5276, 0.0030],
         [0.9656, 0.2639, 0.7806, 0.3129, 0.0603],
         [0.0108, 0.2606, 0.9180, 0.6320, 0.3697]]])
tensor(0.8339)
tensor([0.0773, 0.8339, 0.3463, 0.9860, 0.6884])
tensor([[0.2317, 0.0064, 0.0212, 0.8653, 0.1282],
        [0.8424, 0.4851, 0.3090, 0.8354, 0.9139],
        [0.0773, 0.8339, 0.3463, 0.9860, 0.6884]])
tensor([[0.8818, 0.5276],
        [0.7806, 0.3129],
        [0.9180, 0.6320]])
tensor([[0.8424, 0.4851, 0.3090, 0.8354, 0.9139],
        [0.5210, 0.1660, 0.8940, 0.6825, 0.0226],
        [0.5048, 0.5746, 0.1512, 0.6918, 0.2154],
        [0.6631, 0.3200, 0.1501, 0.5838, 0.9456]])
~~~
我们接着看：
~~~python
x = torch.rand(3,4)
print("x: {}".format(x))
# We can use LongTensor to index out a list of specific values/rows/columns

ind_col = torch.LongTensor([1,1,0,3,-1,-2])   # Index these columns pls
print("ind_col:{}".format(ind_col))
print(x[:, ind_col])


# If we also want to index specific rows, we need to match the shapes

ind_row = torch.LongTensor([2,0])   # Index these rows
ind_row_view=ind_row.view(-1,1)
ind_col_view=ind_col.view(1,-1)
print("ind_row:{}".format(ind_row_view))
print("ind_col:{}".format(ind_col_view))
print(x[ind_row_view, ind_col_view])
~~~

输出结果：
~~~
x: tensor([[0.6129, 0.9581, 0.4423, 0.6743],
        [0.8745, 0.6963, 0.7569, 0.6855],
        [0.9514, 0.8436, 0.1435, 0.5188]])
ind_col:tensor([ 1,  1,  0,  3, -1, -2])
tensor([[0.9581, 0.9581, 0.6129, 0.6743, 0.6743, 0.4423],
        [0.6963, 0.6963, 0.8745, 0.6855, 0.6855, 0.7569],
        [0.8436, 0.8436, 0.9514, 0.5188, 0.5188, 0.1435]])
ind_row:tensor([[2],
        [0]])
ind_col:tensor([[ 1,  1,  0,  3, -1, -2]])
tensor([[0.8436, 0.8436, 0.9514, 0.5188, 0.5188, 0.1435],
        [0.9581, 0.9581, 0.6129, 0.6743, 0.6743, 0.4423]])
~~~
接下来看一个比较绕一些的bool型操作：
~~~python
x = torch.rand(3,4)
print("x:{}".format(x))
# Finally, we do some logical indexing

logical_ind_col = torch.BoolTensor([True, False, False, True])   # Boolean tensor indicating wich columns to keep

print(x[:, logical_ind_col])

# The length of logical indices must match the size of the corresponding dimension (4 in this case)

# Logical indexing is very useful in many cases
# For example, say we just want to keep columns whose average is larger than 0.5:

x2 = x[:, x.mean(dim=0) > 0.5]

print(x2)

print(x.mean(dim=0) > 0.5)  # Lets also check what the index looks like
~~~

输出结果：
~~~
x:tensor([[0.9911, 0.8024, 0.5165, 0.3211],
        [0.4650, 0.0121, 0.7393, 0.4793],
        [0.6919, 0.7944, 0.7805, 0.2905]])
tensor([[0.9911, 0.3211],
        [0.4650, 0.4793],
        [0.6919, 0.2905]])
tensor([[0.9911, 0.8024, 0.5165],
        [0.4650, 0.0121, 0.7393],
        [0.6919, 0.7944, 0.7805]])
tensor([ True,  True,  True, False])
~~~
下面我们看这个案例：
~~~python
x = torch.rand(3,4)
y = 5*torch.ones(2,4)
print("x:{}".format(x))
print("y:{}".format(y))
# Concatenate x and y along the first dimension (dim=0)
z1 = torch.cat([x, y], dim=0)  # The shapes, except in dim=0, must match
print(z1)

# Stacking, on the other hand, creates a new dimension
# Here, all dimensions must match
z2 = torch.stack([x, x], dim=1)
print("z2:{}".format(z2))
print(z2.shape)

# Nicely combined with list comprehension
z3 = torch.cat([n*torch.ones(3,n) for n in range(5)], dim=-1)
print(z3)
~~~

输出结果：
~~~
x:tensor([[0.1018, 0.5091, 0.1645, 0.2813],
        [0.6307, 0.4556, 0.8681, 0.7102],
        [0.9634, 0.4840, 0.6053, 0.3643]])
y:tensor([[5., 5., 5., 5.],
        [5., 5., 5., 5.]])
tensor([[0.1018, 0.5091, 0.1645, 0.2813],
        [0.6307, 0.4556, 0.8681, 0.7102],
        [0.9634, 0.4840, 0.6053, 0.3643],
        [5.0000, 5.0000, 5.0000, 5.0000],
        [5.0000, 5.0000, 5.0000, 5.0000]])
z2:tensor([[[0.1018, 0.5091, 0.1645, 0.2813],
         [0.1018, 0.5091, 0.1645, 0.2813]],

        [[0.6307, 0.4556, 0.8681, 0.7102],
         [0.6307, 0.4556, 0.8681, 0.7102]],

        [[0.9634, 0.4840, 0.6053, 0.3643],
         [0.9634, 0.4840, 0.6053, 0.3643]]])
torch.Size([3, 2, 4])
tensor([[1., 2., 2., 3., 3., 3., 4., 4., 4., 4.],
        [1., 2., 2., 3., 3., 3., 4., 4., 4., 4.],
        [1., 2., 2., 3., 3., 3., 4., 4., 4., 4.]])
~~~
好了，这些操作之后，我们开始一日速成系列的pytorch操作图像等内容，关注新的博客！

