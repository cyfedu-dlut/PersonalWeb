---
title: '短时间内快速认识和掌握向量乘积运算'
date: 2024-09-11
permalink: /posts/2024/09/tensormul/
tags:
  - tensor operation
  - technical skills
---

在PyTorch中，向量和矩阵之间的不同乘积操作非常关键，尤其是在进行深度学习模型构建和数学运算时。下面，我将详细解释您提到的几种乘积类型，并提供具体的代码示例和使用场景。

### 1. Point-wise Production（逐点乘积）/ Element-wise Production（元素级乘积）

**解释**：这两个术语实际上是相同的，指的是两个形状相同的向量或矩阵之间对应元素的乘积。结果矩阵中的每个元素都是原矩阵对应位置元素的乘积。

**代码示例**：

```python
import torch

# 创建两个形状相同的矩阵
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# 逐点/元素级乘积
c = a * b

print(c)
# 输出:
# tensor([[ 5, 12],
#        [21, 32]])
```

**使用场景**：这种操作在需要对两个信号进行元素级处理时非常有用，比如在进行图像处理时调整图像的亮度和对比度。

### 2. Hadamard Product（哈达玛积）

**解释**：Hadamard积其实就是逐点乘积（element-wise production）的另一种说法，两者是完全相同的。

**代码示例**：同上。

### 3. Batch Matrix Multiplication (bmm)

**解释**：`bmm`（Batch Matrix Multiplication）是批矩阵乘法，用于计算一批矩阵的乘积。输入是三维张量，其中前两个维度可以看作是“批量”维度，而最后一个维度是矩阵的维度。输出同样是三维张量，形状与输入的第一个和最后一个维度有关。

**代码示例**：

```python
import torch

# 创建两个三维张量
A = torch.randn(10, 3, 4)  # [batch_size, m, n]
B = torch.randn(10, 4, 5)  # [batch_size, n, p]

# 计算批矩阵乘法
C = torch.bmm(A, B)

print(C.shape)
# 输出: torch.Size([10, 3, 5])
```

**使用场景**：在处理具有批量数据（如批量图像或时间序列数据）的深度学习模型时，经常需要用到批矩阵乘法。

## 在PyTorch和更广泛的数学与计算机科学领域中，除了之前提到的逐点/元素级乘积、Hadamard积和批矩阵乘法外，还有许多其他重要的数学运算。以下是一些常见的数学运算及其简要说明和可能的使用场景：

### 1. **矩阵乘法（Matrix Multiplication）**：
   - 不同于批矩阵乘法，这里的矩阵乘法指的是两个二维矩阵之间的乘法。结果矩阵的维度由输入矩阵的维度决定（即，如果A是m×n矩阵，B是n×p矩阵，则A乘以B的结果是m×p矩阵）。
   - 使用场景：线性变换、图像处理中的仿射变换、神经网络中的全连接层等。

PyTorch代码示例:
```python
import torch

A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.matmul(A, B)  # 或者使用 @ 运算符: C = A @ B

print(C.shape)  # 输出: torch.Size([2, 4])
```

### 2. **点积（Dot Product）/ 内积（Inner Product）**：
   - 也称为标量积，是两个等长向量对应元素乘积的和。结果是一个标量。
   - 使用场景：计算两个向量之间的相似度、计算投影长度、神经网络中的注意力机制等。
对于向量（一维张量），PyTorch没有直接的点积函数，但可以使用`torch.dot`（仅适用于一维张量）或`torch.matmul`（对于更高维张量，需要确保它们可以相乘）。

PyTorch代码示例:
```python
a = torch.randn(3)
b = torch.randn(3)
dot_product = torch.dot(a, b)  # 或者使用 matmul: dot_product = torch.matmul(a.unsqueeze(0), b.unsqueeze(1)).item()

print(dot_product)  # 输出: 一个标量
```

对于二维张量（矩阵），点积通常指的是矩阵乘法的一个特例（即一个向量乘以另一个向量的转置）。

### 3. **外积（Outer Product）**：
   - 不同于点积，外积的结果是一个矩阵，而不是标量。对于两个向量a和b，它们的外积是一个矩阵，其中每个元素是a中对应元素与b中对应元素的乘积。
   - 使用场景：在量子计算、线性代数和某些类型的矩阵分解中。

PyTorch没有直接的外积函数，但可以通过扩展维度和矩阵乘法来实现。

PyTorch代码示例:
```python
a = torch.randn(3)
b = torch.randn(4)
outer_product = torch.ger(a, b)  # PyTorch 1.9之前版本有ger，但新版本中推荐使用unsqueeze和matmul
# 或者使用unsqueeze和matmul:
outer_product = a.unsqueeze(1) * b.unsqueeze(0)

print(outer_product.shape)  # 输出: torch.Size([3, 4])
```
注意：`torch.ger`在PyTorch的新版本中可能已被弃用，因此建议使用`unsqueeze`和`matmul`的方法。


### 4. **克罗内克积（Kronecker Product）**：
   - 也称为直积或张量积，是两个任意大小矩阵之间的运算，结果是一个更大的矩阵。
   - 使用场景：在控制理论、统计学和量子计算中。

PyTorch没有内置的克罗内克积函数，但可以通过NumPy实现并转换为PyTorch张量。

NumPy代码示例:
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.kron(A, B)

print(C.shape)  # 输出: (4, 4)
# 如果需要PyTorch张量:
C_tensor = torch.tensor(C)
```


### 5. **矩阵的逆（Matrix Inversion）**：
   - 对于一个方阵（即行数和列数相等的矩阵），如果存在一个矩阵使得原矩阵与该矩阵的乘积是单位矩阵，则称该矩阵为原矩阵的逆矩阵。
   - 使用场景：求解线性方程组、计算矩阵的伪逆（在最小二乘问题中）、在机器学习的某些算法中（如PCA）。

PyTorch代码示例:
```python
import torch

A = torch.randn(3, 3)
A_inv = torch.inverse(A)  # 注意：仅当A是可逆的时才有效

# 或者使用torch.linalg.inv（PyTorch 1.8+）
A_inv_linalg = torch.linalg.inv(A)

print(A_inv.shape)  # 输出: torch.Size([3, 3])
```

### 6. **矩阵的转置（Matrix Transpose）**：
   - 将矩阵的行和列互换得到的矩阵称为原矩阵的转置矩阵。
   - 使用场景：在求解线性方程组、计算协方差矩阵、图像处理中的翻转操作等。

PyTorch代码示例:
```python
A = torch.randn(2, 3)
A_T = A.T  # 或者使用 torch.transpose(A, 0, 1)

print(A_T.shape)  # 输出: torch.Size([3, 2])
```


### 7. **矩阵的行列式（Determinant of a Matrix）**：
   - 方阵的一个标量值，与矩阵的某些性质（如是否可逆）有关。
   - 使用场景：在几何学中计算面积或体积、在控制理论中分析系统的稳定性等。


PyTorch代码示例:
```python
A = torch.randn(3, 3)
det = torch.det(A)  # 或者使用 torch.linalg.det（PyTorch 1.8+）

print(det)  # 输出: 一个标量
```

### 8. **特征值和特征向量（Eigenvalues and Eigenvectors）**：
   - 对于一个方阵A，如果存在一个非零向量v和一个标量λ，使得Av = λv，则称λ为A的特征值，v为对应的特征向量。
   - 使用场景：在物理学、工程学、经济学和计算机科学中的许多领域，特别是在分析系统的动态行为时。

PyTorch代码示例:
```python
A = torch.randn(3, 3)
eigenvalues, eigenvectors = torch.linalg.eigh(A)  # 对于对称/Hermitian矩阵
# 或者对于非对称矩阵，使用 torch.linalg.eig
eigenvalues_general, eigenvectors_general = torch.linalg.eig(A)

print(eigenvalues)  # 输出: 特征值
print(eigenvectors)  # 输出: 特征向量
```

### 9. **广播（Broadcasting）**：
   - 虽然不是传统意义上的数学运算，但它是PyTorch等库中进行元素级运算时的一个重要概念。它允许NumPy风格的数组操作在形状不完全相同的数组之间进行。
   - 使用场景：在进行元素级运算时，自动扩展较小数组的形状以匹配较大数组的形状。

广播不是一个具体的函数，而是PyTorch（和NumPy）中自动处理的一种机制。

PyTorch:
```python
a = torch.randn(1, 3)
b = torch.randn(4, 3)
c = a + b  # 这里a的第一维被广播到与b的第一维相同的大小

print(c.shape)  # 输出: torch.Size([4, 3])
```

# 【注】每种操作都有其特定的使用场景，正确选择可以显著提高模型的性能和效率。