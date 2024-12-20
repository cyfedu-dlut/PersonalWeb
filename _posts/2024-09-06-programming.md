---
title: '【一日速成编程系列】用Pytorch操作图像'
date: 2024-09-06
permalink: /posts/2024/09/programming/
tags:
  - PyTorch
  - Python
  - Programming
---

通过下面的事例，让你分分钟入门Pytorch操作图像进行计算机视觉领域的入门

## 设置路径
 设置一个本地可以访问的图像的路径：
~~~python
dataset = '/content/gdrive/MyDrive/'
~~~

## 导入必要的库
~~~python
# Import libraries we will need
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
~~~

## 读取并显示图片
~~~python
# We will use PIL to read an image
im_pil = Image.open(dataset + 'scene.jpg')

# We first convert the image to a numpy array
im_np = np.array(im_pil)

# Numpy is the standard matrix library in Python. 
# But PyTorch effectively replaces it together with the functionality needed for deep learning.
# Still, we will enounter Numpy arrays in intermediate stages.
# Similar to PyTorch, we can check the shape of the tensor.
im_np.shape
~~~
这里我们使用下面这张图片作为输入，其分辨率为1278x1706
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1bcf843c911442d2b5e3284805f97328.jpeg#pic_center)
我们将用这张图像作为操作对象，首先使用 PIL（Python Imaging Library，或其更新版本 Pillow）的 `Image.open()` 方法来打开一个图片文件，并将这个图片对象存储在变量 `im_pil` 中。接着，使用 NumPy 的 `np.array()` 函数将这个 PIL 图片对象转换成了一个 NumPy 数组，存储在变量 `im_np` 中。最后，通过 `im_np.shape` 获取了这个 NumPy 数组的形状（shape）。

让我们逐步分析这段代码：

1. **打开图片文件**：
   ```python
   im_pil = Image.open(dataset + 'scene.jpg')
   ```
   这里，`dataset` 应该是一个字符串，表示图片文件所在的目录路径（不包括文件名）。`'scene.jpg'` 是你想要打开的图片文件名。通过 `dataset + 'scene.jpg'`，你构造了一个完整的文件路径，然后使用 `Image.open()` 方法打开这个文件。注意，如果 `dataset` 字符串的末尾没有斜杠（`/` 或 `\`，取决于你的操作系统），这将会导致路径错误。通常，你应该确保 `dataset` 变量以适当的目录分隔符结束，或者在拼接时显式添加它：
   ```python
   im_pil = Image.open(dataset + '/' + 'scene.jpg')  # 对于Unix/Linux/macOS
   # 或者
   im_pil = Image.open(os.path.join(dataset, 'scene.jpg'))  # 使用os.path.join更安全
   ```

2. **将PIL图片转换为NumPy数组**：
   ```python
   im_np = np.array(im_pil)
   ```
   这里，`np.array()` 函数将 PIL 图片对象 `im_pil` 转换为一个 NumPy 数组 `im_np`。这个数组包含了图片的像素数据，其中每个元素代表图片中的一个像素点。对于彩色图片，这通常是一个三维数组，其形状为 `(高度, 宽度, 颜色通道数)`。颜色通道数取决于图片的类型（例如，RGB图片有3个颜色通道：红、绿、蓝）。

3. **获取NumPy数组的形状**：
   ```python
   im_np.shape
   ```
   这行代码将输出 NumPy 数组 `im_np` 的形状。对于一个典型的 RGB 图片，如果它的尺寸是 800x600 像素，那么 `im_np.shape` 的输出将会是 `(600, 800, 3)`。这里，`600` 是图片的高度（像素数），`800` 是图片的宽度（像素数），`3` 是颜色通道数（RGB）。

最终的图像的shape是：(1278, 1706, 3)
我们也可以用plot进行展示，代码如下：
~~~python
# Now lets display the image. We can do this with matplotlib
plt.imshow(im_np)   # Plot image
plt.axis('on')     # Just turns on the axis
plt.show()          # Finally show it
~~~
输出结果如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71d33018f28045889f243ee41886ce53.png)下面是一个标准的预处理步骤，在pytorch框架中经常看到的预处理操作图像的代码，实际使用会使用torchvision进行处理，这里给出一个基本的操作：
~~~python
# We can easily convert the image from numpy to PyTorch
im_torch = torch.from_numpy(im_np)

# Check shape
print(im_torch.shape)

# By default, PyTorch uses the data order C x H x W for images.
# So we should move the RGB channel dimension to the first dimension
im_torch = im_torch.permute(2, 0, 1)

print(im_torch.shape)
~~~

输出结果如下，可以看到图像转为numpy，numpy转为pytorch的数据，用来成为pytorch可以操作的输入。
~~~
torch.Size([1278, 1706, 3])
torch.Size([3, 1278, 1706])
~~~
我们现在看看具体的数据类型：
~~~python
# Lets check some details about the image
print('Type:', im_torch.dtype)
print('Min:', im_torch.min().item())
print('Max:', im_torch.max().item())

# Note that the image is a Byte tensor (uint8)
# For most image operations it is better to first convert it to floating point with values between 0 and 1

im = im_torch.float() / 255

print()
print('Type:', im.dtype)
print('Min:', im.min().item())
print('Max:', im.max().item())
~~~
输出结果如下：
~~~
Type: torch.uint8
Min: 0
Max: 255

Type: torch.float32
Min: 0.0
Max: 1.0
~~~
当然，上面的步骤都是一些没有标准和复用性不高的代码写法，作为programmer我应该进行合理有效的封包：
~~~python 
# Read an image with the given name and convert it to torch
def imread(image_file):
    im_pil = Image.open(dataset + image_file)
    im_np = np.array(im_pil, copy=False)
    im_torch = torch.from_numpy(im_np).permute(2, 0, 1)
    return im_torch.float()/255

# Show a PyTorch image tensor
def imshow(im, normalize=False):
    # Fit the image to the [0, 1] range if normalize is True
    if normalize:
        im = (im - im.min()) / (im.max() - im.min())

    # Remove redundant dimensions 
    im = im.squeeze()    # Mini excersize: check in the documentation what this function does

    is_color = (im.dim() == 3)

    # If there is a color channel dimension, move it to the end
    if is_color:
        im = im.permute(1, 2, 0)

    im_np = im.numpy().clip(0,1)    # Convert to numpy and ensure the values in the range [0, 1]
    if is_color:
        plt.imshow(im_np)
    else:
        plt.imshow(im_np, cmap='gray')
    plt.axis('off')
    plt.show()
~~~

如下是详细的代码注释，这样一目了然：
~~~python
{# 读取给定名称的图片，并将其转换为PyTorch张量  
def imread(image_file):  
    # 使用PIL库打开数据集路径与给定文件名拼接后的图片文件  
    im_pil = Image.open(dataset + image_file)  
    # 将PIL图片对象转换为NumPy数组，设置copy=False以避免数据复制，提高效率  
    im_np = np.array(im_pil, copy=False)  
    # 将NumPy数组转换为PyTorch张量，并调整维度顺序以匹配PyTorch的CHW（通道、高度、宽度）格式  
    im_torch = torch.from_numpy(im_np).permute(2, 0, 1)  
    # 将张量数据类型转换为浮点型，并除以255将其值归一化到[0, 1]区间  
    return im_torch.float()/255  
  
# 显示一个PyTorch图像张量  
def imshow(im, normalize=False):  
    # 如果normalize为True，则将图像张量的值缩放到[0, 1]区间  
    if normalize:  
        im = (im - im.min()) / (im.max() - im.min())  
  
    # 移除张量中多余的维度（即维度大小为1的维度）  
    im = im.squeeze()   
  
    # 判断图像是否为彩色图  
    is_color = (im.dim() == 3)  
  
    # 如果图像是彩色的，将颜色通道维度移动到末尾，以匹配matplotlib的HWC（高度、宽度、通道）格式  
    if is_color:  
        im = im.permute(1, 2, 0)  
  
    # 将PyTorch张量转换为NumPy数组，并确保其值在[0, 1]区间内  
    im_np = im.numpy().clip(0,1)  
    # 根据图像是否为彩色图，使用不同的imshow参数显示图像  
    if is_color:  
        plt.imshow(im_np)  
    else:  
        plt.imshow(im_np, cmap='gray')  # 灰度图使用灰度颜色映射  
    # 关闭坐标轴显示  
    plt.axis('off')  
    # 显示图像  
    plt.show()  
}
~~~
这段代码中有两个重要的知识点：
首先是：`im = im.squeeze()` 
在PyTorch中的作用是移除张量中所有大小为1的维度（也称为单元素维度或冗余维度）。这在处理图像数据或其他类型的多维数据时非常有用，因为在进行某些操作（如扩展维度、广播等）后，张量可能会包含一些不必要的单元素维度。

### 具体作用

- **减少内存占用**：移除不必要的维度可以减少张量在内存中的占用。
- **简化操作**：在处理数据时，单元素维度的存在可能会使得后续的索引、切片或计算变得更加复杂。通过移除这些维度，可以使得后续操作更加直观和简单。

### 举例说明

假设我们有一个PyTorch张量 `im`，它表示一个灰度图像（没有颜色通道），但在进行某些操作后，其形状变成了 `(1, 1, 28, 28)`，其中：

- 第一个维度大小为1，可能是在之前添加了一个批处理维度但只处理了一个图像。
- 第二个维度大小为1，是一个冗余的维度，因为灰度图像没有颜色通道。
- 后两个维度 `(28, 28)` 表示图像的高度和宽度。

我们可以使用 `squeeze()` 方法来移除这些大小为1的维度：

```python
import torch

# 假设的原始张量
im = torch.randn(1, 1, 28, 28)  # 形状为 (1, 1, 28, 28)

# 移除大小为1的维度
im_squeezed = im.squeeze()

# 检查结果
print(im_squeezed.shape)  # 输出应为 torch.Size([28, 28])
```

在这个例子中，`squeeze()` 方法移除了 `im` 张量中的前两个大小为1的维度，因此 `im_squeezed` 的形状变为了 `(28, 28)`，这是一个更自然和紧凑的表示灰度图像的方式。

另外，如果只想移除特定位置的维度，可以通过 `squeeze(dim)` 的形式来指定要移除的维度，其中 `dim` 是要移除的维度的索引（从0开始计数）。但是，在这个例子中，我们使用了不带参数的 `squeeze()`，它会移除所有大小为1的维度。

注意：在某些情况下，可能需要保留某些大小为1的维度以保留数据的“批处理”或“通道”等信息的语义。在这种情况下，就不应该使用 `squeeze()` 或应该谨慎地选择需要移除的维度。

另外一个重要的知识点：
### `clip` 方法的作用

`clip` 方法是NumPy数组的一个非常有用的函数，它接受两个参数：最小值和最大值。该方法会遍历数组中的每个元素，如果元素的值小于最小值，则将其替换为最小值；如果元素的值大于最大值，则将其替换为最大值；如果元素的值在最小值和最大值之间（包括这两个值），则保持不变。

在这个特定的例子中，`clip(0,1)` 确保了转换后的NumPy数组 `im_np` 中的所有值都在0到1的范围内，这对于图像数据来说是非常常见的需求，因为图像像素值通常被归一化到这个范围内以便于处理。

### 举例说明

假设我们有一个PyTorch张量 `im`，它表示一个已经经过某种预处理（可能是归一化到某个非[0,1]范围）的图像数据。我们想要将这个张量转换为NumPy数组，并确保数组中的所有值都在[0,1]范围内，以便我们可以使用matplotlib或其他库来显示图像。

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# 假设的PyTorch张量，其值可能不在[0,1]范围内
im = torch.tensor([[-0.5, 1.5, 0.0], [2.0, -0.2, 0.8]])

# 将PyTorch张量转换为NumPy数组，并使用clip方法限制值在[0,1]范围内
im_np = im.numpy().clip(0,1)

# 现在，im_np是一个NumPy数组，其所有值都在[0,1]范围内
print(im_np)
# 输出可能是：
# [[0.   1.   0. ]
#  [1.   0.   0.8]]

# 使用matplotlib显示图像（注意：这里我们假设im_np是灰度图像的一个通道）
plt.imshow(im_np, cmap='gray')
plt.axis('off')
plt.show()
```

然而，上面的例子实际上展示了一个二维数组，而不是一个图像。在图像处理中，图像通常是三维的（对于彩色图像）或二维的（对于灰度图像），并且具有额外的维度来表示高度和宽度（对于二维图像）或高度、宽度和颜色通道（对于三维图像）。但是，`clip` 方法的使用方式在任何情况下都是相同的：它应用于整个NumPy数组，确保所有值都在指定的范围内。

对于彩色图像，你通常会处理一个形状为 `(高度, 宽度, 颜色通道数)` 的三维NumPy数组，并且 `clip` 方法会独立地应用于数组中的每个元素。

这样有了上面的知识点穿插，然后我们看完整的方法代码，现在进行调用和输出：
~~~python
# Lets try these functions
im = imread('scene.jpg')
print(im)
imshow(im)

# Also check the type
print('Type:', im.dtype)
~~~
输出结果：
~~~
tensor([[[0.0667, 0.0902, 0.1216,  ..., 0.2510, 0.3176, 0.3294],
         [0.0824, 0.0824, 0.0941,  ..., 0.2157, 0.2471, 0.2510],
         [0.0980, 0.0784, 0.0745,  ..., 0.1490, 0.1961, 0.1961],
         ...,
         [0.2627, 0.2510, 0.2353,  ..., 0.2902, 0.2510, 0.2549],
         [0.2784, 0.2706, 0.2471,  ..., 0.2667, 0.2745, 0.2745],
         [0.2745, 0.2706, 0.2471,  ..., 0.2471, 0.2784, 0.2824]],

        [[0.2039, 0.2275, 0.2588,  ..., 0.3765, 0.4392, 0.4510],
         [0.2196, 0.2196, 0.2314,  ..., 0.3412, 0.3686, 0.3725],
         [0.2353, 0.2157, 0.2118,  ..., 0.2745, 0.3176, 0.3176],
         ...,
         [0.3569, 0.3451, 0.3294,  ..., 0.3412, 0.2902, 0.2941],
         [0.3725, 0.3647, 0.3412,  ..., 0.3176, 0.3137, 0.3137],
         [0.3686, 0.3647, 0.3412,  ..., 0.2980, 0.3176, 0.3216]],

        [[0.2118, 0.2353, 0.2745,  ..., 0.4588, 0.5216, 0.5333],
         [0.2275, 0.2353, 0.2471,  ..., 0.4235, 0.4510, 0.4549],
         [0.2510, 0.2314, 0.2275,  ..., 0.3569, 0.4000, 0.4000],
         ...,
         [0.2627, 0.2510, 0.2353,  ..., 0.2314, 0.1922, 0.1961],
         [0.2784, 0.2706, 0.2471,  ..., 0.2078, 0.2157, 0.2157],
         [0.2745, 0.2706, 0.2471,  ..., 0.1882, 0.2196, 0.2235]]])
~~~

下面是对图像的灰度处理：
~~~python
im = imread('scene.jpg')

# Lets start with a grayscale image.
# Color image can easily be converted to grayscale by simply averaging the color channels.

im_gray = im.mean(dim=0)

imshow(im_gray)
~~~
输出结果如下；
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b71aff6813fb49dfa56467f384a7dfbd.png)
下面给出一个经常使用的更加规范一些的代码使用示例：
~~~python
import torch  
from torchvision import transforms  
from PIL import Image  
import matplotlib.pyplot as plt  
  
# 读取图像  
image_pil = Image.open('scene.jpg')  
transform = transforms.ToTensor()  
im = transform(image_pil)  
  
# 转换为灰度图像（注意：这里我们直接使用了torchvision的Grayscale转换）  
transform_gray = transforms.Grayscale(num_output_channels=1)  
im_gray = transform_gray(image_pil)  
im_gray = transform(im_gray)  # 转换为张量  
  
# 显示灰度图像（需要先转换为NumPy数组）  
plt.imshow(im_gray.squeeze().numpy(), cmap='gray')  
plt.axis('off')  
plt.show()
~~~
同样的输出上面的内容。
我们建议你后续阅读本专栏的详细的Pytorch高频代码，这个部分将对你的代码能力提升产生非常重要的影响和作用。

