---
title: '第一视角/第一人称数据集'
date: 2024-10-10
permalink: /posts/2024/10/egocentricdata/
tags:
  - Egocentric Dataset
---

数据集对于算法模型的重要性不需要更多的赘述。我整理了如下的第一视角数据集，并提供了具体的对应主页位置以及部分下载链接位置，方便读者浏览查阅。

# 目录

- [CMU-MMAC](#cmu-mmac)
- [EgoAction](#egoactionuec-dataset-choreographed-videos)
- [EgoBody](#egobody)
- [EgoProcel](#egoprocel)
- [UnrealEgo](#unrealego)
- [EgoPW](#egopw)
- [FIRST PERSON SOCIAL INTERACTIONS DATASET](#first-person-social-interactions-dataset-cvpr2012)
- [Multimodal Focused Interaction Dataset](#multimodal-focused-interaction-dataset)
- [TEgo](#tego-teachable-egocentric-objects-dataset)
- [First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations](#first-person-hand-action-benchmark-with-rgb-d-videos-and-3d-hand-pose-annotationscvpr2018)
- [EgoDexter](#egodexter-iccv2017)
- [DR(EyE)VE](#dreyeve)
- [EgoVMBA](#egovmba)
- [DoMSEV](#domsev-–-dataset-of-multimodal-semantic-egocentric-videocvpr2018)
- [XR-EgoPose](#xr-egopose)
- [EYTH](#eythegoyoutubehands)
- [EgoHos](#egohosegocentric-hand-objectsegmentationeccv2022)
- [EgoPAT3D](#egopat3d)
- [EgoHands](#egohandsiccv2015)
- [EgoGesture](#egogesture)
- [EgoPL](#egopl-recognizing-personal-locations-from-egocentric-videoseccv2016)
- [EgoCart](#egocart-a-benchmark-dataset-for-large-scale-indoor-image-based-localization-in-retail-store)
- [EgoVLAD](#egovlad-egocentric-visitor-localization-and-artwork-detection-in-cultural-sites-using-synthetic-data)
- [EgoUNICT-VEDI](#egounict-vedi-egocentric-point-of-interest-recognition-in-cultural-sites)
- [EgoSum](#egosum-discovering-important-people-and-objects-for-egocentric-video-summarization使用到的是ut-ego)
- [UT-Ego](#ut-ego)
- [FPPA](#fppaiccv2015)
- [JPL](#jpl-第一人称交互数据集cvpr2013)
- [HUJI](#huji-egoseg-dataset)
- [ADL](#adlcvpr2012)
- [EgoActivity](#georgia-tech-egocentric-activity-datasetseccv2018)
- [Charades-Ego](#charades-ego-eccv2016)
- [EgoShots](#egoshots)
- [Ego-CH](#ego-ch)
- [EPIC-Kitchens-55](#epic-kitchens-55eccv2018)
- [EPIC-Tent](#epic-tent)
- [EPIC-Kitchens-100](#epic-kitchens-100-cvpr2021)
- [MECCANO](#meccano-a-multimodal-egocentric-dataset-for-humans-behavior-understanding-in-the-industrial-like-domain)
- [Trek-100](#trek-100)
- [EgoCom](#egocom多人多模式以自我为中心的通信数据集)
- [Ego4D](#ego4d)
- [EgoCap](#egocap)
- [EgoClip](#egoclip)
- [EgoTracks](EgoTracks)
- [IT3DEgo](#it3dego)
- [EgoExo4D](#ego-exo4d)

<img width="200%" src="./hr.gif" />

## CMU-MMAC

👉[主页地址](http://kitchen.cs.cmu.edu/index.php)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/be1916942e83bfead876c231fc5fe357.png#pic_center)

CMU 多模式活动数据库 (CMU-MMAC) 数据库包含执行烹饪和食物准备任务的受试者的人类活动的多模式测量。CMU-MMAC 数据库收集于卡内基梅隆大学的动作捕捉实验室。建造了一个厨房，迄今为止，已记录了 25 名受试者烹饪五种不同食谱的情况：布朗尼蛋糕、披萨、三明治、沙拉和炒鸡蛋。

记录了以下方式：

视频：
(1) 三个低时间分辨率 (30 赫兹) 的高空间分辨率 (1024 x 768) 彩色摄像机。
(2) 两台高时间分辨率 (60 赫兹) 的低空间分辨率 (640 x 480) 彩色摄像机。
(3) 一台可穿戴的高空间分辨率 (800 x 600 / 1024 x 768) 低时间分辨率 (30 赫兹) 相机。
音频：
(1) 五个平衡麦克风。
动作捕捉：
(1) Vicon 动作捕捉系统，配有 12 个红外 MX-40 摄像机。每个摄像头以 4 兆像素分辨率、120 赫兹记录图像。
内部测量单元 (IMU)：
(1) 有线 IMU (3DMGX)。
(2) 蓝牙 IMU (6DOF)。
可穿戴设备：
（1）BodyMedia。
(2)电子手表。
有3个数据集：

- 主要数据集：43 名受试者烹饪 5 种食谱。[查看](http://kitchen.cs.cmu.edu/main.php)
- 异常数据集：三个受试者烹饪五种食谱。会发生一些异常情况（火灾和烟雾、盘子掉落、干扰......）。[查看](http://kitchen.cs.cmu.edu/anomalous.php)
- 试点数据集：第一个记录系统。 [查看](http://kitchen.cs.cmu.edu/pilot.php)

## EgoAction/UEC Dataset (Choreographed Videos)

👉[主页地址](http://www.cs.cmu.edu/~kkitani/datasets/)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/629ed484661002d866aab8847eb4f4c9.png#pic_center)

Quad sequence
[QUAD.MP4.zip (254 MB)](http://www.cs.cmu.edu/~kkitani/egoaction/video/QUAD.MP4.zip)

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/a42d972dc452a6f9d4855a4d5116b999.png#pic_center)

Park sequence
[PARK.MP4.zip (1.56 GB)](http://www.cs.cmu.edu/~kkitani/egoaction/video/PARK.MP4.zip)

## EgoBody

👉[主页地址](https://sanweiliti.github.io/egobody/egobody.html)👈

EgoBody 是一个新颖的大型数据集，用于复杂 3D 场景中的社交互动。

## EgoProcel

👉[主页地址](https://sid2697.github.io/egoprocel/)👈

EgoProceL是用于过程学习的大规模数据集。它由130名受试者录制的62小时以自我为中心的视频组成，这些视频执行16项程序学习任务。EgoProceL包含来自CMU-MMAC，EGTEA Gaze的多个任务的视频和关键步骤注释，以及玩具自行车组装，帐篷组装，PC组装和PC拆卸等单个任务。

## UnrealEgo

👉[主页地址](https://4dqv.mpi-inf.mpg.de/UnrealEgo/)👈

我们提出了UnrealEgo，即一个新的大规模自然主义数据集，用于以自我为中心的3D人类姿势估计。UnrealEgo基于配备两个鱼眼摄像头的高级眼镜概念，可在不受限制的环境中使用。我们设计了它们的虚拟原型，并将它们附加到3D人体模型上以进行立体视图捕获。接下来，我们将生成大量的人类运动。因此，UnrealEgo是第一个提供现有以自我为中心的数据集中的运动种类最多的野外立体图像的数据集。此外，我们提出了一种新的基准方法，该方法具有简单而有效的思想，即为立体声输入设计2D关键点估计模块以改善3D人体姿势估计。广泛的实验表明，我们的方法在定性和定量上都优于以前的最新方法。

## EgoPW

👉[主页地址](https://people.mpi-inf.mpg.de/~jianwang/projects/egopw/)👈

最近，使用单个鱼眼相机进行以自我为中心的3D人体姿势估计引起了很多关注。但是，现有方法难以从野外图像中进行姿势估计，因为由于无法获得大规模的野外以自我为中心的数据集，因此只能在合成数据上进行训练。此外，当身体部位被周围场景遮挡或与周围场景交互时，这些方法很容易失败。为了解决野外数据的短缺，我们收集了一个大规模的野外以自我为中心的数据集，称为野外以自我为中心的姿势 (egoopw)。该数据集由头戴式鱼眼摄像机和辅助外部摄像机捕获，该摄像机在训练过程中从第三人称视角提供了对人体的额外观察。我们提出了一种新的以自我为中心的姿势估计方法，可以在外部监督较弱的新数据集上进行训练。具体来说，我们首先通过结合外部视图监督，使用时空优化方法为EgoPW数据集生成伪标签。然后，伪标签用于训练以自我为中心的姿势估计网络。为了促进网络训练，我们提出了一种新颖的学习策略，以通过预先训练的外部视图姿势估计模型提取的高质量特征来监督以自我为中心的特征。实验表明，我们的方法可以从单个野外自我中心图像中预测准确的3D姿势，并且在定量和定性上都优于最新方法。

## First-Person Social Interactions Dataset 【CVPR2012】

👉[主页地址](http://ai.stanford.edu/~alireza/Disney/)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/5ab10f60bfa6ee86730ddbe8fa480ccd.png#pic_center)


该数据集包含 8 个受试者在佛罗里达州奥兰多迪士尼世界度假区度过一天的全天视频。相机安装在拍摄对象戴的帽子上。

注释场景中活跃参与者的数量以及活动类型（步行、等待、聚会、坐着、买东西、吃饭等）:Alin_Day1, Alireza_Day1, Alireza_Day2, Alireza_Day3, Michael_Day2, Munehiko_Day1.

社交互动注释:Alin_Day1, Alireza_Day1, Alireza_Day2, Alireza_Day3, Denis_Day1, Munehiko_Day1, Michael_Day2, Hussein_Day1,

## Multimodal Focused Interaction Dataset

👉[使用说明](https://cvip.computing.dundee.ac.uk/wp-content/uploads/2020/datasets/focusedinteraction/Readme_fidataset.pdf)/[主页地址](https://cvip.computing.dundee.ac.uk/datasets/focusedinteraction/)👈

该数据集用于自动检测此类数据流中的社交互动，特别是集中互动，其中共同存在的个体具有共同的注意力焦点，通过建立面对面的参与和直接对话进行互动。从第一人称视角捕获的现有社交互动公共数据集往往在持续时间、出现的人数、记录的连续性和可变性方面受到限制。

聚焦交互数据集，其中包括使用肩扛式 GoPro Hero 4 相机采集的视频、惯性传感器数据和 GPS 数据以及语音活动检测器的输出。该数据集包含 19 个会话期间捕获的 377 分钟（包括 566,000 个视频帧）的连续多模式记录，其中 17 个会话伙伴在 18 个不同的室内/室外位置。这些会话包括相机佩戴者进行专注交互、不专注交互和不交互的时期。在数据集的整个持续时间内，为所有聚焦和非聚焦交互提供注释。还提供了参与重点互动的 13 名人员的匿名 ID。

数据集包括：

- RGB 1080p 视频（无音频）数据，25Hz
- 语音活动检测音频特征，25Hz
- 惯性传感器（加速度计、陀螺仪、磁力计）和以 2Hz 捕获的 GPS 数据
- ELAN 格式的重点交互注释（ELAN是一种对视频和音频数据的标识进行创建、编辑、可视化和搜索的标注工具，旨在为标识提供声音技术以及对多媒体剪辑进行开发利用。它可以用于语音识别、行为识别等领域。ELAN文件是一种XML格式的文件，其中包含了音频和视频中的数据。）
- csv 文件中每个视频帧的集中交互注释
- 6-fold 交叉验证数据

数据集（视频、惯性传感器、 语音活动 检测功能）和真实注释：  GOPR0177、  GOPR0184b、  GOPR0185、  GOPR0188、
GOPR0193 、 GOPR0194、  GOPR0196、  GOPR0198、  GOPR0201b、  GOPR0202b、  GOPR020 3  、 GOPR0204 、 GOPR0207、GOPR0208、  GOPR0209、  GOPR4037 ,  GOPR4038 ,  GOPR4041 ,  GOPR4042

## TEgo: Teachable Egocentric Objects Dataset

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/350cd652835fefce620f326ebc66f873.png#pic_center)


👉[主页地址](https://iamlabumd.github.io/tego/)/[下载地址](https://drive.google.com/file/d/1VpHLqn7QePgW8h-Ycgta4-1cpy6M_sv2/view)👈

该数据集包括两个人拍摄的 19 个不同物体的第一视角图像，用于训练和测试可教学的物体识别器。具体来说，盲人和视力正常的人使用智能手机摄像头拍摄物体的照片，以训练和测试他们自己的可教学物体识别器。

数据集

训练：为每个物体拍摄大约 30 张连续照片，以训练可训练的物体识别器。
测试：对每个物体一次拍摄一张照片（总共 5 张），以测试其模型的识别准确性。每次都会随机分配一个对象。
注释过程
使用手遮罩、对象中心热图和对象标签对图像进行手动注释。

请注意，只有训练集包括手部掩码和对象中心注释数据。每个环境文件夹中都有多个文件夹，每个文件夹都包含在不同条件下拍摄的图像。

Training:

- Images (original images)
  - vanilla (in a plain environment)
  - wild (in a cluttered environment)
- Masks (hand mask images)
  - vanilla
  - wild
- Objects (object center annotation images)
  - vanilla
  - wild

Testing

- Images (original images)
  - vanilla (in a plain environment)
  - wild (in a cluttered environment)
- README.txt (readme text)
- labels_for_training.json (object-label mapping for the training data)
- labels_for_testing.json (object-label mapping for the testing data)

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/b9b6f07ea5c1c569aee9fb343fa48120.png#pic_center)


![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/52bd83fca6cdc6be9c1b95513a1f3fb6.png#pic_center)


## First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations【CVPR2018】

👉[主页地址](https://github.com/guiggh/hand_pose_action)/[下载地址](https://docs.google.com/forms/d/e/1FAIpQLScoksYrmthDbJeAV0_ysXJDmfvZmzMMsX0_Uhkb6H6DHYGBtg/viewform)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/de47e50beab30765c4782d421caf6544.png#pic_center)


在这项研究工作中，使用3D手部姿态去识别第一视角动态变化的和3D物体交互的手部动作。这个数据集收集了超过 100K 帧的 RGB-D 视频序列，包含 45 个日常手部动作类别，涉及多种手部配置中的 26 种不同物体。为了获得手部姿势注释，作者团队使用了自己的 mo-cap 系统，该系统通过 6 个磁传感器和逆运动学，自动推断出手部模型 21 个关节中每个关节的 3D 位置。 此外，还记录了 6D 物体姿势，并提供了手与物体交互序列子集的 3D 物体模型。据我们所知，这是首个利用三维手部姿势研究第一人称手部动作的基准。我们通过 18 种基准/最先进的方法对基于 RGB-D 和姿势的动作识别进行了广泛的实验评估。测量了使用外观特征、姿势及其组合的影响，并评估了不同的训练/测试协议。最后，还评估了当手部在自我中心视图中被物体严重遮挡时，三维手部姿势估计领域的准备程度及其对动作识别的影响。从结果来看，与其他数据模式相比，使用手部姿势作为动作识别的线索具有明显的优势。我们的数据集和实验对三维手部姿态估计、6D 物体姿态、机器人学以及动作识别等领域具有重要意义。

### 数据集的结构如下

文件Video_files/Subject_1/put_salt/1/color/color_0015.jpeg 的含义是主题号1的动作类“放盐”第一次重复的颜色帧号15图像信息。

文件Video_files/Subject_1/put_salt/1/depth/depth_0015.png C 含义是主题号1的动作类“放盐”的第一次重复的深度流的帧号15的图像信息。

文件Hand_pose_annotation_v1_1/Subject_1/put_salt/1/skeleton.txt 含义是主题编号1的动作类“放盐”的第一次重复的包含序列的手部姿势（在世界坐标中）位置信息。

文件Object_6D_pose_annotation_v1/Subject_1/put_salt/1/object_pose.txt 含义是：主体编号 1 的动作类“放盐”的第一次重复的包含序列的 6D 对象姿势。

### 图像数据详细信息

摄像头：英特尔 SR300。
颜色数据：1920x1080 32位，jpeg格式。
深度数据：640x480 16bit，png格式。

下面是具体的数据集中的内容介绍：

### 手部姿态数据

skeleton.txt中每行的格式: ```t x_1 y_1 z_1 x_2 y_2 z_2 ... x_21 y_21 z_21```

其中的t是帧的序号，而 x_i y_i z_i 是帧t
中的关节i的世界坐标，精确到mm.

手部关节的定义方式如下：[Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP, TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP], 其中的 ’T’, ’I’, ’M’, ’R’, ’P’ 分别表示 ’Thumb拇指’, ’Index食指’, ’Middle中指’, ’Ring无名指’, ’Pinky小指’.

如下是位置示意图：

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/f43eda08d17668c1f6c3aa0d2455802b.png#pic_center)


### 物体姿态数据

可用对象："juice carton"、"milk bottle"、"salt "和 "liquid soap"。例如object_pose.txt 每行的格式都遵从下面的形式：```t M11 M21 M31 M41 M12 ... Mij... M44```,其中的Mij是i行j列的转移矩阵。

### 物体模型

可用对象："果汁纸盒"、"牛奶瓶"、"盐 "和 "液体肥皂"。

格式为 .PLY。每个对象都附带一个纹理文件 texture.jpg。坐标单位为米（而手部姿势的坐标单位为毫米）。

### 相机参数

#### 深度传感器 (内参)

- 图像中心点:

  - u0 = 315.944855;
  - v0 = 245.287079;

- 焦距:

  - fx = 475.065948;
  - fy = 475.065857;

#### RGB传感器 (内参)

- 图像中心点:

  - u0 = 935.732544;
  - v0 = 540.681030;

- 焦距:

  - fx = 1395.749023;
  - fy = 1395.749268;

#### 外参r和t

R = [0.999988496304, -000468848412856，0.000982563360594;
0.00469115935266, 0.999985218048, -0.00273845880292;
-0.000969709653873, 0.00274303671904, 0.99999576807;
0,0,0]【4x3列矩阵】

t = [25.7; 1.22; 3.902; 1];【4x1列矩阵】

遵从：

$\left[\begin{array}{l}x \\ y \\ 1\end{array}\right]=\left[\begin{array}{llll}f & 0 & 0 & 0 \\ 0 & f & 0 & 0 \\ 0 & 0 & 1 & 0\end{array}\right]\left[\begin{array}{c}X_c \\ Y_c \\ Z_c \\ 1\end{array}\right] / Z_c$

> 关于相机的内参外参内容，可以访问[这里](https://blog.csdn.net/weixin_39188311/article/details/132555310?spm=1001.2014.3001.5502)进行查漏补缺。

## EgoDexter 【ICCV2017】

👉[主页地址](https://handtracker.mpi-inf.mpg.de/projects/OccludedHands/EgoDexter.htm)/\[下载地址\] [Zip](https://handtracker.mpi-inf.mpg.de/data/EgoDexter/): 1.99 GB, SHA-256:
7b7f1b357e9e1ee39b8ac11dcd870128cae69ae2fd15d0ae4aa7000f1eed164c👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/d8f19e53680b0963429a041d193ab22a.png#pic_center)


EgoDexter 是一个RGB深度数据集，用来评测遮挡和杂乱场景下的手部跟踪算法，4个序列并由4位人员（2位女性）与各种物体和杂乱背景的不同互动。在 3190 个帧中，有 1485 个帧的指尖位置是人工标注的。

数据集构成：

- 彩色RGB图像: Intel RealSense SR300 @640x480 px
- 深度图像: Intel RealSense SR300 @640x480 px
- 深度色彩图像: 使用英特尔 RealSense SDK 根据深度和彩色图像构建
- GT真值: 在深度数据上手动标注 3D 指尖位置
- 相机校准: 用于世界坐标系和摄像机坐标系（深度、颜色）之间的映射

评判指标
由于常用于手部跟踪研究，我们计算每帧 3D 的平均欧氏误差。每个序列的平均误差以及低于误差阈值的帧百分比可在论文中找到。

## DR(eye)VE

第一视角驾驶数据集

由74个五分钟时长的视频序列组成，注释了超过500，000帧，标签包含驾驶员的注视固定点及其时间整合，可提供特定任务的显著性映射关系。地理参照位置、行驶速度和路线完善了发布的数据集。

👉[主页地址](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/398b3ce8ee6d1a4dff3f8a4248ca6e76.png#pic_center)


![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/af4585de7411d389b774e159f70ba12f.png#pic_center)


## EgoVMBA

👉[主页地址](https://iplab.dmi.unict.it/vmba/)/[论文地址](https://link.springer.com/chapter/10.1007/978-3-319-46604-0_37)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/99940d0e4ab3fafb8c992cb8cbcb6c4a.png#pic_center)


视觉市场购物篮分析（VMBA）。所提出的应用领域的主要目标是通过安装在购物车（我们称之为叙事车）上的摄像头采集的视频来了解零售中的客户行为。为了正确研究该问题并设置第一个 VMBA 挑战，我们引入了 VMBA15 数据集。该数据集由 15 个不同的以自我为中心的视频组成，这些视频是用户在零售店购物时通过叙事推车获取的。每个视频的帧都通过考虑购物车的 8 种可能行为来标记。考虑的购物车行为反映了顾客在零售店购物从开始（挑选购物车）到结束（释放购物车）的行为。与零售内购物车停止时间或收银台商店相关的推断信息可以与经典的市场购物篮分析信息（即收据）相结合，以帮助零售商更好地管理空间。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/ba73e13a688e84eb7e4b0586139591ca.png#pic_center)


![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/d9d128d18a39ba4622e77c0094149aaa.png#pic_center)


## DoMSEV – Dataset of Multimodal Semantic Egocentric Video【CVPR2018】

👉[主页地址](https://www.verlab.dcc.ufmg.br/semantic-hyperlapse/cvpr2018-dataset/)/[下载地址](https://www.verlab.dcc.ufmg.br/semantic-hyperlapse/cvpr2018-dataset/)/[Github](https://github.com/verlab/SemanticFastForward_CVPR_2018)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/aa1d0cff4c4b26bc43c64f7f0ab8cc15.png#pic_center)


80小时的多模态分割第一视角视频数据集，包含大量的行为动作。视频是使用 GoPro Hero 摄像机或由 3D 惯性运动单元（IMU）和英特尔 Realsense R200 RGB-D 摄像机组成的内置装置录制的。不同的人在不同的光照和天气条件下录制视频。录制者会对视频进行标注，告知某些片段的拍摄场景（如室内、城市、拥挤环境或自然环境）、所进行的活动（行走、跑步、站立、浏览、驾驶、骑自行车、吃饭、做饭、吃东西、观察、交谈、玩耍或购物）、是否有东西吸引了他们的注意力以及他们何时与某些物体进行了互动。此外，我们还为每个记录者创建了一个档案，代表了他们对一系列物体和视觉概念的偏好。

视频影响信息

详细的视频信息，例如持续时间、分辨率、捕捉设备、FOV、FPS、摄像机安装和传感器（GPS、IMU、深度）。详情表格情况查阅主页。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/2749a4e357a9fde8b60505ce084670a5.png#pic_center)


有趣的地方在于作者不仅提供了基于外置设备构建的数据集还提供了自己搭建的外置模型设备的安装方法

Realsense R200 RGB-D 相机外壳的 3D 模型，支持 LORD MicroStrain 3DM-GX3-25 和 GoPro 安装适配器。
[【STL构建地址】](https://www.verlab.dcc.ufmg.br/semantic-hyperlapse/data/multimodal/intel_r200_case.stl)

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/83f1d6cfc4df73f86f104e1b82a4b71a.png#pic_center)


DoMSEV – 多模态语义自我中心视频数据集

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/004d736f4a2fb7da60b0b8d75a86e5ce.png#pic_center)


## XR-EgoPose

👉[主页地址](https://github.com/facebookresearch/xR-EgoPose)👈

xR-EgoPose 是一个以自我为中心的合成数据集，用于以自我为中心的 3D 人体姿态估计。它由各种室内和室外空间中的约 38 万张照片般逼真的以自我为中心的相机图像组成。

## EYTH(EgoYouTubeHands)

👉[主页地址](https://github.com/aurooj/Hand-Segmentation-in-the-Wild)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/61e3a798eb99d09ff3c52bd41ec0849c.png#pic_center)


EYTH 是一个以自我为中心的手部分割数据集，由 1290 个带注释的帧组成，这些帧来自在不受约束的现实世界中录制的 YouTube 视频。 视频因环境、参与者数量和动作而异。 该数据集有助于研究无约束环境中的手部分割问题。

## EgoHOS (Egocentric Hand-Objectsegmentation)【ECCV2022】

👉[主页地址](https://github.com/owenzlz/EgoHOS)👈

EgoHOS（以自我为中心的手部对象分割）是一个包含 11,243 张以自我为中心的图像的标记数据集，其中包含在各种日常活动中交互的手和对象的每像素分割标签。 它是最早标记详细手部对象接触边界的数据集

## EgoPAT3D

👉[主页地址](https://ai4ce.github.io/EgoPAT3D/)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/02db824fe37f55c61a4a7726f4cd1718.png#pic_center)


EgoPAT3D数据集 15个家庭场景 15个点云文件 (每个场景一个) 150个总记录 (每个场景中有10个记录，每个记录中有不同的对象配置) 15000手-对象动作 (每次录制100) 约600分钟的RGBD视频 (每个视频约4分钟) ~ 1,080,000 RGB帧30 fps ~ 900,000个手部动作帧 (假设每个手部对象动作约2秒)

## EgoHands【ICCV2015】

👉[主页地址](http://vision.soic.indiana.edu/projects/egohands/)/[下载地址](http://vision.soic.indiana.edu/egohands_files/egohands_all_frames.zip)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/c116161426748d2e4501a692d9582ac1.png#pic_center)


EgoHands 数据集包含 48 个 Google Glass 视频，内容涉及两个人之间复杂的第一人称交互。该数据集的主要目的是提供更好的数据驱动方法来理解第一人称计算机视觉中的手。该数据集提供

- 高质量、像素级的手部分割
能够在语义上区分观察者的手和其他人的手以及左手和右手
- 当参与者自由地参与一系列联合活动时，几乎不受约束的手部姿势
- 包含 15,053 只真实标记手的大量数据

数据集构成：

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/8ea0b6e72b5f230b328202c2bb206cbe.png#pic_center)


### EgoGesture

👉[主页地址](http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/4d8b1b0b0e562ab07fd490fb0e7f2233.png#pic_center)


该数据集包含来自 50 个不同主题的 2,081 个 RGB-D 视频、24,161 个手势样本和 2,953,224 个帧。我们设计了 83 类静态或动态手势，重点关注与可穿戴设备的交互

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/b18db486081a2156ad03ea9089b31761.png#pic_center)


这些视频收集自 6 个不同的室内和室外场景。我们还考虑人们在行走时做出手势的场景。我们设计的6个场景由4个室内场景组成：

- 主体处于静止状态，背景杂乱
- 主体处于静止状态，背景是动态的
- 对象处于静止状态，面向窗户，阳光变化剧烈
主体处于行走状态
和 2 个室外场景：
- 主体处于静止状态，背景是动态的
- 主体处于行走状态，背景动态。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/caa1a386e7a3ff155d7658b0e38498f3.png#pic_center)


在我们的实验中，我们按主题将数据随机分为训练（SubjectID：3,4,5,6,8,10,15,16,17,20,21,22,23,25,26,27,30, 32, 36, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50), 验证 (SubjectID: 1, 7, 12, 13, 24, 29, 33, 34, 35, 37) ）和测试（SubjectID: 2, 9, 11, 14, 18, 19, 28, 31, 41, 47）集的比例为3:1:1，得到1,239个训练视频，411个验证视频和431个测试视频。训练、验证和测试分组中的手势样本数量分别为 14416、4768 和 4977。

我们提供原始 RGB-D 视频（~46G）、大小为 320ⅹ240 像素（~32G）的图像文件以及注释供下载。注释文件中有三列文本，分别表示视频中每个手势样本的类标签、开始和结束帧索引。

数据集按以下格式组织：

        videos/Subject01/Scene1/Color/rgb1.avi
        …
        videos/Subject01/Scene1/Depth/depth1.avi
        …
        images_320-240/Subject01/Scene1/Color/rgb1/000001.jpg …
        …
        images_320-240/Subject01/Scene1/Depth/depth1/000001.jpg …
        …
        labels-revised1/subject01/Scene1/Group1.csv

要获取数据库，请按照以下步骤操作：

下载并打印EgoGesture 使用 [协议文档](http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/agreement.txt)
签署协议
将协议发送至zhangjunkai2021 AT ia.ac.cn
如果您的申请获得批准，您将收到一封电子邮件，其中包含Google Drive和百度网盘的链接和密码。
48小时内下载数据库。

## EgoPL (Recognizing Personal Locations from Egocentric Videos)【ECCV2016】

👉[主页地址](https://iplab.dmi.unict.it/PersonalLocations/)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/132030889d09c90f51ecb00c00783621.png#pic_center)


我们将研究如何从第一视角视频中识别用户日常活动中产生的个人位置。其中包含用户使用 4 种不同的可穿戴摄像头获取的 8 个个人位置的第一视角视频。为了使我们的分析在实际场景中发挥作用，我们提出了一种剔除负位置的方法，即那些不属于最终用户感兴趣的任何类别的位置。

我们构建了一个由单个用户在五个不同个人位置获取的以自我为中心的视频数据集：汽车、咖啡自动售货机、办公室、电视和家庭办公室。该数据集随后扩展到 8 个个人位置：汽车、咖啡自动售货机、办公室、电视、家庭办公室、厨房、水槽和车库。所考虑的个人位置源自用户的日常活动，并且与生活质量评估和日常监测等辅助应用相关。鉴于市场上存在多种可穿戴设备，我们使用四种不同的摄像头来评估一些设备特定因素的影响，例如佩戴方式和视场 (FOV)

训练集由感兴趣的个人位置的短视频（约 10 秒）组成。在获取训练视频期间，用户转动头部（或胸部，如果是安装在胸部的设备），以覆盖最相关的环境视图。每个训练集中包含每个感兴趣位置的单个视频镜头。

测试集包含用户在执行与感兴趣位置相关的正常活动时在所考虑的位置获取的中等长度的视频（5 到 10 分钟）。每个测试集包含每个感兴趣位置的 5 个视频。为了收集可能的负面样本，我们获取了几个不代表任何正在分析的位置的短视频。负面视频包括室内、室外场景、其他桌子和其他自动售货机。负片视频分为两个独立的组：测试负片和“优化”负片。后一组负样本的作用是提供一组独立的数据，可用于优化负拒绝方法的参数。整个数据集的视频长度超过 20 个小时，总共超过 100 万帧。

为了确保结果的可重复性，我们提供数据集的原始（4 个位置）和扩展（8 个位置）版本以及详细信息。数据集包含从原始视频采样的帧的集合。如需访问完整视频，请发送电子邮件至[此地址](furnari@dmi.unict.it)。

### 10-LOCATIONS DATASET [EPIC 2016]

该数据集与提交给 IEEE Transactions on Human-Machine Systems 的 论文“Temporal Segmentation of Egocentric Videos tohighlight Personal Locations of Interest”相关。

假设用户只需要提供最少的数据来定义他个人感兴趣的位置，训练集包含 10 个短视频（每个位置一个），每个视频的平均长度为 10 秒。验证集提供了一组独立的图像，可用于检查所考虑方法的泛化能力。该测试集由 10 个视频序列组成，涵盖所考虑的个人感兴趣位置、负帧和位置之间的过渡。测试序列中的每一帧均已手动标记为 10 个感兴趣位置之一或阴性位置。验证集包含 10 个中等长度（大约 5 到 10 分钟）的在所考虑地点进行的活动视频（每个地点一个视频）。验证视频已进行时间二次采样，以便在每个位置准确提取 200 个验证帧，同时考虑训练和测试视频中的所有帧。我们还获取了 10 个包含负样本的中等长度视频，我们从中统一提取 300 帧用于训练（以便与使用负样本进行学习或优化的方法进行比较）和 200 帧用于验证。建议的数据集包含 2142 个正向帧，加上 300 个用于训练的负向帧，2000 个正向帧，

这总共提取了 133770 个帧用于实验目的。数据集可以从以下链接下载：

[10个地点下载地址](https://iplab.dmi.unict.it/PersonalLocations/segmentation/10contexts_dataset.zip)

### 8-LOCATIONS DATASET [THMS 2016]

该数据集与 IEEE Transactions on Human-Machine Systems 论文“Recognizing of Personal Locations from Egocentric Videos”相关。

在训练时，使用 10 秒视频镜头中包含的所有帧，而测试视频则进行时间二次采样。为了减少测试集中每个位置要处理的帧数量（同时仍保留一些时间一致性），我们提取 15 个连续帧的 200 个子序列。子序列的起始帧是从每个类别可用的 5 个视频中均匀采样的。对阴性测试应用相同的子采样策略。我们还从优化负视频中提取 300 帧。

这总共提取了 133770 个帧用于实验目的。数据集可以从以下链接下载：

- [8个地点下载【RJ】](https://iplab.dmi.unict.it/PersonalLocations/RJ.zip)
- [8个地点下载【LX2P】](https://iplab.dmi.unict.it/PersonalLocations/LX2P.zip)
- [8个地点下载【LX2W】](https://iplab.dmi.unict.it/PersonalLocations/LX2W.zip)
- [8个地点下载【LX3】](https://iplab.dmi.unict.it/PersonalLocations/LX3.zip)

### 5-CONTEXTS DATASET [ACVR 2015]

该数据集与 ICCV 2015 联合举办的第三届辅助计算机视觉和机器人研讨会上发表的论文“从自我中心图像中识别个人背景”相关。
在训练时，使用 10 秒视频镜头中包含的所有帧，而在测试时，仅使用从测试视频中提取的每类约 1000 帧。此处提供的数据集包含提取的帧，以确保实验的可重复性。每个特定于设备的数据集包含约 6000 个用于测试的帧（每个类别 1000 个，加上 1000 个负样本）和约 1000 个用于训练的帧（平均每个位置 200 个帧）。该数据集可以从以下链接下载：

[下载地址](https://iplab.dmi.unict.it/PersonalLocations/personal_contexts_dataset.zip)

## EgoCart: a Benchmark Dataset for Large-Scale Indoor Image-Based Localization in Retail Store

👉[主页地址](https://iplab.dmi.unict.it/EgocentricShoppingCartLocalization/)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/515c496d4656e589dcc218de653e6ffc.png#pic_center)


视频展示了基于使用深度度量学习（三重网络）获得的图像表征的 1-NN 方法的性能。右侧显示的是查询图像（上图）和 1-NN 选出的最接近的训练图像（下图）。左侧是训练样本的所有二维位置（黑点）和测试视频的地面真实位置（红线）。在每个时间步长，一个圆圈表示推断出的位置。圆圈的颜色表示算法产生的位置误差（参考右侧的颜色条）。我们还报告了将推断位置与地面实况位置连接起来的片段。

根据第一视角图像定位零售店中的购物车。完成这项任务可以推断出顾客的行为信息，从而了解他们在店内的活动方式以及他们更关注的内容。为了研究这个问题，我们提出了一个在真实零售店中收集的大型图像数据集。该数据集包含 19,531 张 RGB 图像以及深度图、地面实况相机姿势和类别标签，这些标签指定了每张图像在商店中的采集区域。我们向公众发布该数据集是为了鼓励基于图像的大规模室内定位研究，并解决缺乏大型数据集来解决这一问题的难题。因此，我们在提出的数据集上利用图像和深度信息对几种基于图像的定位技术进行了基准测试。

数据集是从一家面积为 782 平方米的零售店内采集的九个不同视频中提取的帧建立的。这些视频由安装在购物车上的两个不同的 zed 摄像机采集，其焦轴与商店地板平行。每段视频都以 3 fps 的速度进行了时间子采样。整个数据集由 19,531 对 RGB 图像和深度图像组成。每对图像都标有相关的 3DOF 姿态和商店的相关区域（类标签）。数据集分为训练集和测试集。两个子集分别由 6 个训练视频（13,360 帧）和 3 个测试视频（6,171 帧）组成。每个子集都包含覆盖整个商店的图像。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/c8a21720c1aed20f4e3233d18465593e.png#pic_center)


![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/a7d422c5cf956cb2b0e83431fbb86b8f.png#pic_center)


数据集中的一些示例及其位置。在二维地图中，属于不同类别的样本用不同颜色表示。该数据集为大规模室内定位提供了一个有趣的测试平台。事实上，它包含了几个困难的例子： A）和 H）包含不同比例的同一架子，B）和 G）表示同一走廊中方向相反的帧，C）和 F）是位置相同但方向不同的帧，D）和 E）表示位置不同但内容相似的图像，L）和 I）是视觉相似度很高的两个不同走廊的图像。

## EgoVLAD (Egocentric Visitor Localization and Artwork Detection in Cultural Sites Using Synthetic Data)

👉[主页地址](https://iplab.dmi.unict.it/SimulatedEgocentricNavigations/)👈

我们提出了一种从文化遗址的 3D 模型开始生成模拟自我中心数据的工具，有助于研究文化遗址中基于图像的定位 (IBL) 和艺术品检测问题。我们特别关注佩戴以自我为中心的相机的用户的本地化，这有助于增强文化场所游客的体验。这项工作的贡献是：

创建自动标记相机 6 自由度姿态 (6DoF) 的合成数据集的通用方法，用于研究 IBL 问题和艺术品检测；
考虑S3DIS 数据集（区域 3）中的 3D 模型而收集的第一个大型自我中心导航数据集；
考虑文化遗址 Galleria Regionale Palazzo Bellomo 1的 3D 模型收集的第二个以自我为中心的导航数据集；
3DoF 室内定位的基准研究，以评估我们的工具的有效性，考虑基于三元组网络和图像检索程序的度量学习的 IBL 管道；
对第二个数据集的 16 件艺术品进行艺术品检测的基准研究。

## EgoUNICT-VEDI (Egocentric Point of Interest Recognition in Cultural Sites)

👉[主页地址](https://iplab.dmi.unict.it/VEDI_POIs/)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/1c56b499851b6a8ab0bdaaceb8af72e3.png#pic_center)


我们对UNICT-VEDI 数据集进行了扩展，用边界框标注了 57 个不同的兴趣点。我们只考虑使用头戴式微软 HoloLens 设备获取的数据。UNICT-VEDI 数据集由一组训练视频（每个兴趣点至少一个）和 7 个测试视频组成，测试视频由参观文化遗址的受试者获取。数据集中的每段视频都进行了时间标注，以显示游客所处的环境（标注了 9 个不同的环境）和游客观察到的兴趣点（标注了 57 个兴趣点）。对于 UNICT-VEDI 数据集中的 57 个兴趣点，我们从提供的训练视频中为每个兴趣点标注了约 1,000 个帧，共计 54,248 个帧。

## EgoSum (Discovering Important People and Objects for Egocentric Video Summarization【使用到的是UT Ego】)

👉[主页地址](https://vision.cs.utexas.edu/projects/egocentric/index.html)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/9672989417506e186e280d781296d521.png#pic_center)


使用 Looxcie 可穿戴相机，它以 15 fps、320 x 480 分辨率捕获视频。我们收集了 10 个视频，每个视频长度为 3-5 小时。四名受试者为我们佩戴了相机：一名本科生、两名研究生和一名办公室职员。这些视频记录了各种活动，例如吃饭、购物、听讲座、驾驶和烹饪。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/e8f768aecdaacc50756b38dbff8950cb.png#pic_center)


为了学习有意义的以自我为中心的属性而不过度适应任何特定类别，我们使用 Amazon Mechanical Turk (MTurk) 众包大量注释。对于以自我为中心的视频，必须在摄像机佩戴者的活动背景下看到该对象，才能正确衡量其重要性。我们精心设计了两个注释任务来捕捉这一方面。在第一项任务中，我们要求工作人员观看三分钟的加速视频，并用文本描述他们认为创建视频摘要所必需的重要人物或物体。在第二个任务中，我们显示视频中均匀采样的帧及其从第一个任务获得的相应文本描述，并要求工作人员在任何描述的人或物体周围绘制多边形。有关注释示例，请参见上图。

## UT Ego

👉[主页地址](https://vision.cs.utexas.edu/projects/egocentric_data/UT_Egocentric_Dataset.html)/[下载地址](http://vision.cs.utexas.edu/projects/egocentric/download_register.html)/[注释](http://vision.cs.utexas.edu/projects/egocentric_data/egocentric_GT.zip)👈

数据集包含 4 个从头戴式摄像机捕获的视频。每个视频长约 3-5 小时，在自然、不受控制的环境中拍摄。

我们使用了 Looxcie 可穿戴相机，它可以以 15 fps、320 x 480 分辨率拍摄视频。四位受试者为我们佩戴了相机：一名本科生、两名研究生和一名办公室职员。这些视频记录了各种活动，例如吃饭、购物、听讲座、驾驶和烹饪。

- 出于隐私原因，我们只能分享最初拍摄的 10 个视频中的 4 个（每个主题各一个）。它们对应于我们在 CVPR 2012 和 CVPR 2013 论文中评估的测试视频。

## FPPA【ICCV2015】

👉[主页地址](http://tamaraberg.com/prediction/README.txt)👈

【6.8G】

FPPA（第一人称个性化活动）数据集包含 5 名受试者拍摄的 5 种日常生活活动的 591 个以自我为中心的视频。鼓励受试者在不同条件下、在他们通常执行活动的不同地点多次完成每项活动。

日常生活活动：

- 饮用水
- 穿鞋
- 使用冰箱
- 洗手
- 穿上衣服

科目：

- 单一主题：由单个人的视频组成（subject_01、subject_02）
- 家庭主题：由家庭视频组成，即两个或更多人住在同一地点（subject_03、subject_04、subject_05）

元：

- 分辨率：1280x720
- 帧率：30
- 编解码器：H.264

该数据集仅供研究使用。如有任何疑问，请随时发送电子邮件至 <yipin@cs.unc.edu>。

## JPL 第一人称交互数据集【CVPR2013】

👉[主页地址](http://michaelryoo.com/jpl-interaction.html)👈

👉[【连续视频】](https://drive.google.com/file/d/1Q90Pifwyztld5z5604YIdUYpmNyzjSPM/view?usp=sharing)👈

👉[【分段】](https://drive.google.com/file/d/1eivyF3gPbS3ejea-NYebMBzS40xsRrqF/view?usp=sharing)
此处还提供了数据集的分段版本，其中每个视频在时间上分段以包含单个活动。👈

👉[【标签】](http://michaelryoo.com/datasets/jpl_interaction_labels.xlsx)
描述时间间隔的标签发生的活动及其类别 ID。👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/f0622c91b2315da083e0525d729ecea6.gif#pic_center)


JPL 第一人称交互数据集（JPL-Interaction 数据集）由从第一人称视角拍摄的人类活动视频组成。该数据集特别旨在提供交互级别活动的第一人称视频，记录从参与此类物理交互的人/机器人的角度（即视点）观察事物的视觉外观。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/e09f2690be4fa447c881d61162f347f5.png#pic_center)


该第一人称数据集包含人类与观察者之间交互的视频。我们将 GoPro2 相机连接到人形模型的头部，并要求人类参与者通过执行活动与人形机器人互动。为了模拟真实机器人的移动性，我们还在人形机器人下方放置了轮子，并让操作员从后面推动人形机器人来移动人形机器人。

数据集中有 7 种不同类型的活动，包括 4 种与观察者的积极（即友好）互动、1 种中立互动和 2 种消极（即敌对）互动。“与观察者握手”、“拥抱观察者”、“抚摸观察者”、“向观察者挥手”是四种友好互动。中立互动是指两个人就观察者进行对话，同时偶尔指向观察者的情况。“殴打观察者”和“向观察者扔物体”是两种负面相互作用。在人类活动期间连续录制视频，每个视频序列包含 0 到 3 个活动。视频分辨率为 320*240，帧率为 30 fps。

## HUJI EgoSeg Dataset

👉[主页地址](https://www.vision.huji.ac.il/egoseg/videos/dataset.html)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/a78254b3fae36e7a27a57d88bfe3a316.png#pic_center)


该数据集包含从以自我为中心的相机捕获的 122 个视频。前 44 个视频（文件名 Huji_*）是我们使用头戴式 GoPro Hero3+ 拍摄的。下一组是 YouTube 精选的视频，形成 7 个额外的佩戴者活动类别。我们还使用了 第一人称社交互动 [1]和 GTEA Gaze+ [2]数据集的视频。

[此处](https://www.vision.huji.ac.il/egoseg/annotations/annotations.zip)提供了真实注释，其中包含将所有视频时间分割为 14 种不同的相机佩戴者活动。
我们使用[ELAN](http://tla.mpi.nl/tools/tla-tools/elan/download/)进行注释。为了在 Matlab 中处理注释，我们开发了辅助函数来读取和写入 .EAF 文件，请参阅[此处](https://www.vision.huji.ac.il/egoseg/annotations-elan-format.zip)。

## ADL【CVPR2012】

👉[主页以及下载地址](https://redirect.cs.umbc.edu/~hpirsiav/papers/ADLdataset/)👈

我们提出了一个新颖的数据集和新颖的算法，用于解决第一人称相机视图中日常生活活动（ADL）的检测问题。我们收集了包含数十人执行即兴日常活动的一百万帧的数据集。该数据集标注有活动、对象轨迹、手部位置和交互事件。ADL 与典型动作的不同之处在于，它们可能涉及长期时间结构（泡茶可能需要几分钟）和复杂的对象交互（冰箱门打开时看起来不同）。我们开发了新颖的表示形式，包括（1）时间金字塔，它在对模型进行评分时概括了众所周知的空间金字塔以近似时间对应关系；以及（2）复合对象模型，它利用了对象在交互时看起来不同的事实。我们进行了广泛的实证评估，并证明我们的新颖表示比传统方法产生了两倍的改进。我们的分析表明，现实世界的 ADL 识别“都是关于对象的”，特别是“关于与之交互的对象的”。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/a2568bf25e32e15c2faed49be470e192.png#pic_center)


## EGTEA GAZE+ （Georgia Tech Egocentric Activity Datasets【ECCV2018】）

👉[主页地址](https://cbs.ic.gatech.edu/fpv/)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/4605893f4e20178bcdc29f85834fe06e.png#pic_center)


EGTEA Gaze+ 是我们最大、最全面的 FPV 动作和注视数据集。它包含 GTEA Gaze+，并附带高清视频 (1280x960)、音频、视线跟踪数据、帧级动作注释以及采样帧的像素级手部掩模。
具体来说，EGTEA Gaze+ 包含来自 32 个科目的 86 个独特课程的 28 小时（未识别）烹饪活动。这些视频带有音频和视线跟踪 (30Hz)。我们还进一步提供了动作（人与物体交互）和手势的人工注释。

动作注释包括10325 个细粒度动作实例，例如“切甜椒”或“将调味品（从）调味品容器倒入沙拉中”。

手部注释由视频中 13,847 帧的15,176 个手部掩模组成。
![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/a27245582cd10c0b657c675b040cf137.png#pic_center)


## Charades-Ego 【ECCV2016】

👉[主页地址](https://prior.allenai.org/projects/charades)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/c4b1fef48796f77c16aaa9a2ded21027.png#pic_center)


Charades 是由通过 Amazon Mechanical Turk 收集的 9848 个日常室内活动视频组成的数据集。向 267 名不同的用户呈现一个句子，其中包括固定词汇中的对象和动作，然后他们录制了一段视频来表演该句子（就像在猜谜游戏中一样）。该数据集包含 157 个动作类的 66,500 个时间注释、46 个对象类的 41,104 个标签以及视频的 27,847 个文本描述。

每个视频都使用训练集上的 4 名工作人员和测试集上的 8 名工作人员的共识进行了详尽的注释。详情请参阅更新后的随附出版物。有关数据集的问题，请联系<vision.amt@allenai.org> 。

- [README](https://prior.allenai.org/projects/data/charades/README.txt)
- [License](https://prior.allenai.org/projects/data/charades/license.txt)
- [Annotations & Evaluation Code (3 MB)](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip)
- [Caption evaluation code (70 MB)](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/charades-caption.zip)
- [Data (scaled to 480p, 13 GB)](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip)
- [Data (original size) (55 GB)](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip)
- [RGB frames at 24fps (76 GB)](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_rgb.tar)
- [Optical Flow at 24fps (45 GB)](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_flow.tar)
- [Two-Stream features at 8fps (RGB Stream, 12GB compressed, 37GB uncompressed)](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_features_rgb.tar.gz)
- [Two-Stream features at 8fps (Flow Stream, 16GB compressed, 45GB uncompressed)](https://github.com/gsig/charades-algorithms)
- [Baseline Algorithms @ GitHub](https://github.com/gsig/charades-algorithms)
- [Submission Files for Baseline Algorithms @ GitHub](https://github.com/gsig/temporal-fields)
- [Attributes and Visualization Code @ GitHub](https://github.com/gsig/actions-for-actions)
- [Held-out challenge dataset and publicly available evaluation server](http://vuchallenge.org/charades.html)

## EgoShots

👉[主页地址](https://github.com/NataliaDiaz/Egoshots)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/c276db2c48544d8084b16f5446299586.png#pic_center)


Egoshots: 一个为期2个月的自我视觉数据集，带有Autographer可穿戴相机，带有转移学习的 “免费” 注释。使用了三种最新的预先训练的图像字幕模型。 该数据集代表了2名实习生在飞利浦研究 (荷兰) (2015年5月到7月) 慷慨捐赠数据时的生活: 娜塔莉亚·迪亚斯·罗德里格斯 Vana Panagiotou

## EGO-CH

👉[主页地址](https://iplab.dmi.unict.it/EGO-CH/)/[下载地址](https://docs.google.com/forms/d/e/1FAIpQLSfch95hTsTMbA-PjpYTC8PcdFcge2OAXys2EsgwMyThEpuyCQ/viewform)👈

我们提出了 EGO-CH，一个以自我为中心的视频数据集，用于理解访问者的行为。该数据集是在两个不同的文化场所收集的，包括 70 名受试者（包括志愿者和 60 名真实访客）采集的超过 27 小时的视频。整个数据集包括 26 个环境和 200 多个兴趣点 (POI) 的标签。具体来说，EGO-CH 的每个视频都注释有 1) 指定访问者当前位置和观察到的 POI 的时间标签，2) POI 周围的边界框注释。该数据集的很大一部分由 60 个视频组成，还与访问者在每次访问结束时填写的调查相关联。为了鼓励对该主题的研究，我们提出了 4 项具有挑战性的任务，有助于了解访问者的行为并报告数据集的基线结果。该数据集是使用头戴式 Microsoft HoloLens 设备在意大利西西里岛的两个文化遗址获取的：1) 位于锡拉库扎的“Palazzo Bellomo”， 2)位于卡塔尼亚的 “Monastero dei Benedettini” 。

EGO-CH：贝洛莫宫

22 个环境和 191 个兴趣点 (POI)
视频采集：1280x720，29.97 fps
57 个训练视频和 10 个验证/测试视频
191 张与所考虑的 POI 相关的参考图像
时间标签，指示访客所在的环境以及当前观察到的 POI
70088 个带有边界框注释的帧
从边界框注释中提取的 23727 个图像块

EGO-CH：贝内代蒂尼修道院
4 个环境和 35 个兴趣点 (POI)
视频采集：训练/验证视频 1216x684、24 fps || 测试视频 1408x792、30.03 fps
48 个训练视频和 5 个验证视频
60次真实访问
与所考虑的 POI 相关的 35 张参考图像
时间标签，指示访客所在的环境以及当前观察到的 POI
106911 个带有边界框注释的帧
从边界框注释中提取的 45048 个图像补丁
与 60 次真实访问相关的 60 项调查

四大任务：

- 房间定位
- [baseline](https://iplab.dmi.unict.it/VEDI/#code)

任务包括根据以自我为中心的图像确定文化遗址的参观者所在的房间。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/26b8a3e991544e0a386a53653485feed.png#pic_center)


- 兴趣点识别
- [baseline](https://pjreddie.com/darknet/yolo/)

该任务包括识别用户正在查看的兴趣点。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/bdec9b945951e1be65809eeb15048c72.png#pic_center)


- 对象检索
- [baseline](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

给定包含对象的查询图像，任务包括从数据库中检索同一对象的图像。
![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/cc16210fd92275640d4f82f316bffabe.png#pic_center)


- 内容生成
- [baseline](https://iplab.dmi.unict.it/EGO-CH/downloads/Survey_Prediction_code.zip)

该任务包括通过分析相关的自我中心视频来预测调查的内容。
![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/b00bd8f37801e42cf6e0b049d58add0a.png#pic_center)

## EPIC-Kitchens-55【ECCV2018】

👉[主页地址](https://epic-kitchens.github.io/2018)👈

最大的第一视角数据集（当时），多方面的无脚本原始记录环境信息，例如佩戴者的家庭，捕获好几天中发生在厨房中的每日活动，注释通过新颖的“现场”音频评论方式收集得到。

数据集特征

- 32 个厨房 - 4 个城市
- 头戴式摄像头
- 55 小时录制 - 全高清，60fps
- 1150万帧
- 多语言解说
- 39,594 个动作片段
- 454,255 个对象边界框
- 125 个动词类别，331 个名词类别

视频序列以及下载
[【视频序列下载地址】](http://dx.doi.org/10.5523/bris.3h91syskeag572hl6tvuovwv4d)

【分段脚本下载数据集】

- [视频](https://github.com/epic-kitchens/download-scripts/blob/master/download_videos.sh)
- [视频帧](https://github.com/epic-kitchens/download-scripts/blob/master/download_frames_rgb_flow.sh)
- [目标注释图像信息](https://github.com/epic-kitchens/download-scripts/blob/master/download_object_detection_images.sh)

[【注释获取】](https://github.com/epic-kitchens/annotations)

## EPIC-Tent

👉[主页地址](https://data.bristol.ac.uk/data/dataset/2ite3tu1u53n42hjfh3886sa86)/[下载地址](https://data.bris.ac.uk/datasets/tar/2ite3tu1u53n42hjfh3886sa86.zip)👈

## EPIC-Kitchens-100 【CVPR2021】

👉[主页地址](https://epic-kitchens.github.io/2020-100)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/62055a7e099859c0b880ab8ff36c57e2.png#pic_center)


第一人称（自我中心）视觉中扩展的最大数据集 ；在本地环境（即佩戴者的家中）进行多方面、视听、无脚本记录，捕捉多日厨房中的所有日常活动。使用新颖的“暂停并讲话”叙述界面收集注释。

- 45 个厨房 - 4 个城市
- 头戴式摄像头
- 100小时录制 - 全高清
- 20M帧
- 多语言解说
- 90K 个动作片段
- 20K 独特的旁白
- 97 个动词类别，300 个名词类别
- 5 项挑战

数据和下载脚本

【重要】最近在数据集中的两个视频的预提取 RGB 和光流帧中检测到错误。这不会影响视频本身或此 github 中的任何注释。但是，如果您一直在使用预先提取的帧，则可以按照[此链接](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/README.md#erratum)中的说明最终修复错误。

扩展序列（+RGB 帧、流帧、陀螺仪 + 加速计数据）：可在[Data.Bris 服务器（740GB 压缩包）](http://dx.doi.org/10.5523/bris.2g1n6qdydwa9u22shpxqzp0t8m)或[学术 Torrent](https://academictorrents.com/details/cc2d9afabcbbe33686d2ecd9844b534e3a899f4b) 中 获取

原始序列（+RGB 和流帧）：可在[Data.Bris 服务器（1.1TB 压缩）](http://dx.doi.org/10.5523/bris.3h91syskeag572hl6tvuovwv4d)或通过[学术种子](https://academictorrents.com/details/d08f4591d1865bbe3436d1eb25ed55aae8b8f043)下载

自动注释（掩码、手和物体）：可在[Data.Bris 服务器 (10 GB)](https://data.bris.ac.uk/data/dataset/3l8eci2oqgst92n14w2yqi5ytu)上下载。我们还有两个存储库，可让您可视化并利用这些[手动对象](https://github.com/epic-kitchens/epic-kitchens-100-object-masks)和[蒙版](https://github.com/epic-kitchens/epic-kitchens-100-object-masks)的自动注释。

我们还提供了一个[python 脚本](https://github.com/epic-kitchens/epic-kitchens-download-scripts)来下载数据集的各个部分

注释和pipeline
所有挑战的所有注释（训练/验证/测试）均可在[EPIC-KITCHENS-100-annotations 存储库](https://github.com/epic-kitchens/epic-kitchens-100-annotations)中找到

用于可视化和利用自动注释的代码可用于[对象掩模](https://github.com/epic-kitchens/epic-kitchens-100-object-masks)和[手部对象检测](https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes)。

[用于收集 EPIC-KITCHENS-100 旁白的 EPIC Narrator 在EPIC-Narrator 存储库](https://github.com/epic-kitchens/epic-kitchens-100-narrator)中开源

关于具体的任务挑战：

动作识别
任务：为修剪的片段分配（动词、名词）标签。
训练输入（强监督）：一组经过修剪的动作片段，每个片段都用（动词、名词）标签进行注释。
训练输入（弱监督）：一组未修剪的视频，每个视频都带有（时间戳、动词、名词）标签列表。请注意，对于每个操作，您都会获得一个大致对齐的时间戳，即位于该操作周围的一个时间戳。时间戳可以位于背景帧或属于另一个动作的帧之上。
测试输入：一组经过修剪的未标记动作片段。
分割：训练和训练验证，根据测试分组进行评估。
评估指标。针对所有片段以及未见过的参与者和尾部类别计算的动词、名词和动作（动词+名词）的前 1/5 准确度。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/6489703e49d347cf7463f35d0e5d93e1.png#pic_center)


动作检测
任务：检测未修剪视频 中每个动作的开始和结束。为每个检测到的片段分配一个（动词、名词）标签。
训练输入：一组经过修剪的动作片段，每个片段都用（动词、名词）标签进行注释。
测试输入：一组未剪辑的视频。重要提示：在报告此挑战时，您不得使用测试集中修剪片段的知识。
分裂：训练和训练验证，根据测试分组进行评估。
评估指标：平均精度 (mAP) @ IOU 0.1 至 0.5。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/ec118b1fcbda5b8a2b9bc63a11e88fb9.png#pic_center)


动作预测
任务: 预测未来动作的（动词，名词）标签，观察其发生之前的片段。
训练输入:一组经过修剪的动作片段，每个片段都用（动词、名词）标签进行注释。
测试输入:在测试过程中，您可以观察在您正在测试的操作开始之前至少一秒结束的片段。
分裂:训练和训练验证，根据测试分组进行评估。
评估指标:所有类别的前 5 名平均召回率（如此处定义），针对所有细分以及未见过的参与者和尾部类别进行计算。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/235fdcc51d48da67275d4c866868d6ea.png#pic_center)


无监督自适应动作识别
任务：遵循无监督域适应范式，为修剪后的片段分配（动词、名词）标签：标记的源域用于训练，模型需要适应未标记的目标域。
训练输入：一组经过修剪的动作片段，每个片段都用（动词、名词）标签进行注释。
测试输入：一组经过修剪的未标记动作片段。
分裂。2018 年录制的视频 (EPIC-KITCHENS-55) 构成源域，而为 EPIC-KITCHENS-100 的扩展录制的视频构成未标记的目标域：此挑战使用自定义训练/验证/测试拆分，您可以[在此处](https://github.com/epic-kitchens/epic-kitchens-100-annotations#unsupervised-domain-adaptation-challenge)找到。
评估指标：在目标测试集上，动词、名词和动作（动词+名词）的准确度为 Top-1/5。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/98d9b13e4442bb758d93b77d24e5c093.png#pic_center)


多实例检索

任务：

- 视频到文本：给定查询视频片段，对字幕进行排名，使得排名较高的字幕在语义上与查询视频片段中的动作更相关。
- 文本到视频：给定查询标题，对视频片段进行排名，使得排名较高的视频片段在语义上与查询标题更相关。

训练输入：一组经过修剪的动作片段，每个片段都带有标题注释。字幕对应于从中获得动作片段的英文旁白。
测试输入：一组带有标题的修剪动作片段。重要提示：您不得在测试集中使用已知的对应关系。
分割：此挑战有其自己的自定义分组，可在[此处](https://github.com/epic-kitchens/epic-kitchens-100-annotations/tree/master/retrieval_annotations)获取。
评估指标:归一化累积增益 (nDCG) 和平均精度 (mAP)。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/65e46657c94787fffb31de94a9266026.png#pic_center)


## MECCANO: A Multimodal Egocentric Dataset for Humans Behavior Understanding in the Industrial-like Domain

👉[主页地址](https://iplab.dmi.unict.it/MECCANO/)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/a647b44805840c5ae933b5e8e2421618.gif#pic_center)


可穿戴相机允许从用户的角度获取图像和视频。可以处理这些数据以了解人类的行为。尽管人类行为分析已经在第三人称视觉中得到了彻底的研究，但在以自我为中心的环境中，特别是在工业场景中，它仍然没有得到充分研究。为了鼓励这一领域的研究，我们推出了 MECCANO，这是一个以自我为中心的视频的多模态数据集，用于研究人类在类似工业环境中的行为理解。多模态的特点是存在使用定制耳机同时采集的注视信号、深度图和 RGB 视频。该数据集已被明确标记为从第一人称视角理解人类行为的背景下的基本任务，例如识别和预测人与物体的交互。利用 MECCANO 数据集，我们探索了五种不同的任务，包括 1) 动作识别、2) 活动物体检测和识别、3) 以自我为中心的人与物体交互检测、4) 动作预期和 5) 下一个活动物体检测。我们提出了一个基准，旨在研究在所考虑的类似工业场景中的人类行为，该基准表明所研究的任务和所考虑的场景对最先进的算法具有挑战性。

2 个国家（IT、英国）的 20 个不同科目
3 种模式：RGB、深度和凝视
视频采集。RGB：1920x1080 12.00 fps，深度：640x480 12.00 fps
凝视：频率为 200Hz
11 个培训视频和 9 个验证/测试视频
8857 个带有时间注释的视频片段，指示描述所执行动作的动词
64349 个活动对象在接触框中用边界框注释
48024 个在过去帧中注释的下一个活动对象
89628只手在过去的框架和接触框架中用边界框注释
12 个动词类、20 个宾语类和 61 个动作类

【下载详情】

- [RGB Videos](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_RGB_Videos.zip)
- [RGB Frames](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_RGB_frames.zip)
- [Depth Frames](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_Depth_frames.zip)
- [Gaze Data](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_Gaze_data.zip)
- [Action Temporal Annotations](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_action_annotations.zip)
- [EHOI Verb Temporal Annotations](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_verb_temporal_annotations.zip)
- [Active Object Bounding Box Annotations](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_active_object_bounding_box_annotations.zip) and [frames](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_active_object_frames.zip)
- [Hands Bounding Box Annotations](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_hands_bounding_box_annotations.zip)
- [Next-Active Object Bounding Box Annotations](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_NAO_bounding_box_annotations.zip)
- [CODE](https://github.com/fpv-iplab/MECCANO)

考虑到 MECCANO 数据集的多模性以及获取它时具有挑战性的类似工业场景，MECCANO 数据集适合研究各种任务。我们考虑了与人类行为理解相关的五项任务，并为其提供了基线结果：1) 动作识别，2) 主动对象检测和识别，3) 以自我为中心的人与对象交互 (EHOI) 检测，4) 动作预期和 5) 下一个活动物体检测。

动作识别

动作识别包括从以自我为中心的视频片段中确定相机佩戴者执行的动作。给定一个片段，目标是分配正确的动作类。

主动物体检测和识别

活动对象检测任务的目的是检测 EHOI 中涉及的所有活动对象。目标是使用边界框检测每个活动对象。主动对象识别任务还包括考虑 MECCANO 数据集的 20 个对象类别，为它们分配正确的类别标签。

EHOI检测

目标是确定每幅图像中以自我为中心的人机交互 (EHOI)。特别是，其目的是检测和识别场景中带有边界框的所有活动对象，以及描述人类执行的动作的动词。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/11d6f0d32520f74d544478a57503f296.png#pic_center)


行为预估

行为预估任务的目标是根据对过去的观察来预测未来的以自我为中心的行动。


下一个活动物体检测

下一个活动对象检测任务的目的是检测和识别未来交互中涉及的所有对象。

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/1acf3be866f856ee6a501f8e523beab2.png#pic_center)


## TREK-100

👉[主页地址](https://machinelearning.uniud.it/datasets/trek150/)/[下载地址](https://machinelearning.uniud.it/datasets/trek150/TREK-150-annotations.zip)👈

为此数据集生成的注释包含在该[存档](https://machinelearning.uniud.it/datasets/trek150/TREK-150-annotations.zip)中（您将找到 TREK-150 中包含的每个序列的 zip 存档）。由于 EK-55 政策，TREK-150 序列的视频帧无法直接重新分配。因此，您不会直接在注释文件夹中找到它们，但它们会自动为您下载。

只需运行即可构建完整的 TREK-150 数据集

~~~python
pip install got10k
git clone https://github.com/matteo-dunnhofer/TREK-150-toolkit
cd TREK-150-toolkit
python download.py
~~~

这将下载原始 EK-55 MP4 视频，使用 提取感兴趣的帧ffmpeg，并准备将从 zip 存档中提取的注释文件。整个过程完成后，你会发现dataset文件夹中有100个目录。每个定义一个视频序列。

每个序列文件夹将包含一个目录

img/：将序列的视频帧包含为*.jpg文件。
以及以下*.txt文件：

groundtruth_rect.txt：包含目标物体的真实轨迹。每行上以逗号分隔的值表示目标对象在每个相应帧的边界框位置 [x,y,w,h]（左上角的坐标以及宽度和高度）（第一行 ->第一帧的目标位置，最后一行 -> 最后一帧的目标位置）。值为 -1、-1、-1、-1 的行指定目标对象在此类框架中不可见。
action_target.txt：包含相机佩戴者执行的操作的标签（作为动词-名词对）和目标对象类别。该文件报告 3 个行分隔的数字。第一个值是动作动词标签，第二个是动作名词标签，第三个是目标对象的名词标签（动作名词和目标名词在某些序列上不一致）。动词标签是根据该文件verb_id的索引获得的。名词标签和目标名词标签是根据该文件的索引获得的。noun_id
attributes.txt：包含序列的跟踪属性。该文件报告行分隔的字符串，这些字符串取决于序列中发生的跟踪情况。这些字符串是首字母缩略词，其解释可以在主论文的表 2 中找到。
frames.txt：包含相对于完整 EK-55 视频的序列的帧索引。
anchors.txt：包含起点（锚点）的帧索引和评估方向（0 -> 时间向前，1 -> 时间向后）以实现 MSE（多起点评估）协议。
lh_rect.txt：包含相机佩戴者左手的地面实况边界框。每行上以逗号分隔的值表示手在每个相应帧（第一行 -> 目标位置）的边界框位置 [x,y,w,h]（左上角的坐标以及宽度和高度）对于第一帧，最后一行 -> 最后一帧的手位置）。值为 -1、-1、-1、-1 的行指定手在此类帧中不可见。
rh_rect.txt：包含相机佩戴者右手的地面实况边界框。每行上以逗号分隔的值表示手在每个相应帧（第一行 -> 目标位置）的边界框位置 [x,y,w,h]（左上角的坐标以及宽度和高度）对于第一帧，最后一行 -> 最后一帧的手位置）。值为 -1、-1、-1、-1 的行指定手在此类帧中不可见。
lhi_labels.txt：包含表示相机佩戴者的左手是否与目标对象接触的地面实况标签。每行上的二进制值表示每个相应帧中手与物体之间是否存在接触（0 -> 无接触，1 -> 接触）（第一行 -> 第一帧的交互，最后一行 -> 最后一帧的交互框架）。
rhi_labels.txt：包含表示相机佩戴者的右手是否与目标对象接触的地面实况标签。每行上的二进制值表示每个相应帧中手与物体之间是否存在接触（0 -> 无接触，1 -> 接触）（第一行 -> 第一帧的交互，最后一行 -> 最后一帧的交互框架）。
bhi_labels.txt：包含表示相机佩戴者的双手是否与目标对象接触的地面实况标签。每行上的二进制值表示每个帧中手与物体之间是否存在接触（0 -> 无接触，1 -> 接触）（第一行 -> 第一帧的交互，最后一行 -> 最后一帧的交互框架）。
anchors_hoi.txt：包含起点和终点（锚点）的帧索引和交互类型（0 -> 左手交互，1 -> 右手交互，2 -> 双手交互），以实现 HOI（手部对象交互评估） ） 协议。
该代码使用 Python 3.7.9 和ffmpeg4.0.2 进行了测试。*.MP4下载过程中生成的所有临时文件（例如文件、非相关帧）将在下载过程完成后自动删除。如果提前停止，下载过程可以从上次下载的序列恢复。

下载过程最多可能需要 24 小时才能完成。

## EgoCom：多人多模式以自我为中心的通信数据集

👉[主页地址](https://github.com/facebookresearch/EgoCom-Dataset)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/58d1d51b113bcb8ae3caea14a329802f.png#pic_center)


以自我为中心的通信 (EgoCom) 是首个自然对话数据集，包含从参与者以自我为中心的角度同时捕获的多模式人类通信数据。EgoCom 数据集包括 38.5 小时的对话，其中包括同步的立体声音频和以自我为中心的视频，以及来自 34 位不同说话者的 240,000 个真实情况、带时间戳的字级转录和说话者标签。

该存储库提供：

[EgoCom数据集](https://github.com/facebookresearch/EgoCom-Dataset/tree/main/egocom_dataset)
要下载 EgoCom 数据集，请转到 [此处](https://github.com/facebookresearch/EgoCom-Dataset#download-the-egocom-dataset)
有关 EgoCom 数据集的详细信息请参见 [此处](https://github.com/facebookresearch/EgoCom-Dataset#egocom-dataset-contents)
Pythonegocom包
用于处理多视角以自我为中心的通信数据的包
音频功能、转录、对齐等。详细信息 [此处](https://github.com/facebookresearch/EgoCom-Dataset#egocom-datasetegocom-package----code-details)
EgoCom 论文的工作日志
透明度、可重复性以及 EgoCom 数据集创建的快照。

结构：

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/1760bc76122c0384aec183245c164ff9.png#pic_center)



## Ego4D

👉[主页地址](https://ego4d-data.org/)👈

![在这里插入图片描述](https://cyfedu-dlut.github.io/PersonalWeb/images/58985ee1f8a5b4a122920d6851d99e2d.png#pic_center)


大规模、以自我为中心的数据集和基准套件，收集了全球 74 个地点和 9 个国家/地区的数据，包含超过3,670小时的日常生活活动视频。目前最大的第一视角数据集，包含五大挑战基准，每一个部分又可以细分为四个挑战任务，如下所示：

- Episodic Memory/情景记忆【[主页地址](https://github.com/EGO4D/episodic-memory)/[Github地址](https://github.com/EGO4D/episodic-memory)】
  - Episodic Memory/情景记忆: 我的X在哪里？第一视角视频记录了佩戴者的日常生活，并且可以根据需要用于增强人类记忆。这样的系统也许能够提醒用户他们把钥匙放在哪里，是否在食谱中添加了盐，或者回忆他们参加过的活动。相当于一种视频文本检索的多模态任务。
    - VQ2D：Visual Queries with 2D Localization——该任务询问：“我上次看到\[这个\]是什么时候？” 给定一个第一视角视频片段和一个描述查询对象的图像片段，目的是根据跟踪的边界框（2D + 时序定位）返回输入视频中该对象的最后一次出现。这项任务的新颖之处在于升级传统的对象实例识别以处理视频，特别是具有挑战性的视图转换的第一视角视频。

    - VQ3D： Visual Queries with 3D Localization——该任务询问：“我最后一次看到\[这个\]是在哪里？” 给定一个第一视角视频剪辑和一个描述查询对象的图像片段信息，目标是定位上次在视频中看到它的时间，并返回从查询帧的相机中心到 3D 对象中心的 3D 位移向量。因此，该任务建立在上述 2D 定位的基础上，并将其扩展为需要在 3D 环境中进行定位。该任务的新颖之处在于它需要视频对象实例识别和 3D 推理。
    - NLQ：Natural Language Queries——此任务询问“什么/何时/何地......？” ——有关过去视频的一般自然语言问题。给定一个视频片段和一个用自然语言表达的查询，目标是定位所有视频历史记录中问题答案显而易见的时间窗口。这项任务很新颖，因为它需要搜索视频来回答灵活的语言查询。为简洁起见，这些示例剪辑说明了围绕真实情况的视频（而原始输入视频每个约为 8 分钟）。
    - MQ：Moments Queries——该任务询问：“我什么时候做了 X？” 给定一个以自我为中心的视频和一个活动名称（即“时刻”），目标是定位过去视频中该活动的所有实例。任务是活动检测，但专门针对相机佩戴者的自我中心活动。很大程度上是看不见的。

  - Querying Memory/查询记忆: 根据用于查询内存的输入类型，该基准测试中有三个不同的任务：视觉查询（即找到给定钥匙图像的位置）、文本查询（“我添加了多少杯糖？”）和瞬间查询（查找“我什么时候和狗一起玩”的所有实例）。

  - Construction of Queries/查询的构建: 对于语言查询，设计了一组模板，注释者用来为任务编写问题。示例包括“对象 X 的状态是什么？” 或者“事件Y之后对象X在哪里”？然后这些内容被重新编写以实现多样性。

  - Recalling Lives/回忆生活: 鉴于此基准的广泛性质，该任务中没有重点关注的活动子集，从而形成了现实且具有挑战性的基准。

- HOI（Hand Object Interaction）
  - 手和物体基准测试捕捉相机佩戴者如何通过使用或操纵物体来改变物体的状态——我们称之为物体状态变化。尽管可以通过多种方法（例如，各种工具、力、速度、抓握、末端执行器）来实现将一块木材切成两半，但所有这些都应该被视为相同的状态变化。可以沿着时间、空间和语义轴查看对象状态变化，从而导致以下三个任务：

    - 无返回点时间定位：给定状态变化的短视频片段，目标是估计包含无返回点（PNR）（Point-of-no-return temporal localization:）的关键帧

    - （State change object detection）状态变化目标检测：给定三个时间帧（前、后、PNR），目标是回归经历状态变化的对象的边界框

    - （Object state change classification）目标状态变化分类：给定一个短视频片段，目标是对对象状态变化是否发生进行分类

- AV Diarization视听分类
  - 视听分类 (AVD) 包括专注于检测、跟踪、说话者分割和语音内容转录的任务。为此，我们在此基准测试中提出 4 项任务。AVD 数据集的第一个版本中提供了总计超过 750 小时的对话数据。其中大约 50 小时的数据已被注释以支持这些任务。这对应于 572 个剪辑。其中 389 个用于训练，50 个用于验证，其余将用于测试。每个片段时长 5 分钟。以下架构总结了片段的一些数据统计。
    - 每个片段的发言者：4.71
    - 每帧的发言者：0.74
    - 剪辑中的发言时间：219.81 秒
    - 剪辑中每人的发言时间：43.29 秒
    - 摄像机佩戴者发言时间：77.64 秒
    四个任务如下：
    - 定位和跟踪：此任务的目标是检测视野中的所有说话者并在视频剪辑中跟踪他们。我们为每个参与者的脸部提供边界框以实现此任务。

    - 主动说话人检测：在此任务中，每个被跟踪的说话人都被分配一个匿名标签，包括从未出现在视野中的摄像头佩戴者。

    - 分类（仅音频或视听）：此任务重点关注前 2 个任务中已本地化、跟踪并分配匿名标签的说话者的语音活动。对于此任务，我们提供与剪辑中每个说话者的语音活动相对应的时间段。

    - 转录：对于最后一个任务，我们转录演讲内容。

- Social
    [【查看此处】](https://github.com/EGO4D/social-interactions)

- Forecasting
  - 短期预测：
    - 这项任务旨在预测给定时间戳后发生的下一次人与物体的交互。给定输入视频，目标是预测场景中活动物体的空间位置（如物体周围的边界框）。我们认为下一个活动物体是用户将触摸（用手或工具）以启动交互的下一个物体；每个被检测到的下一个活动物体的类别（如 "小刀"、"西红柿"）；如何使用每个活动物体，即对活动物体执行什么操作（如 "拿"、"切"）；与每个对象的交互何时开始（如 "1 秒后"、"0.25 秒后"）。这是用户接触活动对象的第一帧的时间（接触时间）。这种预测在涉及人机协作的场景中非常有用。例如，如果预测接触潜在危险物体的行动时间很短，辅助系统就可以发出警报。在这项任务中，模型需要在特定的时间戳进行预测，而不是在整个视频中进行密集预测。允许模型处理视频至给定帧 t，此时它必须预测下一个活动物体，以及它们将如何在 Δ 秒内参与互动，其中 Δ 是未知数。模型可以做出零个或多个预测。每项预测都会指出下一个活跃对象，包括名词类别、边界框、表示未来动作的动词以及接触时间（估计未来多少秒后将开始与该对象互动）。每个预测还包含一个用于评估的置信度分数。
  - 长期预测
    - 该任务旨在预测给定动作之后的下一个 Z 未来动作。给定截至特定时间步的输入视频（对应最后一个可见动作），目标是预测其后的动作类别列表[（动词 1，名词 1）、（动词 2，名词 2）......（动词 Z，名词 Z）]。模型应生成 K 个这样的列表，以考虑动作序列的变化。对于这项任务，我们设定 Z=20 和 K=5。
  - 未来手部预测
    - 作为 Ego4D 挑战套件中预测基准的一部分，我们考虑了预测关键帧的未来手部位置这一具有挑战性的任务。给定一个简短的视频片段，根据关键帧中的边界框中心，预测未来手的可见位置

## EgoCap

👉[主页地址](https://vcai.mpi-inf.mpg.de/projects/EgoCap/)/[原始数据集](http://resources.mpi-inf.mpg.de/EgoCap/training_v000.zip)/[增强数据集](http://resources.mpi-inf.mpg.de/EgoCap/training_v002.zip)/[验证集2D](http://resources.mpi-inf.mpg.de/EgoCap/validation_v003_2D.zip)/[验证集3D](http://resources.mpi-inf.mpg.de/EgoCap/validation_v003_3D.zip)👈

基于标记和无标记的光学骨骼运动捕捉方法使用由外向内排列的摄像机围绕场景放置，视点汇聚在中心。他们经常因可能需要的标记套装而造成不适，并且他们的录音音量受到严格限制，并且通常仅限于背景受控的室内场景。替代的基于套装的系统使用多个惯性测量单元或外骨骼来捕捉运动。这使得捕获独立于有限的体积，但需要大量、通常受到限制且难以设置的身体仪器。因此，我们提出了一种实时、无标记和以自我为中心的运动捕捉的新方法，该方法通过连接到头盔或虚拟现实耳机的一对轻型立体鱼眼相机来估计全身骨骼姿势。它将鱼眼视图的新生成姿势估计框架的优势与在大型新数据集上训练的基于 ConvNet 的身体部位检测器相结合。我们的内向方法可以捕捉一般室内和室外场景中的全身运动，也可以捕捉附近有很多人的拥挤场景。设置时间和精力很少，并且被捕获的用户可以自由移动，这使得能够重建更大规模的活动，并且在虚拟现实中特别有用，可以自由漫游和交互，同时看到完全动作捕捉的虚拟身体。还有附近有很多人的拥挤场景。设置时间和精力很少，并且被捕获的用户可以自由移动，这使得能够重建更大规模的活动，并且在虚拟现实中特别有用，可以自由漫游和交互，同时看到完全动作捕捉的虚拟身体。还有附近有很多人的拥挤场景。设置时间和精力很少，并且被捕获的用户可以自由移动，这使得能够重建更大规模的活动，并且在虚拟现实中特别有用，可以自由漫游和交互，同时看到完全动作捕捉的虚拟身体。

## EgoClip

👉[主页地址](https://github.com/showlab/EgoVLP)👈

我们创建了EgoClip，这是一个第一人称视频文本预训练数据集，包括3.8M个从Ego4D中精心选择的剪辑文本对，涵盖了各种人类日常活动。

## EgoTracks
👉[主页地址](https://ego4d-data.org/docs/data/egotracks/)👈
一个用于长期自我中心视觉目标跟踪的新数据集。EgoTracks源自Ego4D数据集，为最近的最先进单对象跟踪器提供了显著挑战，按照传统跟踪指标，我们发现它们在我们的新数据集上的评分明显低于现有的流行基准。我们进一步展示了对STARK跟踪器可以进行的改进，以显著提高其在自我中心数据上的表现，最终形成了我们称之为EgoSTARK的基线模型。
![egotracks](https://cyfedu-dlut.github.io/PersonalWeb/images/egotracks.png)

## IT3DEgo
👉[主页地址](https://github.com/IT3DEgo/IT3DEgo/?tab=readme-ov-file)👈
👉[下载地址](https://drive.usercontent.google.com/download?id=1VVszWG4mmm0g3ai3EoZw-3cGNBmZCN-9&export=download&authuser=0&confirm=t&uuid=5c7c5869-a940-426b-aca4-c058241712eb&at=APvzH3pbj50ZGV5l-i3tiO_IIEtR%3A1734692440468)👈

![it3dego](https://cyfedu-dlut.github.io/PersonalWeb/images/it3dego.png)

基准数据集可在此处下载。基准数据集约为 900GB，包含以下三个部分：

原始视频序列。使用 HoloLens 2 捕获的具有每帧相机姿势的 RGB-D 视频序列。视频数据按以下结构组织：
# Raw video sequence structure

├── Video Seq 1
│   ├── pv                    # rgb camera
│   ├── depth_ahat            # depth camera
│   ├── vlc_ll                # left-left grayscale camera
│   ├── vlc_lf                # left-front grayscale camera
│   ├── vlc_rf                # right-front grayscale camera
│   ├── vlc_rr                # right-right grayscale camera
│   ├── mesh                  # coarse mesh of the surrounding environment
│   ├── pv_pose.json
│   ├── depth_ahat_pose.json
│   ├── vlc_ll_pose.json
│   ├── vlc_lf_pose.json
│   ├── vlc_lf_pose.json
│   └── vlc_rr_pose.json
.
.
.
└── Video Seq N
每个相机姿态 JSON 文件（例如 pv_pose.json 或depth_ahat_pose.json）都包含时间戳和相机矩阵的键值对。我们还包含与每个原始视频序列相对应的校准文件。校准文件指定相机参数和每个相机之间的变换矩阵（有关更多详细信息，请参阅hl2ss）。有关在 HoloLens 2 上重新分级不同相机规格（pv、深度和灰度）的更多信息，请查看arXiv 上的此文档。

注释。我们提供三种类型的注释来支持基准问题的研究。注释数据按以下结构组织：
# Annotations structure

├── Video Seq 1
│   ├── labels.csv
│   ├── 3d_center_annot.txt
│   ├── motion_state_annot.txt
│   ├── 2d_bbox_annot
│   │   ├── 0.txt
│   │   .
│   |   .
│   |   .
│   |   └── K.txt
│   └── visuals                 # clear visuals of object instances on a specific frame, only for visualization
│       ├── timestamp_instance_1.png
│       .
│       .
│       .
│       └── timestamp_instance_K.png
.
.
.
└── Video Seq N
中的每一行都label.csv描述了要跟踪的对象实例的名称，例如 cup_1。该文件motion_state_annot.txt描述了每帧的二进制对象运动状态。此文件中的每一行都具有 的格式。和instance id, timestamp_start, timestamp_end之间的间隔表示对象在此期间保持静止。换句话说，对象实例在每个间隔之外都与用户进行交互。在文件夹中，如果对象可见，我们大约每 5 帧为每个对象提供一次轴对齐的 2D 边界框。对应于中描述的第一个对象实例。中的每一行都具有 的格式。该文件在预定义的世界坐标中包含每个对象实例的 3D 中心。此文件中的每一行都具有 的格式。位置 id 以从零开始的索引描述当前对象的位置变化次数。timestamp_starttimestamp_end2d_bbox_annot0.txtlabel.csv0.txttimestamp, x_min, y_min, x_max, y_max3d_center_annot.txttimestamp_start, timestamp_end, instance id, x, y, z, location id

注册信息。我们研究了两种不同的设置来指定感兴趣的对象实例：单视图在线注册 (SVOE) 和多视图预注册 (MVPE)。请查看我们的论文以了解每种注册的详细描述。
# Single-view online enrollment (SVOE)

├── Video Seq 1
│   └── svoe.txt
.
.
.
└── Video Seq N

===============================================================

# Multi-view pre-enrollment (MVPE)

├── Instance 1                # folder name corresponds to instance names in label.csv in the annotation folder
│   ├── instance_image_1.jpg
│   .
│   .
│   .
│   └── instance_image_24.jpg
.
.
.
└── Instance M
文件中的每一行svoe.txt代表instance id, timestamp, x_min, y_min, x_max, y_max。 Instand id 对应于注释文件夹中的 label.csv，索引从零开始。换句话说，中的第一个对象实例Annotations/Video Seq 1/label.csv对应于 中的实例 0 Video Seq 1/svoe.txt。

## Ego-Exo4D
👉[主页地址](https://ego-exo4d-data.org/)👈
一个多样化、大规模的多模式、 多视图视频数据集和基准，由 740 名相机佩戴者在全球 13 个城市收集，捕捉了 1286.3 小时的熟练人类活动视频。
![egoexo4d](https://cyfedu-dlut.github.io/PersonalWeb/images/egoexo4d.png)
![egoexo4d2](https://cyfedu-dlut.github.io/PersonalWeb/images/egoexo4d2.png)

# CONTINUE
