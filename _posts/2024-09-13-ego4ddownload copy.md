---
title: '国内下载Ego4D数据集'
date: 2024-09-13
permalink: /posts/2024/09/ego4ddownload/
tags:
  - Ego4D Dataset
  - AWS S3 cli
  - technical skills
---

第一人称视角数据集Ego4D的下载对于国人来说不友好，那么我们除了通过VPN挂载北美相关节点获取到的大容量流量下载之外，是否还有其他方式？这里作者亲身实测给出相关技术要领。

下面是如何使用 AWS S3 CLI 下载 Ego4D 数据集的完整步骤。为了加速下载，建议使用 AWS EC2 实例进行中转，最终通过 SCP 下载数据到本地。

### 第一步：配置 AWS CLI
1. **安装 AWS CLI**  
   首先在你的本地机器或 EC2 实例上安装 AWS CLI。如果未安装，可以按照以下命令安装：
   - 在 Linux/Mac 上：
     ```bash
     curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
     sudo installer -pkg AWSCLIV2.pkg -target /
     ```
   - 在 Windows 上：
     下载并运行 [AWS CLI MSI 安装包](https://aws.amazon.com/cli/)。

2. **配置 AWS CLI**  
   获取 AWS 访问密钥，配置 AWS CLI：
   ```bash
   aws configure
   ```
   需要输入以下信息：
   - `AWS Access Key ID`: 你的 AWS 访问密钥 ID。
   - `AWS Secret Access Key`: 你的 AWS 秘密访问密钥。
   - `Default region name`: 你打算使用的 AWS 区域（如 `ap-east-1`，香港区域）。
   - `Default output format`: 可以设置为 `json`、`text` 或 `table`。

### 第二步：创建 EC2 实例
1. **登录 AWS 管理控制台**  
   登录到 [AWS 管理控制台](https://aws.amazon.com/console/) 并导航到 EC2 服务。

2. **启动 EC2 实例**  
   - 选择一个靠近中国的区域，例如香港 (`ap-east-1`) 或东京 (`ap-northeast-1`)。
   - 点击 “Launch Instance”，选择合适的 AMI（Amazon Linux 或 Ubuntu 等）。
   - 选择实例类型（如 `t2.micro`，如果你使用的是免费套餐，适合较轻任务）。
   - 配置网络和安全组，确保允许 SSH 连接。
   - 生成并下载密钥对（`*.pem` 文件），用于 SSH 登录。

3. **连接到 EC2 实例**  
   使用以下命令 SSH 登录到 EC2 实例：
   ```bash
   ssh -i "your-key.pem" ec2-user@your-ec2-instance-public-ip
   ```

### 第三步：从 S3 下载数据集
1. **安装 AWS CLI（如果未预安装）**  
   如果你的 EC2 实例未预装 AWS CLI，可以使用以下命令安装：
   ```bash
   sudo yum install aws-cli -y  # For Amazon Linux
   sudo apt install awscli -y   # For Ubuntu
   ```

2. **下载 Ego4D 数据集**  
   确保你已经获得了下载数据集的权限，并通过以下命令从 S3 下载数据集：
   ```bash
   aws s3 cp s3://bucket-name/path/to/ego4d-dataset/ /your/local/path/ --recursive
   ```
   你需要将 `bucket-name` 和 `path/to/ego4d-dataset` 替换为 Ego4D 数据集的具体路径。

### 第四步：通过 SCP 将数据下载到本地
1. **将数据压缩**  
   因为数据集较大，建议压缩文件以加速传输：
   ```bash
   tar -czvf ego4d-data.tar.gz /path/to/ego4d-data/
   ```

2. **通过 SCP 将数据传回本地**  
   使用以下命令通过 SCP 将数据从 EC2 实例传输到本地机器：
   ```bash
   scp -i "your-key.pem" ec2-user@your-ec2-instance-public-ip:/path/to/ego4d-data.tar.gz /your/local/path/
   ```

### 第五步：清理 EC2 实例
下载完成后，可以终止 EC2 实例来避免产生额外费用：
1. 登录 AWS 管理控制台，进入 EC2 控制台。
2. 选择你的实例，点击 “Terminate” 以终止它。

### 总结
1. 安装并配置 AWS CLI。
2. 创建 EC2 实例并连接到实例。
3. 使用 AWS S3 CLI 从 S3 下载数据集到 EC2 实例。
4. 使用 SCP 从 EC2 实例下载数据到本地。
5. 终止 EC2 实例。

通过这些步骤，你可以有效地在中国环境下下载 Ego4D 数据集。