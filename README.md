# Diffusion Demo - DiT 扩散模型训练与采样

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/CUDA-可选-green.svg" alt="CUDA Optional">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
</p>

一个完整的基于 **DiT (Diffusion Transformer)** 的图像生成演示系统，支持 Web 界面交互和命令行训练。本项目实现了 VAE/VQVAE 变分自编码器和基于 Transformer 的扩散模型，用于 48×48 人脸图像的生成。

## 📋 目录

- [特性](#特性)
- [模型架构](#模型架构)
  - [VAE (变分自编码器)](#vae-变分自编码器)
  - [VQVAE (向量量化变分自编码器)](#vqvae-向量量化变分自编码器)
  - [DiT (扩散 Transformer)](#dit-扩散-transformer)
  - [DDPM (去噪扩散概率模型)](#ddpm-去噪扩散概率模型)
- [数学原理](#数学原理)
- [安装](#安装)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [API 文档](#api-文档)

## ✨ 特性

- 🎨 **完整的图像生成流程**: 数据预览 → VAE 训练 → DiT 训练 → 图像采样
- 🌐 **Web 界面**: 基于 Flask 的交互式界面，支持实时训练监控
- 📊 **指标追踪**: 完整的训练指标记录和可视化报告
- 🖼️ **中间结果可视化**: 训练过程中生成图像和重建对比
- 🔄 **Job 系统**: 后台任务管理，支持任务队列和状态追踪
- ⚡ **高效实现**: 支持 BF16 混合精度、CUDA 加速

## 🧠 模型架构

### VAE (变分自编码器)

VAE 通过编码器将图像压缩到潜在空间，再通过解码器重建图像。

#### 网络结构

```
输入图像 (3×48×48)
    ↓
[编码器]
  Conv2d(3→128, 4×4, stride=2)  → 24×24
  ResBlock(128)
  Conv2d(128→256, 4×4, stride=2) → 12×12
  ResBlock(256)
  Conv2d(256→latent_dim, 3×3)    → 4×4
    ↓
潜在变量 z (latent_dim×4×4)
    ↓
[重参数化技巧]
  z = μ + σ·ε,  ε ~ N(0, I)
    ↓
[解码器]
  ConvTranspose2d(latent_dim→256, 4×4, stride=2) → 8×8
  ResBlock(256)
  ConvTranspose2d(256→128, 4×4, stride=2)        → 16×16
  ResBlock(128)
  ConvTranspose2d(128→128, 4×4, stride=2)        → 32×32
  Interpolate(48×48)
  Conv2d(128→3, 3×3)                               → 48×48
    ↓
重建图像 (3×48×48)
```

#### 损失函数

VAE 的损失函数由两部分组成：

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}
$$

**重建损失** (MSE):

$$
\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|^2
$$

**KL 散度** (正则化项):

$$
\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)
$$

其中：
- $x_i$ 是原始图像
- $\hat{x}_i$ 是重建图像
- $\mu_j, \sigma_j$ 是潜在变量的均值和标准差
- $\beta$ 是 KL 权重系数 (默认 1.0)

### VQVAE (向量量化变分自编码器)

VQVAE 通过离散化的潜在空间学习更有意义的表示。

#### 网络结构

```
输入图像 (3×48×48)
    ↓
[编码器]
  Conv2d(3→128, 4×4, stride=2)   → 24×24
  Conv2d(128→64, 4×4, stride=2)  → 12×12
  Conv2d(64→embedding_dim, 3×3)  → 4×4 (通过 Adaptive Pooling)
    ↓
连续潜在变量 z_e (embedding_dim×4×4)
    ↓
[向量量化层]
  查找最近邻嵌入: z_q = e_k, 其中 k = argmin_j ||z_e - e_j||
    ↓
离散潜在变量 z_q (embedding_dim×4×4)
    ↓
[解码器]
  与 VAE 解码器类似，但输出为量化后的潜在变量
    ↓
重建图像 (3×48×48)
```

#### 损失函数

VQVAE 的损失函数包含三部分：

$$
\mathcal{L}_{\text{VQVAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{VQ}} + \gamma \cdot \mathcal{L}_{\text{commit}}
$$

**VQ 损失**:

$$
\mathcal{L}_{\text{VQ}} = \|z_e - \text{sg}[z_q]\|^2
$$

**承诺损失**:

$$
\mathcal{L}_{\text{commit}} = \beta \cdot \|\text{sg}[z_e] - z_q\|^2
$$

其中：
- $z_e$ 是编码器输出的连续潜在变量
- $z_q$ 是量化后的离散潜在变量
- $\text{sg}[\cdot]$ 是停止梯度操作 (stop-gradient)
- $\beta$ 是承诺成本 (默认 0.25)

### DiT (扩散 Transformer)

DiT 将扩散模型中的 U-Net 替换为 Transformer 架构，在潜在空间进行去噪。

#### 网络结构

```
输入: 带噪潜在变量 x_t (latent_channels×4×4), 时间步 t
    ↓
[Patch Embedding]
  将 4×4 图像划分为 2×2 的 patches → 4 个 patches
  每个 patch 投影到 hidden_size 维度
    ↓
[位置编码] + 可学习的 positional embedding
    ↓
[Transformer Blocks] × depth (默认 6 层)
  - 多头自注意力 (Multi-Head Self-Attention)
  - AdaLN (Adaptive Layer Norm): 根据时间步和类别条件调制
  - FFN (Feed-Forward Network)
    ↓
[输出层]
  LayerNorm + Linear → 预测噪声 ε_θ(x_t, t)
    ↓
输出: 预测的噪声 (latent_channels×4×4)
```

#### AdaLN (自适应层归一化)

每个 Transformer 块使用 AdaLN 进行条件调制：

$$
\text{AdaLN}(x, c) = x \cdot (1 + \gamma(c)) + \beta(c)
$$

其中 $c$ 是时间步和类别条件的嵌入向量，$\gamma$ 和 $\beta$ 是通过 MLP 从 $c$ 学习得到的缩放和偏置参数。

### DDPM (去噪扩散概率模型)

DDPM 定义了前向加噪和反向去噪两个马尔可夫链过程。

#### 前向扩散过程

从原始数据 $x_0$ 逐渐添加高斯噪声，经过 $T$ 步后变成纯噪声：

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

利用重参数化技巧，可以直接采样任意时间步 $t$ 的状态：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

其中：
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$
- $\beta_t$ 是噪声调度 (线性或余弦调度)

#### 反向去噪过程

学习一个神经网络 $\epsilon_\theta$ 来预测噪声，从而从 $x_t$ 恢复 $x_{t-1}$：

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)
$$

其中均值预测为：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

#### 训练目标

简化后的训练目标是预测添加到 $x_0$ 的噪声：

$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x_0, t, \epsilon \sim \mathcal{N}(0,I)} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]
$$

#### 采样算法

```
从标准正态分布采样 x_T ~ N(0, I)
对于 t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1, else z = 0
    ε_θ = model(x_t, t)  # 预测噪声
    x_{t-1} = (x_t - β_t·ε_θ/√(1-ᾱ_t)) / √α_t + σ_t·z
返回 x_0
```

## 📐 数学符号速查

| 符号 | 含义 |
|------|------|
| $x_0$ | 原始图像 |
| $x_t$ | 时间步 $t$ 的带噪图像 |
| $z, z_e, z_q$ | 潜在变量 (连续/量化前/量化后) |
| $\mu, \sigma$ | 潜在变量的均值和标准差 |
| $\epsilon, \epsilon_\theta$ | 真实噪声和模型预测的噪声 |
| $\beta_t$ | 时间步 $t$ 的噪声方差 |
| $\alpha_t = 1 - \beta_t$ | 保留率 |
| $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ | 累积保留率 |
| $T$ | 总扩散时间步 (通常 1000) |
| $d$ | 潜在空间维度 |
| $N$ | 批量大小 |

## 🚀 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA (可选，但推荐用于训练)

### 安装步骤

```bash
# 克隆项目
git clone <repository-url>
cd diffusion_demo2

# 安装依赖
pip install -r requirements.txt

# 或者安装为可编辑包
pip install -e .
```

### 依赖列表

```
flask>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
numpy>=1.21.0
requests>=2.25.0
tqdm>=4.60.0
tensorboard>=2.10.0
einops>=0.6.0
```

## 🎯 使用方法

### Web 界面 (推荐)

启动 Web 服务器：

```bash
python -m diffusion_demo.server
```

访问 http://127.0.0.1:5000，你将看到：

1. **Dataset 标签**: 下载和预览 CelebA 48×48 数据集
2. **Train VAE/VQVAE 标签**: 训练变分自编码器
3. **Train DiT 标签**: 训练扩散 Transformer
4. **Sample 标签**: 使用训练好的模型生成图像
5. **Jobs 标签**: 查看训练任务历史和日志

### 命令行训练

进行快速 VAE 训练并生成可视化报告：

```bash
python cmdline_train.py \
    --model vae \
    --epochs 30 \
    --batch-size 64 \
    --data-limit 30000 \
    --lr 0.001
```

参数说明：
- `--model`: 模型类型 (`vae` 或 `vqvae`)
- `--epochs`: 训练轮数
- `--batch-size`: 批量大小
- `--data-limit`: 使用数据量 (最大 30000)
- `--lr`: 学习率
- `--bf16`: 启用 BF16 混合精度

训练完成后会生成：
- `runs/cmdline/final_model.pth`: 训练好的模型
- `runs/cmdline/report.html`: 包含图表和生成图像的可视化报告

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | 5000 | Web 服务器端口 |
| `HOST` | 127.0.0.1 | 服务器主机 |
| `DEVICE` | auto | 计算设备 (auto/cuda/cpu) |
| `DATA_LIMIT_DEFAULT` | 2048 | 默认数据限制 |

## 📁 项目结构

```
diffusion_demo2/
├── CLAUDE.md                   # 原始 SPEC 文档
├── README.md                   # 本文件
├── requirements.txt            # Python 依赖
├── pyproject.toml              # 包配置
├── cmdline_train.py            # 命令行训练脚本
│
├── src/diffusion_demo/         # 主包
│   ├── server.py               # Flask 服务器入口
│   ├── api.py                  # REST API 路由
│   ├── jobs.py                 # 任务管理系统
│   │
│   ├── core/                   # 核心功能
│   │   ├── data_loader.py      # CelebA 数据加载
│   │   ├── models/             # 模型定义
│   │   │   ├── vae.py          # VAE/VQVAE/判别器
│   │   │   └── dit.py          # DiT 模型
│   │   └── training/           # 训练逻辑
│   │       ├── train_vae.py    # VAE 训练
│   │       └── train_dit.py    # DiT 训练和采样
│   │
│   ├── ui/                     # Web 界面
│   │   ├── templates/
│   │   │   └── index.html      # 主页面
│   │   └── static/
│   │       ├── style.css       # 样式
│   │       └── app.js          # 前端逻辑
│   │
│   └── utils/                  # 工具函数
│       ├── io.py               # I/O 操作
│       └── torch_env.py        # PyTorch 环境
│
├── cache/                      # 数据集缓存
├── checkpoints/                # 模型检查点
├── runs/                       # TensorBoard 日志
├── results/                    # 生成样本
└── var/jobs/                   # 任务元数据
```

## 🔌 API 文档

### 系统信息

```http
GET /api/system
```

返回 PyTorch 版本、CUDA 可用性、数据集缓存状态等信息。

### 数据集操作

```http
POST /api/dataset/ensure          # 下载并缓存数据集
GET  /api/dataset/samples?count=16&seed=0  # 获取随机样本
```

### 任务管理

```http
GET  /api/jobs?type=vae&limit=50  # 列出任务
GET  /api/jobs/<job_id>           # 获取任务详情
GET  /api/jobs/<job_id>/log?offset=0  # 获取任务日志
POST /api/jobs/<job_id>/cancel    # 取消任务
```

### 训练接口

```http
POST /api/jobs/vae    # 启动 VAE/VQVAE 训练
POST /api/jobs/dit    # 启动 DiT 训练
POST /api/jobs/sample # 启动图像采样
```

### 检查点管理

```http
GET /api/checkpoints  # 列出可用检查点
```

## 🎓 参考文献

1. **VAE**: Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv preprint arXiv:1312.6114*.

2. **VQVAE**: van den Oord, A., et al. (2017). Neural Discrete Representation Learning. *NeurIPS*.

3. **DDPM**: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.

4. **DiT**: Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers. *ICCV*.

## 📄 License

MIT License

---

<p align="center">
  用 ❤️ 和 PyTorch 构建
</p>
