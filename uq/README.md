# 不确定性量化分析

## 环境配置

### 一、前提条件 / 环境准备

在安装 GPU 版本的 PyTorch 之前，确保以下几点：

1. **有 NVIDIA CUDA 支持的 GPU**
   你的显卡必须是 NVIDIA 的，且支持 CUDA（Compute Capability）。

2. **安装 NVIDIA 驱动 + CUDA Toolkit / cuDNN**

   * 操作系统中必须安装与 GPU 匹配的 NVIDIA 驱动。
   * 虽然 PyTorch 的官方二进制包很多情况下自带所需的 CUDA / cuDNN 库（即“内嵌 CUDA”），但仍需确保驱动版本足够新以支持这些库。
   * 如果你自己需要用 CUDA 编译或者扩展，需手动安装 CUDA Toolkit + cuDNN。

3. **Python 环境（建议隔离环境）**
   推荐使用 Conda 或 venv/virtualenv 来创建干净的 Python 环境，避免与系统或其他包冲突。

4. **查看 PyTorch 官方支持矩阵**
   在 PyTorch 官网 “Get Started → Locally” 页面，可以选择你的 OS、CUDA 版本、包管理工具 (conda / pip) 等，得到推荐的安装命令。([PyTorch][1])

---

## 二、安装方式（conda / pip）

下面是两种常见方式的说明与示例命令：

### 方法 1：使用 **conda**（推荐，兼容性更好）

如果你用的是 Anaconda / Miniconda：

```bash
# 创建一个新的环境（可选）
conda create -n pytorch-gpu python=3.10
conda activate pytorch-gpu

# 安装带 GPU 支持的 PyTorch
# 这里以 CUDA 11.8 为例
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

解释：

* `pytorch`, `torchvision`, `torchaudio` 是基础包
* `pytorch-cuda=11.8` 指定 PyTorch 要使用 CUDA 11.8 的版本
* `-c pytorch -c nvidia` 是指定从 PyTorch 官方和 NVIDIA 通道获取包

> 如果你选择其他 CUDA 版本（如 12.x），在官网上选择对应版本，会给出相应命令。([PyTorch][1])

### 方法 2：使用 **pip**

如果你更喜欢用 pip（或不使用 conda）：

```bash
# 假设选择 CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

这里 `cu118` 表示 CUDA 11.8 的版本。官网上不同 CUDA 的 pip 包路径不同。([PyTorch][1])

> 注意，在 Windows 上仅用 `pip install torch` 并不一定会安装带 CUDA 的版本，要确保选择的是带 CUDA 从属包。([PyTorch Forums][2])

---

## 三、验证安装是否成功（GPU 是否可用）

安装完成后，可以在 Python 中运行以下代码检查 PyTorch 是否识别 GPU：

```python
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
```

* 如果 `torch.cuda.is_available()` 返回 `True`，说明 GPU 可用。
* 你还可以做一个小张量测试：

```python
x = torch.randn(3, 3).cuda()
y = torch.randn(3, 3).cuda()
print(x + y)
```

如果无报错且能输出张量，则表明基本功能正常。

---

## 四、常见问题 & 版本兼容注意

1. **CUDA 版本兼容问题**
   如果你系统里装的 CUDA 版本和 PyTorch 二进制包里内置的版本冲突，可能导致不能识别 GPU 或出现错误。建议按官网给出的 CUDA 版本来安装。([PyTorch Forums][3])

2. **驱动版本太旧**
   即使你的 PyTorch 包里自带 CUDA 库，也要求系统 NVIDIA 驱动版本高于某个最小版本。驱动太旧可能无法启动 GPU。([telin.ugent.be][4])

3. **Python 版本 / 包冲突**
   有时 Python 版本过新或与其它包有冲突，可能导致安装失败。建议使用稳定版本，比如 Python 3.9 / 3.10 / 3.11。

4. **卸载旧版本**
   如果你之前装过 CPU 版本或错误的 GPU 版本，最好先卸载 `torch`, `torchvision` 等包再安装。

5. **夜间构建 / Preview 版本**
   如果你需要最新特性，可以选择 nightly 或 preview 版本。但这些可能不稳定。官网提供选择。([PyTorch][5])