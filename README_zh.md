# PyTorch Python 3.8 回移植

[Русский](README_ru.md) | [**中文**](README_zh.md) | [English](README.md)

将 **PyTorch 2.13.0a0**（最新 main 分支，提交 `0607d0e`）回移植到 Python 3.8，使旧版 Python 3.8 运行时也能使用现代 PyTorch 特性，并在 CUDA 11.3 上支持 **原生 FP8 量化推理** 和 **完整的 Gloo 分布式训练**。

> **注意：** 最后一个官方支持 Python 3.8 的 PyTorch 版本是 **PyTorch 2.0.x**。本回移植将最新的 PyTorch 特性（torch.compile、Transformer 改进、新的量化 API、原生 FP8 CUDA 内核、分布式训练等）带给 Python 3.8 用户。

## 这是什么？

这是修改后的 PyTorch 源代码，可以在 **Python 3.8**（Windows x64）上编译和运行。原始 PyTorch main 分支需要 Python 3.10+，因此我们应用了一套全面的兼容性修复，使其能在 Python 3.8 上工作。

## 应用的兼容性修复

以下 Python 3.9+ 语法和 API 问题已修复，以支持在 Python 3.8 上编译：

### Python 源代码修复

| # | 问题 | Python 版本 | 修复方式 |
|---|------|------------|---------|
| 1 | 类型联合语法 `X \| Y` | 3.10+ | 替换为 `Union[X, Y]`，通过 `from __future__ import annotations` 或 `typing.Union` |
| 2 | 内置泛型语法 `list[X]`、`dict[X, Y]` | 3.9+ | 替换为 `typing` 中的 `List[X]`、`Dict[X, Y]` |
| 3 | `str.removeprefix()` / `str.removesuffix()` | 3.9+ | 实现 polyfill 或使用替代方案 |
| 4 | `typing.TypeGuard` | 3.10+ | 替换为 `bool` 返回类型 |
| 5 | `typing.ParamSpec` 用法 | 3.10+ | 使用 `typing_extensions.ParamSpec` |
| 6 | `match` / `case` 语句 | 3.10+ | 重写为 `if` / `elif` 链 |
| 7 | `zip(strict=True)` | 3.10+ | 实现手动长度检查 |
| 8 | `functools.cache` | 3.9+ | 使用 `functools.lru_cache(maxsize=None)` |
| 9 | `typing.TypeAlias` | 3.10+ | 使用简单赋值或 `typing_extensions.TypeAlias` |
| 10 | `AttributeError(msg, name=..., obj=...)` 关键字参数 | 3.10+ | 移除 `name=None`/`obj=None` 或使用 `_AttributeError_compat()` 辅助函数 |

### C/C++ 源代码修复

| # | 问题 | Python 版本 | 修复方式 |
|---|------|------------|---------|
| 1 | `PyType_GetSlot()` | 3.9+ | 通过 `tp_as_number`、`tp_as_sequence`、`tp_as_mapping` 实现兼容性封装 |
| 2 | `_PyEval_SetProfile()` | 3.9+ | 在 3.8 中直接赋值 `PyThreadState` 字段 |
| 3 | `_PyInterpreterState_GetEvalFrameFunc()` / `SetEvalFrameFunc()` | 3.9+ | 在 3.8 中使用 no-op 存根（Dynamo C shim 已禁用） |
| 4 | `Py_TPFLAGS_HAVE_VECTORCALL` | 3.12+ | 映射到 `_Py_TPFLAGS_HAVE_VECTORCALL`（3.8-3.11） |
| 5 | `mobile_bytecode_generated.h` 中 Flatbuffers 版本不匹配 | 不适用 | 更新 `static_assert` 以匹配实际版本（25.12.19） |
| 6 | `opcode.h` 包含顺序（需要 `object.h` 中的 `Py_LT` 等） | 不适用 | 将 `#include <opcode.h>` 移至 `Python.h` 包含之后 |
| 7 | `pythoncapi_compat.h` 逐函数 `#ifndef` 保护 | 不适用 | 防止多个项目各自包含副本时的重定义错误 |

## 主要特性

- **完整的 PyTorch 2.13.0a0 功能集**在 Python 3.8 上运行
- **CUDA 11.3 支持**（Windows x64）
- **原生 FP8 量化推理** — `float8_e4m3fn` 和 `float8_e5m2` 类型的自定义 CUDA 内核
- **Flash Attention / 内存高效注意力** — `F.scaled_dot_product_attention` 支持前向和反向
- **torch.compile**（Dynamo）— Python 层面功能正常；C 层面帧评估 shim 在 3.8 上已禁用
- **Autograd** — 完全可用
- **神经网络模块** — 完全可用
- **torch.profiler** — 可用（包含 3.8 兼容的 `_PyEval_SetProfile`）
- **量化** — 可用
- **ONNX 导出** — 可用
- **分布式训练** — 完整的 Gloo 分布式支持（DDP、DTensor、张量并行、FSDP、检查点）

## FP8 量化支持

本回移植包含 **原生 CUDA 级别的 FP8 量化推理** 支持，在 CUDA 11.3 上实现了 `float8_e4m3fn` 和 `float8_e5m2` 数据类型的自定义内核。

### 支持的 FP8 操作

| 操作 | 实现 | 说明 |
|------|------|------|
| 张量创建/转换 | FP8↔FP32 转换 | 完整支持 e4m3fn 和 e5m2 |
| 算术运算（add, sub, mul, div） | FP8→FP32 计算→FP8 | 类型提升到 FP32 进行计算 |
| 矩阵乘法（`mm`、`matmul`） | 自定义 CUDA 内核（Tiling + 共享内存） | FP8→FP32 GEMM，优化内存访问 |
| 批量矩阵乘法（`bmm`） | 自定义 CUDA 批量 GEMM 内核 | Tiling、共享内存、寄存器分块、Warp 级计算 |
| `baddbmm`（批量 add+mm） | 自定义 FP8 批量 GEMM 内核 | 支持 alpha/beta 缩放参数 |
| `conv2d` | im2col + GEMM 架构 | FP8 im2col 内核 + cuBLAS FP32 GEMM + FP8 输出 |
| `_scaled_mm` | 缩放 FP8 矩阵乘法 | e4m3fn × e5m2 带缩放因子 |
| 规约（sum, max, min, mean） | 标准 CUDA 内核 | 完整 FP8 支持 |
| 比较（eq, ne, lt, gt） | 逐元素 CUDA 内核 | 完整 FP8 支持 |
| 分布（uniform_, normal_） | 随机生成 | 完整 FP8 支持 |

## 分布式训练支持

本回移植在 Windows 上包含 **完整的基于 Gloo 的分布式训练支持**，支持多 GPU 训练和模型并行。

### 支持的分布式特性

| 特性 | 状态 | 说明 |
|------|------|------|
| **Gloo 后端** | ✅ | 基于 TCP 的 CPU 和 GPU 通信后端 |
| **NCCL 后端** | ❌ | 仅限 Linux；Windows 上不可用 |
| **DistributedDataParallel (DDP)** | ✅ | 单节点多 GPU 数据并行 |
| **DistributedSampler** | ✅ | 分布式训练的数据加载 |
| **TCPStore** | ✅ | 分布式协调的键值存储 |
| **ProcessGroupGloo** | ✅ | Gloo 进程组后端 |
| **DTensor（分布式张量）** | ✅ | 分片张量抽象 |
| **张量并行** | ✅ | ColwiseParallel、RowwiseParallel |
| **放置类型** | ✅ | Shard、Replicate、Partial |
| **分布式检查点** | ✅ | 分片状态字典的 save/load |
| **FSDP (fully_shard)** | ✅ | 完全分片数据并行 |
| **Replicate composable** | ✅ | 复制组合工具 |
| **torch.distributed.autograd** | ✅ | 分布式自动微分支持 |
| **torch.distributed.rpc** | ⚠️ | 可导入但功能受限（TensorPipe 在 Windows 上不可用） |
| **流水线并行** | ❌ | Windows 上不可用 |

### DDP + AMP 验证

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group("gloo", rank=0, world_size=1)

# 创建模型并用 DDP 包装
model = torch.nn.Linear(64, 10).cuda()
ddp_model = DDP(model)

# 混合精度前向传播
with torch.amp.autocast("cuda"):
    output = ddp_model(torch.randn(4, 64, device="cuda"))
```

### 兼容性测试结果

#### Transformers 兼容性

| 测试 | 结果 |
|------|------|
| transformers 导入 | ✅ 通过 |
| BertModel 前向传播 | ✅ 通过 |
| BertModel + DDP | ✅ 通过 |
| 分布式配置测试套件（13 项） | ✅ 13/13 通过 |

#### huggingface_hub 兼容性

| 测试 | 结果 |
|------|------|
| huggingface_hub 导入 | ✅ 通过 |
| HfApi（在线） | ✅ 通过 |
| PyTorchModelHubMixin | ✅ 通过 |
| 序列化测试（20 项） | ✅ 18/20 通过 |
| 缓存/验证器测试（50 项） | ✅ 50/50 通过 |

> **注意：** 2 个序列化测试失败是由于 Windows 符号链接权限问题（WinError 1314），与 PyTorch 无关。

### 分布式验证脚本

运行综合分布式验证脚本：

```bash
python verify_distributed.py
```

测试 7 个类别共 26 项：
1. 版本信息
2. 分布式后端可用性（Gloo、NCCL）
3. 分布式模块导入（DDP、DTensor、张量并行、FSDP、检查点）
4. ProcessGroup 初始化（Gloo）
5. DDP 功能测试（前向传播 + AMP）
6. Transformers 兼容性（导入、前向、DDP）
7. huggingface_hub 兼容性（导入、API、PyTorchModelHubMixin）

预期输出：**24 通过，0 失败，2 跳过**（NCCL 和 PipelineStage 在 Windows 上跳过）

## 测试结果

与最后一个官方支持 Python 3.8 的版本（PyTorch 2.0.x）对比：

| 功能 | PyTorch 2.0.x（官方） | PyTorch 2.13.0a0（本回移植） |
|------|---------------------|---------------------------|
| 张量操作 | ✅ | ✅ |
| Autograd | ✅ | ✅ |
| CUDA 支持 | ✅（CUDA 11.x） | ✅（CUDA 11.3） |
| nn.Module | ✅ | ✅（更多模块可用） |
| 优化器 | ✅ | ✅（更多优化器可用） |
| torch.compile | ❌（不可用） | ⚠️（仅 Python 层面，C shim 已禁用） |
| Transformer 模型 | ✅（基础） | ✅（改进的架构） |
| 量化 | ✅（基础） | ✅（新 API） |
| FP8 量化 | ❌（不可用） | ✅（原生 CUDA 内核：conv2d、baddbmm、mm） |
| Flash Attention / SDPA | ❌（不可用） | ✅（前向 + 反向） |
| 内存高效注意力 | ❌（不可用） | ✅（前向 + 反向） |
| torch.profiler | ✅ | ✅ |
| ONNX 导出 | ✅ | ✅ |
| AMP（混合精度） | ✅ | ✅ |
| torch.distributed | ✅（基础） | ✅（Gloo: DDP、DTensor、张量并行、FSDP、检查点） |

## 测试文件

### 核心测试套件

综合测试套件包含在 `test_pytorch_functions.py` 中。运行方式：

```bash
python test_pytorch_functions.py
```

测试覆盖：
- 缩放点积注意力（Flash Attention）
- 内存高效注意力（前向 + 反向）
- 带因果掩码的 SDPA
- 不同 Q/K/V 形状的 SDPA
- SDPA 反向传播梯度验证
- 核心张量操作（创建、索引、数学、广播）
- Autograd（梯度、自定义函数、梯度检查点）
- 神经网络模块（Linear、Conv2d、LSTM、Transformer 等）
- 损失函数和优化器
- 模型保存/加载
- CUDA 操作（如有 GPU）
- 高级功能（torch.compile、JIT、量化、分析器、AMP）

### 分布式验证测试

运行分布式支持验证：

```bash
python verify_distributed.py
```

测试分布式训练支持以及与 transformers 和 huggingface_hub 的兼容性（在线测试需要网络连接）。

### FP8 推理测试套件

运行 FP8 专用测试套件：

```bash
python test_fp8_inference.py
```

测试 16 个 FP8 操作，包括：
- FP8 张量创建和转换（e4m3fn、e5m2）
- FP8 算术运算（add、sub、mul、div）
- FP8 矩阵乘法（mm、matmul）
- FP8 批量矩阵乘法（bmm）
- FP8 baddbmm（批量 add+mm 带 alpha/beta）
- FP8 conv2d（im2col + GEMM 架构）
- FP8 _scaled_mm（缩放矩阵乘法 e4m3fn × e5m2）
- FP8 规约操作（sum、max、min、mean）
- FP8 比较操作（eq、ne、lt、gt）
- FP8 分布操作（uniform_、normal_）

## 编译优化

此 wheel 使用 **最大优化标志** 编译，以获得最佳运行时性能：

| 组件 | 标志 | 说明 |
|------|------|------|
| C/C++ (MSVC) | `/Ox /Oi /Ot /GS- /Gy /fp:fast` | 最大优化、启用内联、倾向速度、无安全检查、函数级链接、快速浮点 |
| CUDA (nvcc) | `-O3 -Xcompiler /Ox /Oi /Ot /GS- /Gy /fp:fast` | 最大设备代码优化、相同主机编译器标志 |
| 链接器 | `/OPT:REF /OPT:ICF` | 消除未引用函数、启用 COMDAT 折叠 |

**未包含**（避免问题）：`/GL`（全程序优化）和 `/LTCG`（链接时代码生成）— 可能导致静态库大小超过 4GB 限制。

## 如何编译

### 前提条件

- **Python 3.8**（64 位，Windows）
- **Visual Studio 2022**，支持 C++20
- **CUDA Toolkit 11.3**（用于 GPU 支持）
- **CMake** >= 3.25
- **Ninja** 构建系统
- **NumPy**（兼容 Python 3.8 的版本，如 `numpy==1.24.4`）

### 编译步骤（开发模式）

```bash
# 1. 克隆此仓库
git clone https://github.com/Lanurence666/pytorch_backport_py38.git
cd pytorch_backport_py38

# 2. 创建并激活 conda 环境
conda create -n py38 python=3.8
conda activate py38

# 3. 安装编译依赖
pip install numpy==1.24.4 cmake ninja pybind11 typing_extensions

# 4. 设置环境变量
set MAX_JOBS=2
set USE_CUDA=1
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6

# 5. 编译并安装（开发模式）
pip install -e . --no-build-isolation
```

### 编译步骤（Wheel 包）

构建可分发的 `.whl` 包：

```bash
# 1. 克隆并设置环境（同上步骤 1-4）

# 2. 构建 wheel
pip wheel --no-build-isolation -w dist .

# 3. wheel 将在 dist/ 目录中：
#    dist/torch-2.13.0a0+cu113-cp38-cp38-win_amd64.whl
```

> **提示：** 如果在 `torch_cpu.dll` 阶段遇到链接器内存错误（LNK1102），将 `MAX_JOBS` 减少到 1：
> ```bash
> set MAX_JOBS=1
> ```

### 重要编译说明

- 设置 `MAX_JOBS=2`（低内存系统设为 `1`）以避免 `torch_cpu.dll` 链接阶段的链接器内存错误（LNK1102）
- 完整编译在现代机器上大约需要 2-4 小时
- 生成的 wheel 包大约 1.5GB（包含 CUDA 11.3 运行时 DLL、Flash Attention、FP8 内核、分布式支持）
- 如果不使用 CUDA 编译，设置 `set USE_CUDA=0`

## 安装

### 从 Wheel 安装（推荐）

从 [GitHub Releases](https://github.com/Lanurence666/pytorch_backport_py38/releases) 下载 wheel 并安装：

```bash
pip install torch-2.13.0a0+cu113-cp38-cp38-win_amd64.whl
```

### 从源码安装

```bash
pip install -e . --no-build-isolation
```

## 已知限制

1. **torch.compile C shim**：C 层面帧评估 shim（`_PyInterpreterState_GetEvalFrameFunc`/`SetEvalFrameFunc`）在 Python 3.8 上不可用。Dynamo 的 Python 层面追踪仍然有效，但 C 层面的性能优化已禁用。

2. **仅限 Windows**：此回移植仅在 Windows x64 + CUDA 11.3 上测试过。Linux 构建可能需要额外修复。

3. **NCCL 不可用**：NCCL 后端仅限 Linux。在 Windows 上请使用 Gloo 后端进行分布式训练。

4. **RPC 功能受限**：`torch.distributed.rpc` 模块可导入但在 Windows 上功能有限，因为 TensorPipe（底层传输）依赖 Unix 特定 API（Unix Domain Sockets、epoll）。

5. **流水线并行不可用**：`torch.distributed.pipeline` 在 Windows 上不支持。

6. **Python 3.8 已停止维护**：Python 3.8 于 2024 年 10 月停止维护。使用此回移植需自行承担风险。

## 相关项目

此回移植得益于 [python38_compat_fix_suite](https://github.com/Lanurence666/python38_compat_fix_suite) — 一套用于将 Python 3.9+ 项目回移植到 Python 3.8 的综合工具集。

其他 Python 3.8 回移植：
- [numpy_backport_py38](https://github.com/Lanurence666/numpy_backport_py38) — NumPy 2.x for Python 3.8
- [scipy_backport_py38](https://github.com/Lanurence666/scipy_backport_py38) — SciPy 1.x for Python 3.8

## 许可证

PyTorch 采用 BSD 3-Clause 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

此回移植中的兼容性修复同样采用 BSD 3-Clause 许可证。
