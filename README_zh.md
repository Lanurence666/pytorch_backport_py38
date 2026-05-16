# PyTorch Python 3.8 回移植

[Русский](README_ru.md) | [**中文**](README_zh.md) | [English](README.md)

将 **PyTorch 2.13.0a0**（最新 main 分支，提交 `03855a7`）回移植到 Python 3.8，使旧版 Python 3.8 运行时也能使用现代 PyTorch 特性。

> **注意：** 最后一个官方支持 Python 3.8 的 PyTorch 版本是 **PyTorch 2.0.x**。本回移植将最新的 PyTorch 特性（torch.compile、Transformer 改进、新的量化 API 等）带给 Python 3.8 用户。

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
- **CUDA 12.4 支持**（Windows x64）
- **torch.compile**（Dynamo）— Python 层面功能正常；C 层面帧评估 shim 在 3.8 上已禁用
- **Autograd** — 完全可用
- **神经网络模块** — 完全可用
- **torch.profiler** — 可用（包含 3.8 兼容的 `_PyEval_SetProfile`）
- **量化** — 可用
- **ONNX 导出** — 可用
- **分布式训练** — 基本功能可用

## 测试结果

与最后一个官方支持 Python 3.8 的版本（PyTorch 2.0.x）对比：

| 功能 | PyTorch 2.0.x（官方） | PyTorch 2.13.0a0（本回移植） |
|------|---------------------|---------------------------|
| 张量操作 | ✅ | ✅ |
| Autograd | ✅ | ✅ |
| CUDA 支持 | ✅（CUDA 11.x） | ✅（CUDA 12.4） |
| nn.Module | ✅ | ✅ |
| 优化器 | ✅ | ✅（更多优化器可用） |
| torch.compile | ❌（不可用） | ⚠️（仅 Python 层面，C shim 已禁用） |
| Transformer 模型 | ✅（基础） | ✅（改进的架构） |
| 量化 | ✅（基础） | ✅（新 API） |
| torch.profiler | ✅ | ✅ |
| ONNX 导出 | ✅ | ✅ |
| AMP（混合精度） | ✅ | ✅ |
| torch.distributed | ✅（基础） | ✅（改进） |

## 测试文件

综合测试套件包含在 `test_pytorch_functions.py` 中。运行方式：

```bash
python test_pytorch_functions.py
```

测试覆盖：
- 核心张量操作（创建、索引、数学、广播）
- Autograd（梯度、自定义函数、梯度检查点）
- 神经网络模块（Linear、Conv2d、LSTM、Transformer 等）
- 损失函数和优化器
- 模型保存/加载
- CUDA 操作（如有 GPU）
- 高级功能（torch.compile、JIT、量化、分析器、AMP）

## 如何编译

### 前提条件

- **Python 3.8**（64 位，Windows）
- **Visual Studio 2022**，支持 C++20
- **CUDA Toolkit 12.4**（用于 GPU 支持）
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
set TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0

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
#    dist/torch-2.13.0a0+git03855a7-cp38-cp38-win_amd64.whl
```

> **提示：** 如果在 `torch_cpu.dll` 阶段遇到链接器内存错误（LNK1102），将 `MAX_JOBS` 减少到 1：
> ```bash
> set MAX_JOBS=1
> ```

### 重要编译说明

- 设置 `MAX_JOBS=2`（低内存系统设为 `1`）以避免 `torch_cpu.dll` 链接阶段的链接器内存错误（LNK1102）
- 完整编译在现代机器上大约需要 2-4 小时
- 生成的 wheel 包大约 160MB
- 如果不使用 CUDA 编译，设置 `set USE_CUDA=0`

## 安装

### 从 Wheel 安装（推荐）

从 [GitHub Releases](https://github.com/Lanurence666/pytorch_backport_py38/releases) 下载 wheel 并安装：

```bash
pip install torch-2.13.0a0+git03855a7-cp38-cp38-win_amd64.whl
```

### 从源码安装

```bash
pip install -e . --no-build-isolation
```

## 已知限制

1. **torch.compile C shim**：C 层面帧评估 shim（`_PyInterpreterState_GetEvalFrameFunc`/`SetEvalFrameFunc`）在 Python 3.8 上不可用。Dynamo 的 Python 层面追踪仍然有效，但 C 层面的性能优化已禁用。

2. **仅限 Windows**：此回移植仅在 Windows x64 + CUDA 12.4 上测试过。Linux 构建可能需要额外修复。

3. **Python 3.8 已停止维护**：Python 3.8 于 2024 年 10 月停止维护。使用此回移植需自行承担风险。

## 相关项目

此回移植得益于 [python38_compat_fix_suite](https://github.com/Lanurence666/python38_compat_fix_suite) — 一套用于将 Python 3.9+ 项目回移植到 Python 3.8 的综合工具集。

其他 Python 3.8 回移植：
- [numpy_backport_py38](https://github.com/Lanurence666/numpy_backport_py38) — NumPy 2.x for Python 3.8
- [scipy_backport_py38](https://github.com/Lanurence666/scipy_backport_py38) — SciPy 1.x for Python 3.8

## 许可证

PyTorch 采用 BSD 3-Clause 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

此回移植中的兼容性修复同样采用 BSD 3-Clause 许可证。
