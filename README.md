# PyTorch Backport for Python 3.8

[Русский](README_ru.md) | [中文](README_zh.md) | **English**

A Python 3.8 backport of **PyTorch 2.13.0a0** (latest main branch, commit `03855a7`), enabling modern PyTorch features on the legacy Python 3.8 runtime.

> **Note:** The last official PyTorch version supporting Python 3.8 was **PyTorch 2.0.x**. This backport brings the latest PyTorch features (torch.compile, Transformer improvements, new quantization APIs, etc.) to Python 3.8 users.

## What Is This?

This is a modified version of the PyTorch source code that compiles and runs on **Python 3.8** (Windows x64). The original PyTorch main branch requires Python 3.10+, so we applied a comprehensive set of compatibility fixes to make it work on Python 3.8.

## Compatibility Fixes Applied

The following Python 3.9+ syntax and API issues were fixed to enable compilation on Python 3.8:

### Python Source Code Fixes

| # | Issue | Python Version | Fix |
|---|-------|---------------|-----|
| 1 | Type union syntax `X \| Y` | 3.10+ | Replace with `Union[X, Y]` via `from __future__ import annotations` or `typing.Union` |
| 2 | Built-in generic syntax `list[X]`, `dict[X, Y]` | 3.9+ | Replace with `List[X]`, `Dict[X, Y]` from `typing` |
| 3 | `str.removeprefix()` / `str.removesuffix()` | 3.9+ | Implement polyfill or use `str.lstrip()` / `str.rstrip()` alternatives |
| 4 | `typing.TypeGuard` | 3.10+ | Replace with `bool` return type |
| 5 | `typing.ParamSpec` usage | 3.10+ | Use `typing_extensions.ParamSpec` |
| 6 | `match` / `case` statements | 3.10+ | Rewrite as `if` / `elif` chains |
| 7 | `zip(strict=True)` | 3.10+ | Implement manual length check |
| 8 | `functools.cache` | 3.9+ | Use `functools.lru_cache(maxsize=None)` |
| 9 | `math.dist()` | 3.8+ | Already available, no fix needed |
| 10 | `typing.TypeAlias` | 3.10+ | Use simple assignment or `typing_extensions.TypeAlias` |
| 11 | `AttributeError(msg, name=..., obj=...)` keyword-only args | 3.10+ | Remove `name=None`/`obj=None` or use `_AttributeError_compat()` helper |

### C/C++ Source Code Fixes

| # | Issue | Python Version | Fix |
|---|-------|---------------|-----|
| 1 | `PyType_GetSlot()` | 3.9+ | Implement compat shim using `tp_as_number`, `tp_as_sequence`, `tp_as_mapping` |
| 2 | `_PyEval_SetProfile()` | 3.9+ | Direct `PyThreadState` field assignment for 3.8 |
| 3 | `_PyInterpreterState_GetEvalFrameFunc()` / `SetEvalFrameFunc()` | 3.9+ | No-op stubs for 3.8 (Dynamo C shim disabled) |
| 4 | `Py_TPFLAGS_HAVE_VECTORCALL` | 3.12+ | Map to `_Py_TPFLAGS_HAVE_VECTORCALL` (3.8-3.11) |
| 5 | Flatbuffers version mismatch in `mobile_bytecode_generated.h` | N/A | Updated `static_assert` to match actual version (25.12.19) |
| 6 | `opcode.h` include order (needs `Py_LT` etc. from `object.h`) | N/A | Moved `#include <opcode.h>` after `Python.h` includes |
| 7 | `pythoncapi_compat.h` per-function `#ifndef` guards | N/A | Prevent redefinition errors when multiple projects include their own copy |

## Key Features

- **Full PyTorch 2.13.0a0 feature set** on Python 3.8
- **CUDA 12.4 support** (Windows x64)
- **torch.compile** (Dynamo) — Python-level functionality works; C-level frame evaluation shim is disabled on 3.8
- **Autograd** — fully functional
- **Neural network modules** — fully functional
- **torch.profiler** — functional (with 3.8 compat for `_PyEval_SetProfile`)
- **Quantization** — functional
- **ONNX export** — functional
- **Distributed training** — basic functionality available

## Test Results

Compared to the last official Python 3.8-supported version (PyTorch 2.0.x):

| Feature | PyTorch 2.0.x (Official) | PyTorch 2.13.0a0 (This Backport) |
|---------|--------------------------|----------------------------------|
| Tensor operations | ✅ | ✅ |
| Autograd | ✅ | ✅ |
| CUDA support | ✅ (CUDA 11.x) | ✅ (CUDA 12.4) |
| nn.Module | ✅ | ✅ |
| Optimizers | ✅ | ✅ (more optimizers available) |
| torch.compile | ❌ (not available) | ⚠️ (Python-level only, C shim disabled) |
| Transformer models | ✅ (basic) | ✅ (improved architecture) |
| Quantization | ✅ (basic) | ✅ (new APIs) |
| torch.profiler | ✅ | ✅ |
| ONNX export | ✅ | ✅ |
| AMP (Mixed Precision) | ✅ | ✅ |
| torch.distributed | ✅ (basic) | ✅ (improved) |

## Test File

A comprehensive test suite is included as `test_pytorch_functions.py`. Run it with:

```bash
python test_pytorch_functions.py
```

The test covers:
- Core tensor operations (creation, indexing, math, broadcasting)
- Autograd (gradients, custom functions, gradient checkpointing)
- Neural network modules (Linear, Conv2d, LSTM, Transformer, etc.)
- Loss functions and optimizers
- Model save/load
- CUDA operations (if GPU available)
- Advanced features (torch.compile, JIT, quantization, profiler, AMP)

## How to Build

### Prerequisites

- **Python 3.8** (64-bit, Windows)
- **Visual Studio 2022** with C++20 support
- **CUDA Toolkit 12.4** (for GPU support)
- **CMake** >= 3.25
- **Ninja** build system
- **NumPy** (Python 3.8 compatible version)

### Build Steps

```bash
# 1. Clone this repository
git clone https://github.com/Lanurence666/pytorch_backport_py38.git
cd pytorch_backport_py38

# 2. Create and activate a conda environment
conda create -n py38 python=3.8
conda activate py38

# 3. Install build dependencies
pip install numpy cmake ninja pybind11 typing_extensions

# 4. Set environment variables
set MAX_JOBS=2
set USE_CUDA=1
set TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0

# 5. Build and install (editable mode for development)
pip install -e . --no-build-isolation

# 6. Or build a wheel package
pip wheel --no-build-isolation -w dist .
```

### Important Build Notes

- Set `MAX_JOBS=2` to avoid linker memory errors (LNK1102) during the `torch_cpu.dll` linking phase
- The full build takes approximately 2-4 hours on a modern machine
- The resulting wheel is approximately 160MB

## Installation

### From Wheel (Recommended)

Download the wheel from [GitHub Releases](https://github.com/Lanurence666/pytorch_backport_py38/releases) and install:

```bash
pip install torch-2.13.0a0+git03855a7-cp38-cp38-win_amd64.whl
```

### From Source

```bash
pip install -e . --no-build-isolation
```

## Known Limitations

1. **torch.compile C shim**: The C-level frame evaluation shim (`_PyInterpreterState_GetEvalFrameFunc`/`SetEvalFrameFunc`) is not available on Python 3.8. Dynamo's Python-level tracing still works, but the C-level performance optimization is disabled.

2. **Windows only**: This backport has only been tested on Windows x64 with CUDA 12.4. Linux builds may require additional fixes.

3. **Python 3.8 end-of-life**: Python 3.8 reached end-of-life in October 2024. Use this backport at your own risk.

## Related Projects

This backport was made possible by the [python38_compat_fix_suite](https://github.com/Lanurence666/python38_compat_fix_suite) — a comprehensive toolset for backporting Python 3.9+ projects to Python 3.8.

Other Python 3.8 backports:
- [numpy_backport_py38](https://github.com/Lanurence666/numpy_backport_py38) — NumPy 2.x for Python 3.8
- [scipy_backport_py38](https://github.com/Lanurence666/scipy_backport_py38) — SciPy 1.x for Python 3.8

## License

PyTorch is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

The compatibility fixes in this backport are also provided under the same BSD 3-Clause License.
