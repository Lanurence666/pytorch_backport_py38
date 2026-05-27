# PyTorch Backport for Python 3.8

[Русский](README_ru.md) | [中文](README_zh.md) | **English**

A Python 3.8 backport of **PyTorch 2.13.0a0** (latest main branch, commit `03855a7`), enabling modern PyTorch features on the legacy Python 3.8 runtime, with **native FP8 quantization inference support** on CUDA 11.3.

> **Note:** The last official PyTorch version supporting Python 3.8 was **PyTorch 2.0.x**. This backport brings the latest PyTorch features (torch.compile, Transformer improvements, new quantization APIs, native FP8 CUDA kernels, etc.) to Python 3.8 users.

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
- **CUDA 11.3 support** (Windows x64)
- **Native FP8 quantization inference** — custom CUDA kernels for `float8_e4m3fn` and `float8_e5m2` types
- **Flash Attention / Memory-Efficient Attention** — `F.scaled_dot_product_attention` with forward and backward support
- **torch.compile** (Dynamo) — Python-level functionality works; C-level frame evaluation shim is disabled on 3.8
- **Autograd** — fully functional
- **Neural network modules** — fully functional
- **torch.profiler** — functional (with 3.8 compat for `_PyEval_SetProfile`)
- **Quantization** — functional
- **ONNX export** — functional
- **Distributed training** — full Gloo-based distributed support (DDP, DTensor, Tensor Parallel, FSDP, Checkpoint)

## FP8 Quantization Support

This backport includes **native CUDA-level FP8 quantization inference** support, implementing custom kernels for the `float8_e4m3fn` and `float8_e5m2` data types on CUDA 11.3.

### Supported FP8 Operations

| Operation | Implementation | Notes |
|-----------|---------------|-------|
| Tensor creation / casting | FP8↔FP32 conversion | Full support for both e4m3fn and e5m2 |
| Arithmetic (add, sub, mul, div) | FP8→FP32 compute→FP8 | Type promotion to FP32 for computation |
| Matrix multiplication (`mm`, `matmul`) | Custom CUDA kernel with Tiling + Shared Memory | FP8→FP32 GEMM with optimized memory access |
| Batch matrix multiplication (`bmm`) | Custom CUDA Batched GEMM kernel | Tiling, Shared Memory, Register Blocking, Warp-level computation |
| `baddbmm` (batched add+mm) | Custom FP8 Batched GEMM kernel | Supports alpha/beta scaling parameters |
| `conv2d` | im2col + GEMM architecture | FP8 im2col kernel + cuBLAS FP32 GEMM + FP8 output |
| `_scaled_mm` | Scaled FP8 matmul | e4m3fn × e5m2 with scale factors |
| Reduction (sum, max, min, mean) | Standard CUDA kernels | Full FP8 support |
| Comparison (eq, ne, lt, gt) | Element-wise CUDA kernels | Full FP8 support |
| Distribution (uniform_, normal_) | Random generation | Full FP8 support |

### FP8 Kernel Architecture

- **Conv2d**: Custom `fp8_im2col_kernel` (FP8→FP32 columns) → cuBLAS GEMM (FP32) → `float_to_fp8_kernel` (FP32→FP8 output)
- **Baddbmm**: Custom Batched GEMM CUDA kernel with:
  - **Tiling**: 64×64 tile size for optimal memory access
  - **Shared Memory**: Double-buffered tile loading for compute/memory overlap
  - **Register Blocking**: 4×4 output tile per thread
  - **Warp-level computation**: 32-thread cooperative matrix multiply
  - **FP8→FP32 conversion**: Inside kernel for maximum precision
- **mm/matmul**: Same Batched GEMM kernel architecture as baddbmm

### FP8 Test Suite

Run the comprehensive FP8 test suite:

```bash
python test_fp8_inference.py
```

This tests 16 FP8 operations including tensor creation, arithmetic, matmul, conv2d, baddbmm, and more.

## Test Results

Compared to the last official Python 3.8-supported version (PyTorch 2.0.x):

| Feature | PyTorch 2.0.x (Official) | PyTorch 2.13.0a0 (This Backport) |
|---------|--------------------------|----------------------------------|
| Tensor operations | ✅ | ✅ |
| Autograd | ✅ | ✅ |
| CUDA support | ✅ (CUDA 11.x) | ✅ (CUDA 11.3) |
| nn.Module | ✅ | ✅ (more modules available) |
| Optimizers | ✅ | ✅ (more optimizers available) |
| torch.compile | ❌ (not available) | ⚠️ (Python-level only, C shim disabled) |
| Transformer models | ✅ (basic) | ✅ (improved architecture) |
| Quantization | ✅ (basic) | ✅ (new APIs) |
| FP8 quantization | ❌ (not available) | ✅ (native CUDA kernels: conv2d, baddbmm, mm) |
| Flash Attention / SDPA | ❌ (not available) | ✅ (forward + backward) |
| Memory-Efficient Attention | ❌ (not available) | ✅ (forward + backward) |
| torch.profiler | ✅ | ✅ |
| ONNX export | ✅ | ✅ |
| AMP (Mixed Precision) | ✅ | ✅ |
| torch.distributed | ✅ (basic) | ✅ (Gloo: DDP, DTensor, Tensor Parallel, FSDP, Checkpoint) |

## Test Files

### Core Test Suite

A comprehensive test suite is included as `test_pytorch_functions.py`. Run it with:

```bash
python test_pytorch_functions.py
```

The test covers:
- Scaled Dot-Product Attention (Flash Attention)
- Memory-Efficient Attention (forward + backward)
- SDPA with causal masking
- SDPA with different Q/K/V shapes
- SDPA backward pass gradient verification
- Core tensor operations (creation, indexing, math, broadcasting)
- Autograd (gradients, custom functions, gradient checkpointing)
- Neural network modules (Linear, Conv2d, LSTM, Transformer, etc.)
- Loss functions and optimizers
- Model save/load
- CUDA operations (if GPU available)
- Advanced features (torch.compile, JIT, quantization, profiler, AMP)

### Distributed Verification Test

Run the distributed support verification:

```bash
python verify_distributed.py
```

This tests distributed training support and compatibility with transformers and huggingface_hub (online tests require internet connection).

### FP8 Inference Test Suite

Run the FP8-specific test suite:

```bash
python test_fp8_inference.py
```

This tests 16 FP8 operations including:
- FP8 tensor creation and casting (e4m3fn, e5m2)
- FP8 arithmetic operations (add, sub, mul, div)
- FP8 matrix multiplication (mm, matmul)
- FP8 batch matrix multiplication (bmm)
- FP8 baddbmm (batched add+mm with alpha/beta)
- FP8 conv2d (im2col + GEMM architecture)
- FP8 _scaled_mm (scaled matmul with e4m3fn × e5m2)
- FP8 reduction operations (sum, max, min, mean)
- FP8 comparison operations (eq, ne, lt, gt)
- FP8 distribution operations (uniform_, normal_)

## Distributed Training Support

This backport includes **full Gloo-based distributed training support** on Windows, enabling multi-GPU training and model parallelism.

### Supported Distributed Features

| Feature | Status | Notes |
|---------|--------|-------|
| **Gloo backend** | ✅ | TCP-based communication backend for CPU and GPU |
| **NCCL backend** | ❌ | Linux-only; not available on Windows |
| **DistributedDataParallel (DDP)** | ✅ | Single-node multi-GPU data parallelism |
| **DistributedSampler** | ✅ | Data loading for distributed training |
| **TCPStore** | ✅ | Key-value store for distributed coordination |
| **ProcessGroupGloo** | ✅ | Gloo process group backend |
| **DTensor (Distributed Tensor)** | ✅ | Sharded tensor abstraction |
| **Tensor Parallelism** | ✅ | ColwiseParallel, RowwiseParallel |
| **Placement types** | ✅ | Shard, Replicate, Partial |
| **Distributed Checkpoint** | ✅ | save/load for sharded state dicts |
| **FSDP (fully_shard)** | ✅ | Fully Sharded Data Parallelism |
| **Replicate composable** | ✅ | Replication composable utility |
| **torch.distributed.autograd** | ✅ | Distributed autograd support |
| **torch.distributed.rpc** | ⚠️ | Importable but limited (TensorPipe unavailable on Windows) |
| **Pipeline Parallelism** | ❌ | Not available on Windows |

### DDP + AMP Verification

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group("gloo", rank=0, world_size=1)

# Create model and wrap with DDP
model = torch.nn.Linear(64, 10).cuda()
ddp_model = DDP(model)

# Forward pass with mixed precision
with torch.amp.autocast("cuda"):
    output = ddp_model(torch.randn(4, 64, device="cuda"))
```

### Compatibility Test Results

#### Transformers Compatibility

| Test | Result |
|------|--------|
| transformers import | ✅ Pass |
| BertModel forward pass | ✅ Pass |
| BertModel + DDP | ✅ Pass |
| Distributed config test suite (13 tests) | ✅ 13/13 Pass |

#### huggingface_hub Compatibility

| Test | Result |
|------|--------|
| huggingface_hub import | ✅ Pass |
| HfApi (online) | ✅ Pass |
| PyTorchModelHubMixin | ✅ Pass |
| Serialization tests (20 tests) | ✅ 18/20 Pass |
| Cache/validator tests (50 tests) | ✅ 50/50 Pass |

> **Note:** 2 serialization test failures are due to Windows symlink permissions (WinError 1314), not PyTorch issues.

### Distributed Verification Script

Run the comprehensive distributed verification script:

```bash
python verify_distributed.py
```

This tests 26 items across 7 categories:
1. Version information
2. Distributed backend availability (Gloo, NCCL)
3. Distributed module imports (DDP, DTensor, Tensor Parallel, FSDP, Checkpoint)
4. ProcessGroup initialization (Gloo)
5. DDP functional test (forward pass + AMP)
6. Transformers compatibility (import, forward, DDP)
7. huggingface_hub compatibility (import, API, PyTorchModelHubMixin)

Expected output: **24 passed, 0 failed, 2 skipped** (NCCL and PipelineStage skipped on Windows)

## Build Optimizations

This wheel is compiled with **maximum optimization flags** for best runtime performance:

| Component | Flags | Description |
|-----------|-------|-------------|
| C/C++ (MSVC) | `/Ox /Oi /Ot /GS- /Gy /fp:fast` | Maximum optimization, enable intrinsics, favor speed, no security checks, function-level linking, fast floating point |
| CUDA (nvcc) | `-O3 -Xcompiler /Ox /Oi /Ot /GS- /Gy /fp:fast` | Maximum device code optimization, same host compiler flags |
| Linker | `/OPT:REF /OPT:ICF` | Eliminate unreferenced functions, enable COMDAT folding |

**Not included** (to avoid issues): `/GL` (Whole Program Optimization) and `/LTCG` (Link-Time Code Generation) — these can cause static library sizes exceeding 4GB limits.

## How to Build

### Prerequisites

- **Python 3.8** (64-bit, Windows)
- **Visual Studio 2022** with C++20 support
- **CUDA Toolkit 11.3** (for GPU support)
- **CMake** >= 3.25
- **Ninja** build system
- **NumPy** (Python 3.8 compatible version, e.g. `numpy==1.24.4`)

### Build Steps (Editable / Development Mode)

```bash
# 1. Clone this repository
git clone https://github.com/Lanurence666/pytorch_backport_py38.git
cd pytorch_backport_py38

# 2. Create and activate a conda environment
conda create -n py38 python=3.8
conda activate py38

# 3. Install build dependencies
pip install numpy==1.24.4 cmake ninja pybind11 typing_extensions

# 4. Set environment variables
set MAX_JOBS=2
set USE_CUDA=1
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6

# 5. Build and install (editable mode for development)
pip install -e . --no-build-isolation
```

### Build Steps (Wheel Package)

To build a redistributable `.whl` package:

```bash
# 1. Clone and set up environment (same as above steps 1-4)

# 2. Build the wheel
pip wheel --no-build-isolation -w dist .

# 3. The wheel will be in the dist/ directory:
#    dist/torch-2.13.0a0+cu113-cp38-cp38-win_amd64.whl
```

> **Tip:** If you encounter linker memory errors (LNK1102) during the `torch_cpu.dll` phase, reduce `MAX_JOBS` to 1:
> ```bash
> set MAX_JOBS=1
> ```

### Important Build Notes

- Set `MAX_JOBS=2` (or `1` for low-memory systems) to avoid linker memory errors (LNK1102) during the `torch_cpu.dll` linking phase
- The full build takes approximately 2-4 hours on a modern machine
- The resulting wheel is approximately 1.5GB (includes CUDA 11.3 runtime DLLs, Flash Attention, FP8 kernels, distributed support)
- If building without CUDA, set `set USE_CUDA=0` instead

## Installation

### From Wheel (Recommended)

Download the wheel from [GitHub Releases](https://github.com/Lanurence666/pytorch_backport_py38/releases) and install:

```bash
pip install torch-2.13.0a0+cu113-cp38-cp38-win_amd64.whl
```

### From Source

```bash
pip install -e . --no-build-isolation
```

## Known Limitations

1. **torch.compile C shim**: The C-level frame evaluation shim (`_PyInterpreterState_GetEvalFrameFunc`/`SetEvalFrameFunc`) is not available on Python 3.8. Dynamo's Python-level tracing still works, but the C-level performance optimization is disabled.

2. **Windows only**: This backport has only been tested on Windows x64 with CUDA 11.3. Linux builds may require additional fixes.

3. **NCCL not available**: NCCL backend is Linux-only. Use Gloo backend for distributed training on Windows.

4. **RPC limited**: The `torch.distributed.rpc` module is importable but has limited functionality on Windows because TensorPipe (the underlying transport) depends on Unix-specific APIs (Unix Domain Sockets, epoll).

5. **Pipeline Parallelism not available**: `torch.distributed.pipeline` is not supported on Windows.

6. **Python 3.8 end-of-life**: Python 3.8 reached end-of-life in October 2024. Use this backport at your own risk.

## Related Projects

This backport was made possible by the [python38_compat_fix_suite](https://github.com/Lanurence666/python38_compat_fix_suite) — a comprehensive toolset for backporting Python 3.9+ projects to Python 3.8.

Other Python 3.8 backports:
- [numpy_backport_py38](https://github.com/Lanurence666/numpy_backport_py38) — NumPy 2.x for Python 3.8
- [scipy_backport_py38](https://github.com/Lanurence666/scipy_backport_py38) — SciPy 1.x for Python 3.8

## License

PyTorch is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

The compatibility fixes in this backport are also provided under the same BSD 3-Clause License.
