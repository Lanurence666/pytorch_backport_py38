# PyTorch Backport for Python 3.8

[Русский](README_ru.md) | [中文](README_zh.md) | **English**

A Python 3.8 backport of **PyTorch 2.13.0a0** (latest main branch, commit `03855a7`), enabling modern PyTorch features on the legacy Python 3.8 runtime.

> **Note:** The last official PyTorch version supporting Python 3.8 was **PyTorch 2.0.x**. This backport brings the latest PyTorch features (torch.compile, Transformer improvements, new quantization APIs, etc.) to Python 3.8 users.

## What Is This?

This is a modified version of the PyTorch source code that compiles and runs on **Python 3.8** (Windows x64). The original PyTorch main branch requires Python 3.10+, so we applied a comprehensive set of compatibility fixes to make it work on Python 3.8.

## What We Modified / Fixed

### Python Source Code Fixes

Using our [Python 3.8 Compatibility Fix Suite](https://github.com/Lanurence666/python38_compat_fix_suite), we automatically fixed the following categories of issues:

| # | Issue | Python Version | Fix Applied |
|---|-------|----------------|-------------|
| 1 | Built-in generics (`list[X]`, `dict[K,V]`) | 3.9+ | Replaced with `typing.List[X]`, `typing.Dict[K,V]` |
| 2 | Union types (`X \| Y` in annotations) | 3.10+ | Replaced with `Union[X, Y]` |
| 3 | Dictionary merge operators (`d1 \| d2`) | 3.9+ | Replaced with `{**d1, **d2}` |
| 4 | `str.removeprefix()` / `str.removesuffix()` | 3.9+ | Fallback implementation |
| 5 | `functools.cache` | 3.9+ | Replaced with `lru_cache(maxsize=None)` |
| 6 | `importlib.metadata` | 3.9+ | Fallback to `importlib_metadata` |
| 7 | `typing.TypeAlias` / `ParamSpec` / `Concatenate` | 3.9+ | Fallback to `typing_extensions` |
| 8 | `isinstance(x, A \| B)` | 3.10+ | Replaced with `isinstance(x, (A, B))` |
| 9 | `zoneinfo` | 3.9+ | Fallback to `backports.zoneinfo` |
| 10 | `math.lcm()` / `math.nextafter()` / `math.ulp()` | 3.9+ | Fallback implementations |
| 11 | `AttributeError(msg, name=..., obj=...)` | 3.10+ | Removed keyword-only args |
| 12 | `zip(..., strict=True)` | 3.10+ | `_zip_strict()` fallback |
| 13 | `int.bit_count()` | 3.10+ | Fallback implementation |
| 14 | `dataclass(slots=True)` | 3.10+ | Removed `slots` parameter |
| 15 | `collections.abc.Callable[...]` subscripting | 3.9+ | Replaced with `typing.Callable[...]` |
| 16 | `from __future__ import annotations` insertion | — | Added where needed for `X \| Y` syntax |
| 17 | PEP 585 type annotations in `torchgen/` | 3.9+ | Custom script `fix_py38_compat.py` |
| 18 | Type alias definitions with `\|` syntax | 3.10+ | Custom script `fix_type_aliases.py` |

### C/C++ Source Code Fixes

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | Python 3.9+ C API functions (`PyObject_CallNoArgs`, `Py_IS_TYPE`, etc.) | Deployed `pythoncapi_compat.h` compatibility header |
| 2 | `PyType_GetModule` / `PyType_GetModuleByDef` (3.9+) | Compat implementations in `pythoncapi_compat.h` |
| 3 | `Py_NewRef` / `Py_XNewRef` / `Py_Is` etc. (3.10+) | Inline wrappers via `pythoncapi_compat.h` |
| 4 | Static inline redefinition conflicts | Per-function `#ifndef` guards in `pythoncapi_compat.h` |
| 5 | `PY_SSIZE_T_CLEAN` not defined before `Python.h` | Auto-fix in C files |
| 6 | MSVC-specific pragmas on GCC | `#ifdef _MSC_VER` guards |
| 7 | Python version constraints in `setup.py` / `pyproject.toml` | Updated to allow Python 3.8 |

### Manual Fixes

Some issues could not be fixed automatically and required manual intervention:

- **`torchgen` package**: The PyTorch code generation tool uses Python 3.10+ features extensively. We added `from __future__ import annotations` and converted type aliases manually.
- **`pyproject.toml`**: Updated `requires-python` from `>=3.10` to `>=3.8`
- **`setup.py`**: Updated Python version checks
- **Various `.py` files**: Fixed remaining `X | Y` type union syntax not caught by the automated tools

## Key Features

This backport provides all PyTorch 2.13 features on Python 3.8, including:

- **torch.compile / TorchDynamo**: Just-in-time compilation for PyTorch code
- **Transformer models**: Full `nn.TransformerEncoder`, `nn.TransformerDecoder` support
- **Automatic Mixed Precision (AMP)**: `torch.autocast` and `GradScaler` for CPU and CUDA
- **torch.jit**: TorchScript tracing and scripting
- **torch.fx**: Graph transformation and symbolic tracing
- **Dynamic quantization**: `torch.quantization.quantize_dynamic`
- **ONNX export**: Model export to ONNX format
- **torch.profiler**: Performance profiling
- **CUDA 12.4 support**: Full GPU acceleration with CUDA
- **Distributed training**: `torch.distributed` support
- **All nn modules**: Conv2d, LSTM, BatchNorm, Embedding, etc.

## Build Configuration

| Setting | Value |
|---------|-------|
| PyTorch version | 2.13.0a0+git03855a7 |
| Python version | 3.8 |
| Platform | Windows x64 |
| C++ standard | C++17 (MSVC 19.44) |
| CUDA version | 12.4 |
| NVCC arch flags | sm_86 |
| Build type | Release |
| OpenMP | 2019 |
| CPU capability | AVX2 |

## Test Results

We tested this backport against the last officially supported PyTorch version for Python 3.8 (**PyTorch 2.0.1**). Here is a comparison:

| Feature | PyTorch 2.0.1 (official 3.8) | PyTorch 2.13.0a0 (this backport) |
|---------|-------------------------------|----------------------------------|
| `torch.compile` | Not available | Working |
| `nn.TransformerEncoder` | Basic | Improved (batch_first default) |
| `torch.fx` symbolic_trace | Basic | Enhanced |
| Dynamic quantization | Working | Working |
| AMP (CPU bfloat16) | Not available | Working |
| `torch.profiler` | Basic | Enhanced |
| CUDA 12.4 | Not supported (max 11.8) | Supported |
| `torch.onnx.export` | Working | Working (improved) |
| `torch.jit.script` | Working | Working |
| Gradient checkpointing | Working | Working |
| Custom autograd.Function | Working | Working |

### Running the Test Suite

A comprehensive test file is included at `test_pytorch_functions.py`. To run it:

```bash
python test_pytorch_functions.py
```

The test suite covers:
- Core tensor operations (creation, math, indexing, reshape, broadcasting, dtypes)
- Autograd (basic, chain rule, custom functions, gradient checkpointing)
- Neural network modules (Linear, Conv2d, BatchNorm, LSTM, Transformer, etc.)
- Loss functions (MSE, CrossEntropy, BCE)
- Optimizers (SGD, Adam) and training loops
- Data utilities (DataLoader)
- Model save/load and serialization
- Advanced features (JIT trace/script, FX, quantization, ONNX, profiler, AMP)
- CUDA operations (if GPU available)
- torch.compile / Dynamo

## How to Build from Source

### Prerequisites

- **Python 3.8** (installed at `C:\Python38` or similar)
- **Visual Studio 2022** with C++ Build Tools (MSVC 19.44+)
- **CUDA 12.4** (optional, for GPU support)
- **Ninja** build system
- **Git** with long path support enabled

### Step 1: Clone the Repository

```bash
git config --global core.longpaths true
git clone https://github.com/Lanurence666/pytorch_backport_py38.git
cd pytorch_backport_py38
git submodule sync
git submodule update --init --recursive
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
pip install mkl-static mkl-include
pip install ninja cmake
```

### Step 3: Apply Compatibility Fixes

If you want to re-apply the fixes (they are already applied in this repo):

```bash
# Using the Python 3.8 Compatibility Fix Suite
pip install git+https://github.com/Lanurence666/python38_compat_fix_suite.git
python fix_py38_compat.py    # Fix torchgen type unions
python fix_type_aliases.py   # Fix type alias definitions
```

Or use the standalone scripts:

```bash
# From the python38_compat_fix_suite repository
python fix_py38_python.py .
python fix_py38_c.py .
```

### Step 4: Build PyTorch

```cmd
:: Set up Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Build and install in editable mode
python -m pip install --no-build-isolation -v -e .
```

For CPU-only builds:

```cmd
set USE_CUDA=0
python -m pip install --no-build-isolation -v -e .
```

### Step 5: Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
python test_pytorch_functions.py
```

## Known Limitations

1. **Python 3.8 end-of-life**: Python 3.8 reached EOL in October 2024. Consider upgrading to Python 3.10+ when possible.
2. **Some typing features**: Complex type annotations with nested `|` unions may still have issues in edge cases.
3. **`torch.distributed`**: Limited testing on Windows; primarily designed for Linux.
4. **`torch.compile`**: May have edge cases with Python 3.8-specific bytecode differences.
5. **No pre-built wheel**: Currently only available as source code with editable install. Building from source requires significant time (30-60 minutes) and disk space (10+ GB).

## Related Projects

| Project | Description |
|---------|-------------|
| [python38_compat_fix_suite](https://github.com/Lanurence666/python38_compat_fix_suite) | Automated Python 3.8 compatibility fix scripts |
| [numpy_backport_py38](https://github.com/Lanurence666/numpy_backport_py38) | NumPy 2.x backport for Python 3.8 |
| [scipy_backport_py38](https://github.com/Lanurence666/scipy_backport_py38) | SciPy 1.x backport for Python 3.8 |

## License

PyTorch is licensed under the **BSD-3-Clause** license. See [LICENSE](LICENSE) for details.

The `pythoncapi_compat.h` file is licensed under the **Zero Clause BSD (0BSD)** license from [python/pythoncapi-compat](https://github.com/python/pythoncapi-compat).

The compatibility fix scripts are licensed under the **MIT** license.
