#pragma once

#include <ATen/cuda/CUDAContext.h>
#if defined(USE_ROCM)
#include <hipsparse/hipsparse-version.h>
#define HIPSPARSE_VERSION ((hipsparseVersionMajor*100000) + (hipsparseVersionMinor*100) + hipsparseVersionPatch)
#endif


// cuSparse Generic API spsv function was added in CUDA 11.3.0
// hipSparse supports SpSV as well
#if (defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 11500) || defined(USE_ROCM)
#define AT_USE_CUSPARSE_GENERIC_SPSV() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SPSV() 0
#endif

// cuSparse Generic API spsm function was added in CUDA 12.0
// cusparseSpSMDescr is not available in CUDA 11.x
#if (defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12000) || defined(USE_ROCM)
#define AT_USE_CUSPARSE_GENERIC_SPSM() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SPSM() 0
#endif

// ConstCuSparseDescriptor requires CUDA 12+ const descriptor APIs
#if (defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && CUSPARSE_VERSION >= 12000) || defined(USE_ROCM)
#define AT_USE_CUSPARSE_CONST_DESCRIPTORS() 1
#else
#define AT_USE_CUSPARSE_CONST_DESCRIPTORS() 0
#endif
