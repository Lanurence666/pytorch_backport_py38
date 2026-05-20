#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

void maximum_kernel_cuda(TensorIteratorBase& iter) {
  if (iter.input_dtype() == ScalarType::Bool) {
    opmath_symmetric_gpu_kernel_with_scalars<bool>(
        iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a || b;
    });
  } else if (isIntegralType(iter.input_dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.input_dtype(), "max_elementwise_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::max(a, b);
      });
    });
  } else if (isFloat8Type(iter.input_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND4(
        at::ScalarType::Float8_e4m3fn, at::ScalarType::Float8_e5m2,
        at::ScalarType::Float8_e4m3fnuz, at::ScalarType::Float8_e5m2fnuz,
        iter.input_dtype(), "max_elementwise_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        if (a != a) {
          return a;
        } else if (b != b) {
          return b;
        } else {
          return ::max(a, b);
        }
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "max_elementwise_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        if (a != a) {
          return a;
        } else if (b != b) {
          return b;
        } else {
          return ::max(a, b);
        }
      });
    });
  }
}

void minimum_kernel_cuda(TensorIteratorBase& iter) {
  if (iter.input_dtype() == ScalarType::Bool) {
    opmath_symmetric_gpu_kernel_with_scalars<bool>(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a && b;
    });
  } else if (isIntegralType(iter.input_dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.input_dtype(), "minimum_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::min(a, b);
      });
    });
  } else if (isFloat8Type(iter.input_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND4(
        at::ScalarType::Float8_e4m3fn, at::ScalarType::Float8_e5m2,
        at::ScalarType::Float8_e4m3fnuz, at::ScalarType::Float8_e5m2fnuz,
        iter.input_dtype(), "min_elementwise_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        if (a != a) {
          return a;
        } else if (b != b) {
          return b;
        } else {
          return ::min(a, b);
        }
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "min_elementwise_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        if (a != a) {
          return a;
        } else if (b != b) {
          return b;
        } else {
          return ::min(a, b);
        }
      });
    });
  }
}

void fmax_kernel_cuda(TensorIteratorBase& iter) {
  if (isFloat8Type(iter.input_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND4(
        at::ScalarType::Float8_e4m3fn, at::ScalarType::Float8_e5m2,
        at::ScalarType::Float8_e4m3fnuz, at::ScalarType::Float8_e5m2fnuz,
        iter.input_dtype(), "fmax_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::fmax(a, b);
      });
    });
  } else if (isFloatingType(iter.input_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "fmax_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::fmax(a, b);
      });
    });
  } else {
    maximum_kernel_cuda(iter);
  }
}

void fmin_kernel_cuda(TensorIteratorBase& iter) {
  if (isFloat8Type(iter.input_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND4(
        at::ScalarType::Float8_e4m3fn, at::ScalarType::Float8_e5m2,
        at::ScalarType::Float8_e4m3fnuz, at::ScalarType::Float8_e5m2fnuz,
        iter.input_dtype(), "fmin_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::fmin(a, b);
      });
    });
  } else if (isFloatingType(iter.input_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "fmin_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::fmin(a, b);
      });
    });
  } else {
    minimum_kernel_cuda(iter);
  }
}

REGISTER_DISPATCH(maximum_stub, &maximum_kernel_cuda)
REGISTER_DISPATCH(minimum_stub, &minimum_kernel_cuda)
REGISTER_DISPATCH(fmax_stub, &fmax_kernel_cuda)
REGISTER_DISPATCH(fmin_stub, &fmin_kernel_cuda)

} // namespace at::native
