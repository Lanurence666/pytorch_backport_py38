#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {

// -----------------------------------
// prelu
// -----------------------------------
void prelu_kernel(TensorIterator &iter) {
  if (isFloat8Type(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND4(
        at::ScalarType::Float8_e4m3fn, at::ScalarType::Float8_e5m2,
        at::ScalarType::Float8_e4m3fnuz, at::ScalarType::Float8_e5m2fnuz,
        iter.dtype(), "prelu_cuda", [&] {
          using opmath_t = at::opmath_type<scalar_t>;
          gpu_kernel(iter,
            [] GPU_LAMBDA (scalar_t input, scalar_t weight) -> scalar_t {
              opmath_t iop = static_cast<opmath_t>(input);
              opmath_t wop = static_cast<opmath_t>(weight);
              return (iop > 0) ? iop : wop * iop;
            });
        });
    return;
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_cuda", [&] {
    gpu_kernel(iter,
      [] GPU_LAMBDA (scalar_t input, scalar_t weight) -> scalar_t {
        return (input > 0) ? input : weight * input;
      });
  });
}

void prelu_backward_kernel(TensorIterator &iter) {
  if (isFloat8Type(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND4(
        at::ScalarType::Float8_e4m3fn, at::ScalarType::Float8_e5m2,
        at::ScalarType::Float8_e4m3fnuz, at::ScalarType::Float8_e5m2fnuz,
        iter.dtype(), "prelu_backward_cuda", [&] {
          using opmath_t = at::opmath_type<scalar_t>;
          gpu_kernel_multiple_outputs(iter,
            [] GPU_LAMBDA (scalar_t input, scalar_t weight, scalar_t grad) -> thrust::tuple<scalar_t, scalar_t> {
              opmath_t iop = static_cast<opmath_t>(input);
              opmath_t wop = static_cast<opmath_t>(weight);
              opmath_t gop = static_cast<opmath_t>(grad);
              auto mask = iop > 0;
              auto grad_input = mask ? gop : wop * gop;
              auto grad_weight = mask ? opmath_t{0} : iop * gop;
              return {static_cast<scalar_t>(grad_input), static_cast<scalar_t>(grad_weight)};
            });
        });
    return;
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_backward_cuda", [&] {
    gpu_kernel_multiple_outputs(iter,
      [] GPU_LAMBDA (scalar_t input, scalar_t weight, scalar_t grad) -> thrust::tuple<scalar_t, scalar_t> {
        auto mask = input > 0;
        auto grad_input = mask ? grad : weight * grad;
        auto grad_weight = mask ? scalar_t{0} : input * grad;
        return {grad_input, grad_weight};
      });
  });
}

REGISTER_DISPATCH(prelu_stub, &prelu_kernel)
REGISTER_DISPATCH(prelu_backward_stub, &prelu_backward_kernel)

} // namespace at::native
