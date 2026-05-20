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
#include <ATen/NumericUtils.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {
namespace {

void softshrink_kernel(TensorIteratorBase& iter, const Scalar& value) {
  if (isFloat8Type(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND4(
        at::ScalarType::Float8_e4m3fn, at::ScalarType::Float8_e5m2,
        at::ScalarType::Float8_e4m3fnuz, at::ScalarType::Float8_e5m2fnuz,
        iter.dtype(), "softshrink_cuda", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          auto lambd = value.to<opmath_t>();
          gpu_kernel(iter, [lambd] GPU_LAMBDA(scalar_t a) -> scalar_t {
            opmath_t aop = static_cast<opmath_t>(a);
            return at::_isnan(a) ? a : (aop > lambd ? aop - lambd : (aop < -lambd ? aop + lambd : opmath_t(0)));
          });
        });
    return;
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "softshrink_cuda",
      [&]() {
        auto lambd = value.to<scalar_t>();
        gpu_kernel(iter, [lambd] GPU_LAMBDA(scalar_t a) -> scalar_t {
          auto fa = static_cast<float>(a);
          auto fl = static_cast<float>(lambd);
          return at::_isnan(a) ? a : (fa > fl ? a - lambd : (fa < -fl ? a + lambd : scalar_t(0)));
        });
      });
}

void shrink_backward_kernel(TensorIteratorBase& iter, const Scalar& value) {
  if (isFloat8Type(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND4(
        at::ScalarType::Float8_e4m3fn, at::ScalarType::Float8_e5m2,
        at::ScalarType::Float8_e4m3fnuz, at::ScalarType::Float8_e5m2fnuz,
        iter.dtype(), "shrink_backward_cuda", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          auto lambd = value.to<opmath_t>();
          gpu_kernel(
              iter,
              [lambd] GPU_LAMBDA(
                  scalar_t grad_val, scalar_t self_val) -> scalar_t {
                opmath_t sv = static_cast<opmath_t>(self_val);
                return (sv >= -lambd && sv <= lambd) ? opmath_t(0)
                                                     : static_cast<opmath_t>(grad_val);
              });
        });
    return;
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "shrink_backward_cuda",
      [&]() {
        auto lambd = value.to<scalar_t>();
        gpu_kernel(
            iter,
            [lambd] GPU_LAMBDA(
                scalar_t grad_val, scalar_t self_val) -> scalar_t {
              auto fsv = static_cast<float>(self_val);
              auto fl = static_cast<float>(lambd);
              return (fsv >= -fl && fsv <= fl) ? scalar_t(0)
                                                               : grad_val;
            });
      });
}
} // namespace

REGISTER_DISPATCH(softshrink_stub, &softshrink_kernel)
REGISTER_DISPATCH(shrink_backward_stub, &shrink_backward_kernel)

} // namespace at::native
