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
#include <ATen/OpMathType.h>

namespace at::native {
namespace {

template <typename scalar_t>
void threshold_kernel_impl(
    TensorIteratorBase& iter,
    scalar_t threshold,
    scalar_t value) {
  gpu_kernel_with_scalars(
      iter, [=] GPU_LAMBDA(scalar_t x, scalar_t other) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return static_cast<opmath_t>(x) <= static_cast<opmath_t>(threshold) ? value : other;
      });
}

static void threshold_kernel_cuda(
    TensorIteratorBase& iter,
    const Scalar& threshold,
    const Scalar& value) {
  if (isFloat8Type(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND4(
        at::ScalarType::Float8_e4m3fn, at::ScalarType::Float8_e5m2,
        at::ScalarType::Float8_e4m3fnuz, at::ScalarType::Float8_e5m2fnuz,
        iter.dtype(), "threshold_cuda", [&] {
          threshold_kernel_impl<scalar_t>(
              iter, threshold.to<scalar_t>(), value.to<scalar_t>());
        });
    return;
  }
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "threshold_cuda",
      [&] {
        threshold_kernel_impl<scalar_t>(
            iter, threshold.to<scalar_t>(), value.to<scalar_t>());
      });
}

} // namespace

REGISTER_DISPATCH(threshold_stub, &threshold_kernel_cuda)

} // namespace at::native
