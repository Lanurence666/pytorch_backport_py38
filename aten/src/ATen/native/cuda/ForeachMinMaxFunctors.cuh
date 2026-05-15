#pragma once

#include <ATen/NumericUtils.h>
#include <ATen/AccumulateType.h>
#include <ATen/OpMathType.h>

namespace at::native {

template <typename T>
struct minimum {
  __device__ T operator()(const T& a, const T& b) const {
    using opmath_t = at::opmath_type<T>;
    return (_isnan(a) || static_cast<opmath_t>(a) < static_cast<opmath_t>(b)) ? a : b;
  }
};

template <typename T>
struct maximum {
  __device__ T operator()(const T& a, const T& b) const {
    using opmath_t = at::opmath_type<T>;
    return (_isnan(a) || static_cast<opmath_t>(a) > static_cast<opmath_t>(b)) ? a : b;
  }
};

} // namespace at::native
