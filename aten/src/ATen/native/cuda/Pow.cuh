#pragma once
#include <ATen/native/Pow.h>
#include <c10/core/Scalar.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2fnuz.h>

namespace at::native {

namespace {


// SFINAE doesn't work well with NVCC under Windows for math functions like pow and sqrt.
// So we need to define the functions with the explicit function signatures.
// As for pow, the following signatures are defined as the device function:
//   pow(float, int)
//   pow(double, int)
//   pow(float, float)
//   pow(double, double)
#if defined(_MSC_VER) || defined(_LIBCPP_VERSION)
// Functions for pow
// pow for at::Half
static inline __host__ __device__ at::Half pow_(at::Half base, at::Half exp) {
  return static_cast<at::Half>(std::pow(static_cast<float>(base), static_cast<float>(exp)));
}
// pow for at::BFloat16
static inline __host__ __device__ at::BFloat16 pow_(at::BFloat16 base, at::BFloat16 exp) {
  return static_cast<at::BFloat16>(std::pow(static_cast<float>(base), static_cast<float>(exp)));
}
// pow for Float8 types
static inline __host__ __device__ c10::Float8_e4m3fn pow_(c10::Float8_e4m3fn base, c10::Float8_e4m3fn exp) {
  return static_cast<c10::Float8_e4m3fn>(std::pow(static_cast<float>(base), static_cast<float>(exp)));
}
static inline __host__ __device__ c10::Float8_e5m2 pow_(c10::Float8_e5m2 base, c10::Float8_e5m2 exp) {
  return static_cast<c10::Float8_e5m2>(std::pow(static_cast<float>(base), static_cast<float>(exp)));
}
static inline __host__ __device__ c10::Float8_e4m3fnuz pow_(c10::Float8_e4m3fnuz base, c10::Float8_e4m3fnuz exp) {
  return static_cast<c10::Float8_e4m3fnuz>(std::pow(static_cast<float>(base), static_cast<float>(exp)));
}
static inline __host__ __device__ c10::Float8_e5m2fnuz pow_(c10::Float8_e5m2fnuz base, c10::Float8_e5m2fnuz exp) {
  return static_cast<c10::Float8_e5m2fnuz>(std::pow(static_cast<float>(base), static_cast<float>(exp)));
}
// pow (floating, floating/int)
template <typename Base_type, typename Exp_type>
static inline __host__ __device__ typename std::enable_if_t<std::is_floating_point_v<Base_type> && (std::is_same_v<Base_type, Exp_type> || std::is_same_v<Exp_type, int>), Base_type>
  pow_(Base_type base, Exp_type exp) {
  return std::pow(base, exp);
}
// pow (Otherwise)
template <typename Base_type, typename Exp_type>
static inline __host__ __device__ typename std::enable_if_t<!std::is_same_v<Base_type, Exp_type> && !std::is_same_v<Exp_type, int>, Base_type>
  pow_(Base_type base, Exp_type exp) {
  return static_cast<Base_type>(std::pow(static_cast<double>(base), static_cast<double>(exp)));
}
#else
template <typename Base_type, typename Exp_type>
static inline __host__ __device__ Base_type pow_(Base_type base, Exp_type exp) {
  return ::pow(base, exp);
}
#endif

template <typename T>
static inline __host__ __device__ std::enable_if_t<std::is_integral_v<T>, T> pow_(
    T base, T exp) {
  return at::native::powi(base, exp);
}

template <typename T>
static inline __host__ __device__ c10::complex<T> pow_(c10::complex<T> base, c10::complex<T> exp) {
  return c10_complex_math::pow(base, exp);
}

} // namespace
} // namespace at::native
