#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#ifdef _MSC_VER
#pragma optimize("", off)
#endif
#include <ATen/native/LinearAlgebra.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/irange.h>
#include <cstring>

namespace at::native { namespace {

#if defined(_MSC_VER) && _MSC_VER < 1930
static void addr_kernel_scalar(TensorIterator &iter,
                               const Scalar& beta, const Scalar& alpha) {
  if (iter.dtype() == ScalarType::Bool) {
    int64_t itemsize = iter.element_size(0);
    auto beta_val = beta.to<bool>();
    auto alpha_val = alpha.to<bool>();
    iter.for_each([itemsize, beta_val, alpha_val](char** base, const int64_t* strides, int64_t size) {
      char* out_ptr = base[0];
      char* self_ptr = base[1];
      char* vec1_ptr = base[2];
      char* vec2_ptr = base[3];
      for (int64_t i = 0; i < size; i++) {
        bool self_val = *reinterpret_cast<const bool*>(self_ptr);
        bool vec1_val = *reinterpret_cast<const bool*>(vec1_ptr);
        bool vec2_val = *reinterpret_cast<const bool*>(vec2_ptr);
        bool result = (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
        std::memcpy(out_ptr, &result, sizeof(bool));
        out_ptr += strides[0];
        self_ptr += strides[1];
        vec1_ptr += strides[2];
        vec2_ptr += strides[3];
      }
    });
    iter.cast_outputs();
    return;
  }

  int64_t itemsize = iter.element_size(0);
  auto dtype = iter.dtype();

  if (dtype == ScalarType::Float) {
    auto beta_val = beta.to<float>();
    auto alpha_val = alpha.to<float>();
    const float zero_val(0);
    iter.for_each([itemsize, beta_val, alpha_val, zero_val](char** base, const int64_t* strides, int64_t size) {
      char* out_ptr = base[0];
      char* self_ptr = base[1];
      char* vec1_ptr = base[2];
      char* vec2_ptr = base[3];
      for (int64_t i = 0; i < size; i++) {
        float self_val = *reinterpret_cast<const float*>(self_ptr);
        float vec1_val = *reinterpret_cast<const float*>(vec1_ptr);
        float vec2_val = *reinterpret_cast<const float*>(vec2_ptr);
        float result = (beta_val == zero_val) ? (alpha_val * vec1_val * vec2_val) : (beta_val * self_val + alpha_val * vec1_val * vec2_val);
        std::memcpy(out_ptr, &result, static_cast<size_t>(itemsize));
        out_ptr += strides[0];
        self_ptr += strides[1];
        vec1_ptr += strides[2];
        vec2_ptr += strides[3];
      }
    });
  } else if (dtype == ScalarType::Double) {
    auto beta_val = beta.to<double>();
    auto alpha_val = alpha.to<double>();
    const double zero_val(0);
    iter.for_each([itemsize, beta_val, alpha_val, zero_val](char** base, const int64_t* strides, int64_t size) {
      char* out_ptr = base[0];
      char* self_ptr = base[1];
      char* vec1_ptr = base[2];
      char* vec2_ptr = base[3];
      for (int64_t i = 0; i < size; i++) {
        double self_val = *reinterpret_cast<const double*>(self_ptr);
        double vec1_val = *reinterpret_cast<const double*>(vec1_ptr);
        double vec2_val = *reinterpret_cast<const double*>(vec2_ptr);
        double result = (beta_val == zero_val) ? (alpha_val * vec1_val * vec2_val) : (beta_val * self_val + alpha_val * vec1_val * vec2_val);
        std::memcpy(out_ptr, &result, static_cast<size_t>(itemsize));
        out_ptr += strides[0];
        self_ptr += strides[1];
        vec1_ptr += strides[2];
        vec2_ptr += strides[3];
      }
    });
  } else if (dtype == ScalarType::ComplexFloat) {
    auto beta_val = beta.to<c10::complex<float>>();
    auto alpha_val = alpha.to<c10::complex<float>>();
    const c10::complex<float> zero_val(0);
    iter.for_each([itemsize, beta_val, alpha_val, zero_val](char** base, const int64_t* strides, int64_t size) {
      char* out_ptr = base[0];
      char* self_ptr = base[1];
      char* vec1_ptr = base[2];
      char* vec2_ptr = base[3];
      for (int64_t i = 0; i < size; i++) {
        c10::complex<float> self_val = *reinterpret_cast<const c10::complex<float>*>(self_ptr);
        c10::complex<float> vec1_val = *reinterpret_cast<const c10::complex<float>*>(vec1_ptr);
        c10::complex<float> vec2_val = *reinterpret_cast<const c10::complex<float>*>(vec2_ptr);
        c10::complex<float> result = (beta_val == zero_val) ? (alpha_val * vec1_val * vec2_val) : (beta_val * self_val + alpha_val * vec1_val * vec2_val);
        std::memcpy(out_ptr, &result, static_cast<size_t>(itemsize));
        out_ptr += strides[0];
        self_ptr += strides[1];
        vec1_ptr += strides[2];
        vec2_ptr += strides[3];
      }
    });
  } else if (dtype == ScalarType::ComplexDouble) {
    auto beta_val = beta.to<c10::complex<double>>();
    auto alpha_val = alpha.to<c10::complex<double>>();
    const c10::complex<double> zero_val(0);
    iter.for_each([itemsize, beta_val, alpha_val, zero_val](char** base, const int64_t* strides, int64_t size) {
      char* out_ptr = base[0];
      char* self_ptr = base[1];
      char* vec1_ptr = base[2];
      char* vec2_ptr = base[3];
      for (int64_t i = 0; i < size; i++) {
        c10::complex<double> self_val = *reinterpret_cast<const c10::complex<double>*>(self_ptr);
        c10::complex<double> vec1_val = *reinterpret_cast<const c10::complex<double>*>(vec1_ptr);
        c10::complex<double> vec2_val = *reinterpret_cast<const c10::complex<double>*>(vec2_ptr);
        c10::complex<double> result = (beta_val == zero_val) ? (alpha_val * vec1_val * vec2_val) : (beta_val * self_val + alpha_val * vec1_val * vec2_val);
        std::memcpy(out_ptr, &result, static_cast<size_t>(itemsize));
        out_ptr += strides[0];
        self_ptr += strides[1];
        vec1_ptr += strides[2];
        vec2_ptr += strides[3];
      }
    });
  } else if (dtype == ScalarType::Long) {
    auto beta_val = beta.to<int64_t>();
    auto alpha_val = alpha.to<int64_t>();
    const int64_t zero_val(0);
    iter.for_each([itemsize, beta_val, alpha_val, zero_val](char** base, const int64_t* strides, int64_t size) {
      char* out_ptr = base[0];
      char* self_ptr = base[1];
      char* vec1_ptr = base[2];
      char* vec2_ptr = base[3];
      for (int64_t i = 0; i < size; i++) {
        int64_t self_val = *reinterpret_cast<const int64_t*>(self_ptr);
        int64_t vec1_val = *reinterpret_cast<const int64_t*>(vec1_ptr);
        int64_t vec2_val = *reinterpret_cast<const int64_t*>(vec2_ptr);
        int64_t result = (beta_val == zero_val) ? (alpha_val * vec1_val * vec2_val) : (beta_val * self_val + alpha_val * vec1_val * vec2_val);
        std::memcpy(out_ptr, &result, static_cast<size_t>(itemsize));
        out_ptr += strides[0];
        self_ptr += strides[1];
        vec1_ptr += strides[2];
        vec2_ptr += strides[3];
      }
    });
  } else if (dtype == ScalarType::Int) {
    auto beta_val = beta.to<int32_t>();
    auto alpha_val = alpha.to<int32_t>();
    const int32_t zero_val(0);
    iter.for_each([itemsize, beta_val, alpha_val, zero_val](char** base, const int64_t* strides, int64_t size) {
      char* out_ptr = base[0];
      char* self_ptr = base[1];
      char* vec1_ptr = base[2];
      char* vec2_ptr = base[3];
      for (int64_t i = 0; i < size; i++) {
        int32_t self_val = *reinterpret_cast<const int32_t*>(self_ptr);
        int32_t vec1_val = *reinterpret_cast<const int32_t*>(vec1_ptr);
        int32_t vec2_val = *reinterpret_cast<const int32_t*>(vec2_ptr);
        int32_t result = (beta_val == zero_val) ? (alpha_val * vec1_val * vec2_val) : (beta_val * self_val + alpha_val * vec1_val * vec2_val);
        std::memcpy(out_ptr, &result, static_cast<size_t>(itemsize));
        out_ptr += strides[0];
        self_ptr += strides[1];
        vec1_ptr += strides[2];
        vec2_ptr += strides[3];
      }
    });
  } else if (dtype == ScalarType::Short) {
    auto beta_val = beta.to<int16_t>();
    auto alpha_val = alpha.to<int16_t>();
    const int16_t zero_val(0);
    iter.for_each([itemsize, beta_val, alpha_val, zero_val](char** base, const int64_t* strides, int64_t size) {
      char* out_ptr = base[0];
      char* self_ptr = base[1];
      char* vec1_ptr = base[2];
      char* vec2_ptr = base[3];
      for (int64_t i = 0; i < size; i++) {
        int16_t self_val = *reinterpret_cast<const int16_t*>(self_ptr);
        int16_t vec1_val = *reinterpret_cast<const int16_t*>(vec1_ptr);
        int16_t vec2_val = *reinterpret_cast<const int16_t*>(vec2_ptr);
        int16_t result = (beta_val == zero_val) ? (alpha_val * vec1_val * vec2_val) : (beta_val * self_val + alpha_val * vec1_val * vec2_val);
        std::memcpy(out_ptr, &result, static_cast<size_t>(itemsize));
        out_ptr += strides[0];
        self_ptr += strides[1];
        vec1_ptr += strides[2];
        vec2_ptr += strides[3];
      }
    });
  } else if (dtype == ScalarType::BFloat16) {
    auto beta_val = beta.to<BFloat16>();
    auto alpha_val = alpha.to<BFloat16>();
    const BFloat16 zero_val(0);
    iter.for_each([itemsize, beta_val, alpha_val, zero_val](char** base, const int64_t* strides, int64_t size) {
      char* out_ptr = base[0];
      char* self_ptr = base[1];
      char* vec1_ptr = base[2];
      char* vec2_ptr = base[3];
      for (int64_t i = 0; i < size; i++) {
        BFloat16 self_val = *reinterpret_cast<const BFloat16*>(self_ptr);
        BFloat16 vec1_val = *reinterpret_cast<const BFloat16*>(vec1_ptr);
        BFloat16 vec2_val = *reinterpret_cast<const BFloat16*>(vec2_ptr);
        BFloat16 result = (beta_val == zero_val) ? (alpha_val * vec1_val * vec2_val) : (beta_val * self_val + alpha_val * vec1_val * vec2_val);
        std::memcpy(out_ptr, &result, static_cast<size_t>(itemsize));
        out_ptr += strides[0];
        self_ptr += strides[1];
        vec1_ptr += strides[2];
        vec2_ptr += strides[3];
      }
    });
  } else if (dtype == ScalarType::Half) {
    auto beta_val = beta.to<Half>();
    auto alpha_val = alpha.to<Half>();
    const Half zero_val(0);
    iter.for_each([itemsize, beta_val, alpha_val, zero_val](char** base, const int64_t* strides, int64_t size) {
      char* out_ptr = base[0];
      char* self_ptr = base[1];
      char* vec1_ptr = base[2];
      char* vec2_ptr = base[3];
      for (int64_t i = 0; i < size; i++) {
        Half self_val = *reinterpret_cast<const Half*>(self_ptr);
        Half vec1_val = *reinterpret_cast<const Half*>(vec1_ptr);
        Half vec2_val = *reinterpret_cast<const Half*>(vec2_ptr);
        Half result = (beta_val == zero_val) ? (alpha_val * vec1_val * vec2_val) : (beta_val * self_val + alpha_val * vec1_val * vec2_val);
        std::memcpy(out_ptr, &result, static_cast<size_t>(itemsize));
        out_ptr += strides[0];
        self_ptr += strides[1];
        vec1_ptr += strides[2];
        vec2_ptr += strides[3];
      }
    });
  }
  iter.cast_outputs();
}
#endif

void addr_kernel(TensorIterator &iter,
                 const Scalar& beta, const Scalar& alpha) {
#if defined(_MSC_VER) && _MSC_VER < 1930
  addr_kernel_scalar(iter, beta, alpha);
#else
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    if (beta_val == false) {
      cpu_kernel(iter,
        [=](scalar_t /*self_val*/,
            scalar_t vec1_val,
            scalar_t vec2_val) -> scalar_t {
          return alpha_val && vec1_val && vec2_val;
        }
      );
    } else {
      cpu_kernel(iter,
        [=](scalar_t self_val,
            scalar_t vec1_val,
            scalar_t vec2_val) -> scalar_t {
          return (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
        }
      );
    }
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf,
    iter.dtype(), "addr_cpu", [&]() {
      using Vec = Vectorized<scalar_t>;
      auto beta_val = beta.to<scalar_t>();
      auto alpha_val = alpha.to<scalar_t>();
      auto beta_vec = Vec(beta_val);
      auto alpha_vec = Vec(alpha_val);
      const scalar_t zero_val(0);
      if (beta_val == zero_val) {
        cpu_kernel_vec(iter,
          [=](scalar_t /*self_val*/,
              scalar_t vec1_val,
              scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
            return alpha_val * vec1_val * vec2_val;
          },
          [=](Vec /*self_vec*/,
              Vec vec1_vec,
              Vec vec2_vec) __ubsan_ignore_undefined__ {
            return alpha_vec * vec1_vec * vec2_vec;
          }
        );
      } else {
        cpu_kernel_vec(iter,
          [=](scalar_t self_val,
              scalar_t vec1_val,
              scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
            return beta_val * self_val + alpha_val * vec1_val * vec2_val;
          },
          [=](Vec self_vec,
              Vec vec1_vec,
              Vec vec2_vec) __ubsan_ignore_undefined__ {
            return beta_vec * self_vec + alpha_vec * vec1_vec * vec2_vec;
          }
        );
      }
    }
  );
#endif
}

} // anonymous namespace

REGISTER_DISPATCH(addr_stub, &addr_kernel)
} // namespace at::native
