#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/div_rtn.h>
#include <c10/core/ScalarType.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/im2col.cuh>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_slow_conv2d_forward_native.h>
#include <ATen/ops/_slow_conv2d_backward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#endif

namespace at::native {

namespace {

template <typename fp8_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void fp8_im2col_kernel(
    const int64_t n,
    const fp8_t* data_im,
    const int64_t height,
    const int64_t width,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t height_col,
    const int64_t width_col,
    float* data_col) {
  CUDA_KERNEL_LOOP_TYPE(index, n, int64_t) {
    int64_t w_out = index % width_col;
    int64_t idx = index / width_col;
    int64_t h_out = idx % height_col;
    int64_t channel_in = idx / height_col;
    int64_t channel_out = channel_in * kernel_height * kernel_width;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = w_out * stride_width - pad_width;

    float* col = data_col + (channel_out * height_col + h_out) * width_col + w_out;
    const fp8_t* im = data_im + (channel_in * height + h_in) * width + w_in;

    for (int64_t i = 0; i < kernel_height; ++i) {
      for (int64_t j = 0; j < kernel_width; ++j) {
        int64_t h = h_in + i * dilation_height;
        int64_t w = w_in + j * dilation_width;
        float val = 0.0f;
        if (h >= 0 && w >= 0 && h < height && w < width) {
          val = static_cast<float>(im[i * dilation_height * width + j * dilation_width]);
        }
        *col = val;
        col += height_col * width_col;
      }
    }
  }
}

C10_LAUNCH_BOUNDS_1(1024)
__global__ void fp8_e4m3fn_im2col_kernel(
    const int64_t n,
    const c10::Float8_e4m3fn* data_im,
    const int64_t height, const int64_t width,
    const int64_t kh, const int64_t kw,
    const int64_t ph, const int64_t pw,
    const int64_t sh, const int64_t sw,
    const int64_t dh, const int64_t dw,
    const int64_t h_col, const int64_t w_col,
    float* data_col) {
  CUDA_KERNEL_LOOP_TYPE(index, n, int64_t) {
    int64_t w_out = index % w_col;
    int64_t idx = index / w_col;
    int64_t h_out = idx % h_col;
    int64_t c_in = idx / h_col;
    int64_t c_out = c_in * kh * kw;
    int64_t h_in = h_out * sh - ph;
    int64_t w_in = w_out * sw - pw;
    float* col = data_col + (c_out * h_col + h_out) * w_col + w_out;
    const c10::Float8_e4m3fn* im = data_im + (c_in * height + h_in) * width + w_in;
    for (int64_t i = 0; i < kh; ++i) {
      for (int64_t j = 0; j < kw; ++j) {
        int64_t h = h_in + i * dh;
        int64_t w = w_in + j * dw;
        float val = 0.0f;
        if (h >= 0 && w >= 0 && h < height && w < width) {
          val = static_cast<float>(im[i * dh * width + j * dw]);
        }
        *col = val;
        col += h_col * w_col;
      }
    }
  }
}

C10_LAUNCH_BOUNDS_1(1024)
__global__ void fp8_e5m2_im2col_kernel(
    const int64_t n,
    const c10::Float8_e5m2* data_im,
    const int64_t height, const int64_t width,
    const int64_t kh, const int64_t kw,
    const int64_t ph, const int64_t pw,
    const int64_t sh, const int64_t sw,
    const int64_t dh, const int64_t dw,
    const int64_t h_col, const int64_t w_col,
    float* data_col) {
  CUDA_KERNEL_LOOP_TYPE(index, n, int64_t) {
    int64_t w_out = index % w_col;
    int64_t idx = index / w_col;
    int64_t h_out = idx % h_col;
    int64_t c_in = idx / h_col;
    int64_t c_out = c_in * kh * kw;
    int64_t h_in = h_out * sh - ph;
    int64_t w_in = w_out * sw - pw;
    float* col = data_col + (c_out * h_col + h_out) * w_col + w_out;
    const c10::Float8_e5m2* im = data_im + (c_in * height + h_in) * width + w_in;
    for (int64_t i = 0; i < kh; ++i) {
      for (int64_t j = 0; j < kw; ++j) {
        int64_t h = h_in + i * dh;
        int64_t w = w_in + j * dw;
        float val = 0.0f;
        if (h >= 0 && w >= 0 && h < height && w < width) {
          val = static_cast<float>(im[i * dh * width + j * dw]);
        }
        *col = val;
        col += h_col * w_col;
      }
    }
  }
}

C10_LAUNCH_BOUNDS_1(1024)
__global__ void float_to_fp8_e4m3fn_kernel(
    const int64_t n,
    const float* src,
    c10::Float8_e4m3fn* dst) {
  CUDA_KERNEL_LOOP_TYPE(index, n, int64_t) {
    dst[index] = static_cast<c10::Float8_e4m3fn>(src[index]);
  }
}

C10_LAUNCH_BOUNDS_1(1024)
__global__ void float_to_fp8_e5m2_kernel(
    const int64_t n,
    const float* src,
    c10::Float8_e5m2* dst) {
  CUDA_KERNEL_LOOP_TYPE(index, n, int64_t) {
    dst[index] = static_cast<c10::Float8_e5m2>(src[index]);
  }
}

void slow_conv2d_shape_check(
    const Tensor& input, const Tensor& grad_output,
    const Tensor& weight, const Tensor& bias,
    int64_t kH, int64_t kW,
    int64_t dH, int64_t dW,
    int64_t padH, int64_t padW,
    bool weight_nullable) {
  TORCH_CHECK(kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got kH: ", kH, " kW: ", kW);
  TORCH_CHECK(dW > 0 && dH > 0,
              "stride should be greater than zero, but got dH: ", dH, " dW: ", dW);

  TORCH_CHECK(weight_nullable || weight.defined(),
              "weight tensor is expected to be non-nullable");
  TORCH_CHECK(!weight.defined() ||
              ((weight.numel() > 0) && (weight.dim() == 2)),
              "non-empty 2D weight tensor expected, but got: ", weight.sizes());
  TORCH_CHECK(!bias.defined() || (bias.dim() == 1 && bias.sizes()[0] == weight.sizes()[0]),
              "Expected bias to have shape [", weight.sizes()[0], "] but got ", bias.sizes());

  const auto in_sizes = input.sizes();
  constexpr int ndim = 4;
  constexpr int dimf = 1;
  constexpr int dimh = 2;
  constexpr int dimw = 3;
  TORCH_CHECK(in_sizes.size() == ndim, "Expected 4D input tensor, but got ", in_sizes);

  // Allow for empty batch size but not other dimensions
  const bool valid_empty = c10::multiply_integers(in_sizes.slice(1)) != 0;
  TORCH_CHECK(valid_empty, "non-empty input tensor expected but got: ", in_sizes);

  int64_t inputHeight = in_sizes[dimh];
  int64_t inputWidth = in_sizes[dimw];

  int64_t exactInputHeight = inputHeight + 2 * padH;
  int64_t exactInputWidth = inputWidth + 2 * padW;

  TORCH_CHECK(exactInputHeight >= kH && exactInputWidth >= kW,
              "Calculated padded input size per channel: ",
              IntArrayRef{exactInputHeight, exactInputWidth},
              ". Kernel size: ", IntArrayRef{kH, kW},
              ". Kernel size can't be greater than actual input size");

  // NOTE: can't use conv_output_size if the weight isn't defined
  auto outputHeight = div_rtn<int64_t>(exactInputHeight - kH, dH) + 1;
  auto outputWidth = div_rtn<int64_t>(exactInputWidth - kW, dW) + 1;

  TORCH_CHECK(outputWidth >= 1 && outputHeight >= 1,
              "Given input size per channel: ",
              IntArrayRef{inputHeight, inputWidth},
              ". Calculated output size per channel: ",
              IntArrayRef{outputHeight, outputWidth},
              ". Output size is too small");

  if (weight.defined()) {
    const auto w_sizes = weight.sizes();
    int64_t nInputPlane = w_sizes[1];
    if (w_sizes.size() == 2) {
      nInputPlane /= (kH * kW);
    }
    TORCH_CHECK(in_sizes[dimf] == nInputPlane,
                "Expected input dim ", dimf, " to have size ", nInputPlane,
                " but got ", in_sizes[dimf]);
  }

  if (grad_output.defined()) {
    const auto gO_sizes = grad_output.sizes();
    TORCH_CHECK(gO_sizes.size() == ndim,
                "Expected grad_output to have ", ndim,
                " dimensions but got shape", gO_sizes);

    if (weight.defined()) {
      const auto w_sizes = weight.sizes();
      TORCH_CHECK(gO_sizes[dimf] == w_sizes[0],
                  "Expected  dim ", dimf, " to have size ", w_sizes[0],
                  " but got ", gO_sizes[dimf]);
    } else if (bias.defined()) {
      const auto b_sizes = bias.sizes();
      int64_t nOutputPlane = b_sizes.size() == 0 ? 1 : b_sizes[0];
      TORCH_CHECK(gO_sizes[dimf] == nOutputPlane,
                  "Expected grad_output dim ", dimf, " to have size ",
                  nOutputPlane, " but got ", gO_sizes[dimf]);
    }
    TORCH_CHECK(gO_sizes[dimh] == outputHeight,
                "Expected grad_output dim ", dimh, " to have size ",
                outputHeight, " but got ", gO_sizes[dimh]);
    TORCH_CHECK(gO_sizes[dimw] == outputWidth,
                "Expected grad_output dim ", dimw, " to have size ",
                outputWidth, " but got ", gO_sizes[dimw]);
  }
}

Tensor new_view_weight_MM2d(const Tensor& weight_) {
  auto weight = weight_.expect_contiguous();
  const auto w_sizes = weight->sizes();
  TORCH_CHECK(w_sizes.size() == 4);
  int64_t s1 = w_sizes[0];
  int64_t s2 = c10::multiply_integers(w_sizes.slice(1));
  return weight->view({s1, s2});
}

void fp8_e4m3fn_conv2d_forward(
    const Tensor &input,
    const Tensor &weight_,
    const Tensor &bias,
    Tensor &output,
    int64_t kH, int64_t kW,
    int64_t dH, int64_t dW,
    int64_t padH, int64_t padW) {
  auto weight = new_view_weight_MM2d(weight_);
  constexpr int dimf = 1, dimh = 2, dimw = 3;
  auto in_sizes = input.sizes();
  int64_t batchSize = in_sizes[0];
  int64_t nInputPlane  = in_sizes[dimf];
  int64_t inputHeight  = in_sizes[dimh];
  int64_t inputWidth   = in_sizes[dimw];
  int64_t nOutputPlane = weight.sizes()[0];
  int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  resize_output(output, {batchSize, nOutputPlane, outputHeight, outputWidth});
  const bool requires_columns = (kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);
  at::Tensor columns_f32;
  if (requires_columns) {
    columns_f32 = at::empty({nInputPlane * kW * kH, outputHeight * outputWidth},
                            input.options().dtype(at::ScalarType::Float));
  }
  at::Tensor weight_f32 = weight_.to(at::ScalarType::Float);
  auto weight_view = new_view_weight_MM2d(weight_f32);
  at::Tensor output_f32 = at::empty({batchSize, nOutputPlane, outputHeight, outputWidth},
                                     input.options().dtype(at::ScalarType::Float));
  if (bias.defined()) {
    output_f32.copy_(bias.to(at::ScalarType::Float).view({-1, 1, 1}));
  } else {
    output_f32.zero_();
  }
  auto stream = c10::cuda::getCurrentCUDAStream();
  for (int elt = 0; elt < batchSize; elt++) {
    auto input_n = input.select(0, elt);
    auto output_n = output_f32.select(0, elt);
    if (requires_columns) {
      int64_t num_kernels = nInputPlane * outputHeight * outputWidth;
      fp8_e4m3fn_im2col_kernel<<<GET_BLOCKS(num_kernels), 1024, 0, stream>>>(
          num_kernels,
          static_cast<const c10::Float8_e4m3fn*>(input_n.const_data_ptr()),
          inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, 1, 1,
          outputHeight, outputWidth,
          columns_f32.mutable_data_ptr<float>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    int64_t m = nOutputPlane;
    int64_t n = outputHeight * outputWidth;
    int64_t k = nInputPlane * kH * kW;
    const float* gemm_in_ptr = requires_columns ?
        columns_f32.const_data_ptr<float>() : nullptr;
    at::Tensor input_n_f32_flat;
    if (!requires_columns) {
      input_n_f32_flat = input_n.to(at::ScalarType::Float).view({nInputPlane, -1});
      gemm_in_ptr = input_n_f32_flat.const_data_ptr<float>();
    }
    at::cuda::blas::gemm(
        'n', 'n', n, m, k, 1.0f,
        gemm_in_ptr, n,
        static_cast<const float*>(weight_view.const_data_ptr()), k,
        1.0f,
        output_n.mutable_data_ptr<float>(), n);
  }
  int64_t output_numel = output_f32.numel();
  float_to_fp8_e4m3fn_kernel<<<GET_BLOCKS(output_numel), 1024, 0, stream>>>(
      output_numel,
      output_f32.const_data_ptr<float>(),
      static_cast<c10::Float8_e4m3fn*>(output.mutable_data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_e5m2_conv2d_forward(
    const Tensor &input,
    const Tensor &weight_,
    const Tensor &bias,
    Tensor &output,
    int64_t kH, int64_t kW,
    int64_t dH, int64_t dW,
    int64_t padH, int64_t padW) {
  auto weight = new_view_weight_MM2d(weight_);
  constexpr int dimf = 1, dimh = 2, dimw = 3;
  auto in_sizes = input.sizes();
  int64_t batchSize = in_sizes[0];
  int64_t nInputPlane  = in_sizes[dimf];
  int64_t inputHeight  = in_sizes[dimh];
  int64_t inputWidth   = in_sizes[dimw];
  int64_t nOutputPlane = weight.sizes()[0];
  int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  resize_output(output, {batchSize, nOutputPlane, outputHeight, outputWidth});
  const bool requires_columns = (kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);
  at::Tensor columns_f32;
  if (requires_columns) {
    columns_f32 = at::empty({nInputPlane * kW * kH, outputHeight * outputWidth},
                            input.options().dtype(at::ScalarType::Float));
  }
  at::Tensor weight_f32 = weight_.to(at::ScalarType::Float);
  auto weight_view = new_view_weight_MM2d(weight_f32);
  at::Tensor output_f32 = at::empty({batchSize, nOutputPlane, outputHeight, outputWidth},
                                     input.options().dtype(at::ScalarType::Float));
  if (bias.defined()) {
    output_f32.copy_(bias.to(at::ScalarType::Float).view({-1, 1, 1}));
  } else {
    output_f32.zero_();
  }
  auto stream = c10::cuda::getCurrentCUDAStream();
  for (int elt = 0; elt < batchSize; elt++) {
    auto input_n = input.select(0, elt);
    auto output_n = output_f32.select(0, elt);
    if (requires_columns) {
      int64_t num_kernels = nInputPlane * outputHeight * outputWidth;
      fp8_e5m2_im2col_kernel<<<GET_BLOCKS(num_kernels), 1024, 0, stream>>>(
          num_kernels,
          static_cast<const c10::Float8_e5m2*>(input_n.const_data_ptr()),
          inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, 1, 1,
          outputHeight, outputWidth,
          columns_f32.mutable_data_ptr<float>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    int64_t m = nOutputPlane;
    int64_t n = outputHeight * outputWidth;
    int64_t k = nInputPlane * kH * kW;
    const float* gemm_in_ptr = requires_columns ?
        columns_f32.const_data_ptr<float>() : nullptr;
    at::Tensor input_n_f32_flat;
    if (!requires_columns) {
      input_n_f32_flat = input_n.to(at::ScalarType::Float).view({nInputPlane, -1});
      gemm_in_ptr = input_n_f32_flat.const_data_ptr<float>();
    }
    at::cuda::blas::gemm(
        'n', 'n', n, m, k, 1.0f,
        gemm_in_ptr, n,
        static_cast<const float*>(weight_view.const_data_ptr()), k,
        1.0f,
        output_n.mutable_data_ptr<float>(), n);
  }
  int64_t output_numel = output_f32.numel();
  float_to_fp8_e5m2_kernel<<<GET_BLOCKS(output_numel), 1024, 0, stream>>>(
      output_numel,
      output_f32.const_data_ptr<float>(),
      static_cast<c10::Float8_e5m2*>(output.mutable_data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void slow_conv2d_forward(
           const Tensor &input,
           const Tensor &output,
           const Tensor &weight_,
           const Tensor &bias,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW) {
  auto weight = new_view_weight_MM2d(weight_);
  slow_conv2d_shape_check(
      input, {}, weight, bias, kH, kW, dH, dW, padH, padW, /*weight_nullable*/false);

  constexpr int dimf = 1;
  constexpr int dimh = 2;
  constexpr int dimw = 3;

  auto in_sizes = input.sizes();
  int64_t batchSize = in_sizes[0];
  int64_t nInputPlane  = in_sizes[dimf];
  int64_t inputHeight  = in_sizes[dimh];
  int64_t inputWidth   = in_sizes[dimw];
  int64_t nOutputPlane = weight.sizes()[0];
  int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

  // Resize output
  resize_output(output, {batchSize, nOutputPlane, outputHeight, outputWidth});

  // Create temporary columns
  at::Tensor columns;

  const bool requires_columns = (
      kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);

  if (requires_columns) {
    columns = at::empty({nInputPlane * kW * kH, outputHeight * outputWidth}, input.options());
  }

  if (bias.defined()) {
    TORCH_CHECK(bias.scalar_type() == input.scalar_type(),
                "Expected bias to have type ", input.scalar_type(),
                " but got ", bias.scalar_type());
    output.copy_(bias.view({-1, 1, 1}));
  } else {
    output.zero_();
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                  "slow_conv2d_cuda", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt ++) {
      // Matrix multiply per output:
      auto input_n = input.select(0, elt);
      auto output_n = output.select(0, elt);

      if (requires_columns) {
        // Extract columns:
        at::native::im2col(
          c10::cuda::getCurrentCUDAStream(),
          input_n.const_data_ptr<scalar_t>(),
          nInputPlane, inputHeight, inputWidth,
          outputHeight, outputWidth,
          kH, kW, padH, padW, dH, dW,
          1, 1,
          columns.mutable_data_ptr<scalar_t>()
        );
      }

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m = nOutputPlane;
      int64_t n = outputHeight * outputWidth;
      int64_t k = nInputPlane*kH*kW;

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      auto gemm_in_ptr = requires_columns ?
          columns.const_data_ptr<scalar_t>() :
          input_n.const_data_ptr<scalar_t>();
      at::cuda::blas::gemm(
          'n', 'n',
          n, m, k,
          scalar_t(1),
          gemm_in_ptr, n,
          weight.const_data_ptr<scalar_t>(), k,
          scalar_t(1),
          output_n.mutable_data_ptr<scalar_t>(), n
      );
    }
  });
}

void slow_conv2d_backward(
    const Tensor &input,
    const Tensor &grad_output,
    const Tensor &grad_input,
    const Tensor &weight_,
    const Tensor &grad_columns,
    int kH, int kW,
    int dH, int dW,
    int padH, int padW) {
  Tensor weight = new_view_weight_MM2d(weight_);
  slow_conv2d_shape_check(input, grad_output, weight, {},
                          kH, kW, dH, dW, padH, padW, /*weight_nullable=*/false);

  // Params
  auto weight_sizes = weight.sizes();
  int nInputPlane = weight_sizes[1]/(kW*kH);
  int nOutputPlane = weight_sizes[0];

  TORCH_INTERNAL_ASSERT(grad_output.is_contiguous());

  auto input_sizes = input.sizes();
  int64_t inputWidth   = input_sizes[3];
  int64_t inputHeight  = input_sizes[2];
  auto output_sizes = grad_output.sizes();
  int64_t outputWidth  = output_sizes[3];
  int64_t outputHeight = output_sizes[2];

  // Batch size + input planes
  int64_t batchSize = input_sizes[0];

  // Resize output
  resize_output(grad_input, input_sizes);
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");

  // Resize temporary columns
  resize_output(grad_columns, {nInputPlane*kW*kH, outputHeight*outputWidth});
  TORCH_CHECK(grad_columns.is_contiguous(), "grad_columns must be contiguous");

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                  "slow_conv2d_backward_cuda", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt ++) {
      // Matrix multiply per sample:
      auto grad_input_n = grad_input.select(0, elt);
      auto grad_output_n = grad_output.select(0, elt);

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m = nInputPlane*kW*kH;
      int64_t n = grad_columns.sizes()[1];
      int64_t k = nOutputPlane;

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      at::cuda::blas::gemm<scalar_t>(
          'n', 't',
          n, m, k,
          scalar_t(1),
          grad_output_n.const_data_ptr<scalar_t>(), n,
          weight.const_data_ptr<scalar_t>(), m,
          scalar_t(0),
          grad_columns.mutable_data_ptr<scalar_t>(), n
      );

      // Unpack columns back into input:
      using acc_t = at::acc_type<scalar_t, true>;
      at::native::col2im<scalar_t, acc_t>(
        c10::cuda::getCurrentCUDAStream(),
        grad_columns.const_data_ptr<scalar_t>(),
        nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW,
        1, 1, grad_input_n.mutable_data_ptr<scalar_t>()
      );
    }
  });
}

void slow_conv2d_grad_weight(
           const Tensor &input,
           const Tensor &grad_output,
           const Tensor &grad_weight_,
           const Tensor &columns,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW) {
  TORCH_CHECK(grad_weight_.is_contiguous(), "grad_weight needs to be contiguous");
  auto grad_weight = new_view_weight_MM2d(grad_weight_);
  slow_conv2d_shape_check(input, grad_output, grad_weight, {},
                          kH, kW, dH, dW, padH, padW, /*weight_nullable=*/true);

  // Params
  TORCH_INTERNAL_ASSERT(input.is_contiguous());
  TORCH_INTERNAL_ASSERT(grad_output.is_contiguous());

  auto input_sizes = input.sizes();
  int64_t nInputPlane = input_sizes[1];
  int64_t nOutputPlane = grad_output.sizes()[1];

  int64_t inputWidth   = input_sizes[3];
  int64_t inputHeight  = input_sizes[2];
  int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input_sizes[0];

  // Resize temporary columns
  resize_output(columns, {nInputPlane * kH * kW, outputHeight * outputWidth});

  const bool requires_columns = (
      kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                  "slow_conv2d_grad_weight_cuda", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt ++) {
      // Matrix multiply per output:
      auto grad_output_n = grad_output.select(0, elt);

      // Matrix multiply per output:
      auto input_n = input.select(0, elt);

      if (requires_columns) {
        // Extract columns:
        at::native::im2col<scalar_t>(
          c10::cuda::getCurrentCUDAStream(),
          input_n.const_data_ptr<scalar_t>(),
          nInputPlane, inputHeight, inputWidth,
          outputHeight, outputWidth,
          kH, kW, padH, padW, dH, dW,
          1, 1,
          columns.mutable_data_ptr<scalar_t>()
        );
      }

      // M,N,K are dims of matrix A and B
      // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
      int64_t m = nOutputPlane;
      int64_t n = nInputPlane*kW*kH;
      int64_t k = columns.sizes()[1];

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      auto gemm_in_ptr = requires_columns ?
          columns.const_data_ptr<scalar_t>() :
          input_n.const_data_ptr<scalar_t>();
      at::cuda::blas::gemm(
          't', 'n',
          n, m, k,
          scalar_t(1),
          gemm_in_ptr, k,
          grad_output_n.const_data_ptr<scalar_t>(), k,
          scalar_t(1),
          grad_weight.mutable_data_ptr<scalar_t>(), n
      );
    }
  });
}

}  // namespace (anonymous)


Tensor& slow_conv2d_forward_out_cuda(
    const Tensor &self_,
    const Tensor &weight_,
    IntArrayRef kernel_size,
    const std::optional<Tensor> &bias_,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor &output) {
  TORCH_CHECK(kernel_size.size() == 2);
  TORCH_CHECK(stride.size() == 2);
  TORCH_CHECK(padding.size() == 2);

  if (isFloat8Type(self_.scalar_type())) {
    auto self = self_.expect_contiguous();
    auto weight = weight_.expect_contiguous();
    auto bias = [&] {
      if (bias_.has_value() && bias_->defined()) {
        return bias_->expect_contiguous();
      }
      return MaybeOwned<Tensor>::owned(std::in_place);
    }();

    slow_conv2d_shape_check(*self, {}, new_view_weight_MM2d(*weight),
                            *bias, kernel_size[0], kernel_size[1],
                            stride[0], stride[1], padding[0], padding[1], false);

    if (self_.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      fp8_e4m3fn_conv2d_forward(
          *self, *weight, *bias, output,
          kernel_size[0], kernel_size[1],
          stride[0], stride[1],
          padding[0], padding[1]);
    } else if (self_.scalar_type() == at::ScalarType::Float8_e5m2) {
      fp8_e5m2_conv2d_forward(
          *self, *weight, *bias, output,
          kernel_size[0], kernel_size[1],
          stride[0], stride[1],
          padding[0], padding[1]);
    } else {
      TORCH_CHECK(false, "Unsupported FP8 type for conv2d: ", self_.scalar_type());
    }
    return output;
  }

  auto self = self_.expect_contiguous();
  auto weight = weight_.expect_contiguous();
  auto bias = [&] {
    if (bias_.has_value() && bias_->defined()) {
      return bias_->expect_contiguous();
    }
    return MaybeOwned<Tensor>::owned(std::in_place);
  }();

  slow_conv2d_forward(
      *self,
      output,
      *weight,
      *bias,
      kernel_size[0], kernel_size[1],
      stride[0], stride[1],
      padding[0], padding[1]
    );
  return output;
}

Tensor slow_conv2d_forward_cuda(
    const Tensor &self,
    const Tensor &weight,
    IntArrayRef kernel_size,
    const std::optional<Tensor> &bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto output = at::empty({0}, self.options());
  return slow_conv2d_forward_out_cuda(
      self, weight, kernel_size, bias, stride, padding, output);
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_backward_out_cuda(
    const Tensor& grad_output_,
    const Tensor& self_,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  if (isFloat8Type(self_.scalar_type())) {
    auto self_f32 = self_.to(at::ScalarType::Float);
    auto grad_output_f32 = grad_output_.to(at::ScalarType::Float);
    auto weight_f32 = weight_.to(at::ScalarType::Float);

    Tensor grad_input_f32, grad_weight_f32, grad_bias_f32;
    if (grad_input.defined()) {
      grad_input_f32 = at::empty(self_f32.sizes(), self_f32.options());
    }
    if (grad_weight.defined()) {
      grad_weight_f32 = at::empty(weight_f32.sizes(), weight_f32.options());
    }
    if (grad_bias.defined()) {
      grad_bias_f32 = at::empty({weight_f32.size(0)}, self_f32.options());
    }

    auto result_f32 = slow_conv2d_backward_cuda(
        grad_output_f32, self_f32, weight_f32,
        kernel_size, stride, padding,
        {grad_input.defined(), grad_weight.defined(), grad_bias.defined()});

    if (grad_input.defined()) {
      resize_output(grad_input, self_.sizes());
      grad_input.copy_(std::get<0>(result_f32).to(self_.scalar_type()));
    }
    if (grad_bias.defined()) {
      at::sum_out(grad_bias, grad_output_, IntArrayRef{0, 2, 3});
    }
    if (grad_weight.defined()) {
      resize_output(grad_weight, weight_.sizes());
      grad_weight.copy_(std::get<1>(result_f32).to(self_.scalar_type()));
    }
    return std::tuple<Tensor&, Tensor&, Tensor&>{
        grad_input, grad_weight, grad_bias};
  }

  auto grad_output = grad_output_.expect_contiguous();

  Tensor columns = at::empty({0}, self_.options());
  if (grad_input.defined()) {
    resize_output(grad_input, self_.sizes());
    auto weight = weight_.expect_contiguous();

    slow_conv2d_backward(
        self_, *grad_output,
        grad_input, *weight,
        columns,
        kernel_size[0], kernel_size[1],
        stride[0], stride[1],
        padding[0], padding[1]);
  }
  if (grad_bias.defined()) {
    at::sum_out(grad_bias, *grad_output, IntArrayRef{0, 2, 3});
  }
  if (grad_weight.defined()) {
    resize_output(grad_weight, weight_.sizes());
    grad_weight.zero_();
    auto self = self_.expect_contiguous();
    slow_conv2d_grad_weight(
        *self,
        *grad_output,
        grad_weight,
        columns,
        kernel_size[0], kernel_size[1],
        stride[0], stride[1],
        padding[0], padding[1]
      );
  }
  return std::tuple<Tensor&, Tensor&, Tensor&>{
      grad_input, grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> slow_conv2d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    std::array<bool, 3> output_mask) {
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  }

  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  }

  if (output_mask[2]) {
    grad_bias = at::empty({0}, grad_output.options());
  }

  return native::slow_conv2d_backward_out_cuda(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      grad_input,
      grad_weight,
      grad_bias);
}

} // namespace at::native
