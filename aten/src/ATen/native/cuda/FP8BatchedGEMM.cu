#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/empty.h>

namespace at::native {

namespace {

using namespace at::cuda::detail;

constexpr int BLOCK_ROWS = 64;
constexpr int BLOCK_COLS = 64;
constexpr int BLOCK_K = 32;
constexpr int THREAD_ROWS = 4;
constexpr int THREAD_COLS = 4;
constexpr int WARP_ROWS = 32;
constexpr int WARP_COLS = 16;
constexpr int WARP_SIZE = 32;
constexpr int PAD = 4;

template <typename fp8_t>
__device__ __forceinline__ float fp8_to_float(fp8_t val) {
  return static_cast<float>(val);
}

template <typename fp8_t>
__device__ __forceinline__ fp8_t float_to_fp8(float val) {
  return static_cast<fp8_t>(val);
}

template <typename fp8_t, bool TRANS_A, bool TRANS_B>
__global__ void __launch_bounds__(256)
fp8_bgemm_tiled_kernel(
    const fp8_t* __restrict__ A_global,
    const fp8_t* __restrict__ B_global,
    const float* __restrict__ C_global,
    float* __restrict__ D_global,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const float alpha,
    const float beta,
    const int64_t stride_a_batch,
    const int64_t stride_b_batch,
    const int64_t stride_c_batch,
    const int64_t stride_d_batch,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldd) {
  __shared__ float sA[2][BLOCK_K][BLOCK_ROWS + PAD];
  __shared__ float sB[2][BLOCK_K][BLOCK_COLS + PAD];

  const int64_t batch_idx = blockIdx.z;
  const fp8_t* A = A_global + batch_idx * stride_a_batch;
  const fp8_t* B = B_global + batch_idx * stride_b_batch;
  const float* C = C_global + batch_idx * stride_c_batch;
  float* D = D_global + batch_idx * stride_d_batch;

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  const int thread_id = warp_id * WARP_SIZE + lane_id;

  const int64_t block_row = blockIdx.y * BLOCK_ROWS;
  const int64_t block_col = blockIdx.x * BLOCK_COLS;

  float regA[THREAD_ROWS];
  float regB[THREAD_COLS];
  float accum[THREAD_ROWS][THREAD_COLS];

  #pragma unroll
  for (int i = 0; i < THREAD_ROWS; ++i) {
    #pragma unroll
    for (int j = 0; j < THREAD_COLS; ++j) {
      accum[i][j] = 0.0f;
    }
  }

  const int warp_row_offset = (warp_id / 4) * WARP_ROWS;
  const int warp_col_offset = (warp_id % 4) * WARP_COLS;
  const int thread_row_in_warp = lane_id / 4;
  const int thread_col_in_warp = lane_id % 4;

  const int thread_row_base = warp_row_offset + thread_row_in_warp * THREAD_ROWS;
  const int thread_col_base = warp_col_offset + thread_col_in_warp * THREAD_COLS;

  auto load_tile_a = [&](int buf, int64_t k_start) {
    #pragma unroll
    for (int idx = thread_id; idx < BLOCK_K * BLOCK_ROWS; idx += 256) {
      int ki = idx / BLOCK_ROWS;
      int mi = idx % BLOCK_ROWS;
      int64_t global_row = block_row + mi;
      int64_t global_k = k_start + ki;
      if (global_row < M && global_k < K) {
        float val;
        if constexpr (TRANS_A) {
          val = fp8_to_float(A[global_k * lda + global_row]);
        } else {
          val = fp8_to_float(A[global_row * lda + global_k]);
        }
        sA[buf][ki][mi] = val;
      } else {
        sA[buf][ki][mi] = 0.0f;
      }
    }
  };

  auto load_tile_b = [&](int buf, int64_t k_start) {
    #pragma unroll
    for (int idx = thread_id; idx < BLOCK_K * BLOCK_COLS; idx += 256) {
      int ki = idx / BLOCK_COLS;
      int ni = idx % BLOCK_COLS;
      int64_t global_k = k_start + ki;
      int64_t global_col = block_col + ni;
      if (global_k < K && global_col < N) {
        float val;
        if constexpr (TRANS_B) {
          val = fp8_to_float(B[global_col * ldb + global_k]);
        } else {
          val = fp8_to_float(B[global_k * ldb + global_col]);
        }
        sB[buf][ki][ni] = val;
      } else {
        sB[buf][ki][ni] = 0.0f;
      }
    }
  };

  int read_buf = 0;
  int write_buf = 1;

  load_tile_a(0, 0);
  load_tile_b(0, 0);
  __syncthreads();

  for (int64_t k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    if (k_tile + BLOCK_K < K) {
      load_tile_a(write_buf, k_tile + BLOCK_K);
      load_tile_b(write_buf, k_tile + BLOCK_K);
    }

    #pragma unroll
    for (int kk = 0; kk < BLOCK_K; ++kk) {
      #pragma unroll
      for (int i = 0; i < THREAD_ROWS; ++i) {
        int row_idx = thread_row_base + i;
        if (row_idx < BLOCK_ROWS) {
          regA[i] = sA[read_buf][kk][row_idx];
        } else {
          regA[i] = 0.0f;
        }
      }

      #pragma unroll
      for (int j = 0; j < THREAD_COLS; ++j) {
        int col_idx = thread_col_base + j;
        if (col_idx < BLOCK_COLS) {
          regB[j] = sB[read_buf][kk][col_idx];
        } else {
          regB[j] = 0.0f;
        }
      }

      #pragma unroll
      for (int i = 0; i < THREAD_ROWS; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_COLS; ++j) {
          accum[i][j] += regA[i] * regB[j];
        }
      }
    }

    __syncthreads();
    read_buf = write_buf;
    write_buf = 1 - write_buf;
  }

  #pragma unroll
  for (int i = 0; i < THREAD_ROWS; ++i) {
    #pragma unroll
    for (int j = 0; j < THREAD_COLS; ++j) {
      int64_t global_row = block_row + thread_row_base + i;
      int64_t global_col = block_col + thread_col_base + j;
      if (global_row < M && global_col < N) {
        float c_val = 0.0f;
        if (beta != 0.0f) {
          c_val = C[global_row * ldc + global_col];
        }
        D[global_row * ldd + global_col] = alpha * accum[i][j] + beta * c_val;
      }
    }
  }
}

template <typename fp8_t, bool TRANS_A, bool TRANS_B>
__global__ void __launch_bounds__(256)
fp8_gemm_single_kernel(
    const fp8_t* __restrict__ A,
    const fp8_t* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ D,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const float alpha,
    const float beta,
    const int64_t lda,
    const int64_t ldb,
    const int64_t ldc,
    const int64_t ldd) {
  __shared__ float sA[2][BLOCK_K][BLOCK_ROWS + PAD];
  __shared__ float sB[2][BLOCK_K][BLOCK_COLS + PAD];

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;

  const int64_t block_row = blockIdx.y * BLOCK_ROWS;
  const int64_t block_col = blockIdx.x * BLOCK_COLS;

  float regA[THREAD_ROWS];
  float regB[THREAD_COLS];
  float accum[THREAD_ROWS][THREAD_COLS];

  #pragma unroll
  for (int i = 0; i < THREAD_ROWS; ++i) {
    #pragma unroll
    for (int j = 0; j < THREAD_COLS; ++j) {
      accum[i][j] = 0.0f;
    }
  }

  const int warp_row_offset = (warp_id / 4) * WARP_ROWS;
  const int warp_col_offset = (warp_id % 4) * WARP_COLS;
  const int thread_row_in_warp = lane_id / 4;
  const int thread_col_in_warp = lane_id % 4;

  const int thread_row_base = warp_row_offset + thread_row_in_warp * THREAD_ROWS;
  const int thread_col_base = warp_col_offset + thread_col_in_warp * THREAD_COLS;

  auto load_tile_a = [&](int buf, int64_t k_start) {
    #pragma unroll
    for (int idx = threadIdx.y * WARP_SIZE + threadIdx.x; idx < BLOCK_K * BLOCK_ROWS; idx += 256) {
      int ki = idx / BLOCK_ROWS;
      int mi = idx % BLOCK_ROWS;
      int64_t global_row = block_row + mi;
      int64_t global_k = k_start + ki;
      if (global_row < M && global_k < K) {
        float val;
        if constexpr (TRANS_A) {
          val = fp8_to_float(A[global_k * lda + global_row]);
        } else {
          val = fp8_to_float(A[global_row * lda + global_k]);
        }
        sA[buf][ki][mi] = val;
      } else {
        sA[buf][ki][mi] = 0.0f;
      }
    }
  };

  auto load_tile_b = [&](int buf, int64_t k_start) {
    #pragma unroll
    for (int idx = threadIdx.y * WARP_SIZE + threadIdx.x; idx < BLOCK_K * BLOCK_COLS; idx += 256) {
      int ki = idx / BLOCK_COLS;
      int ni = idx % BLOCK_COLS;
      int64_t global_k = k_start + ki;
      int64_t global_col = block_col + ni;
      if (global_k < K && global_col < N) {
        float val;
        if constexpr (TRANS_B) {
          val = fp8_to_float(B[global_col * ldb + global_k]);
        } else {
          val = fp8_to_float(B[global_k * ldb + global_col]);
        }
        sB[buf][ki][ni] = val;
      } else {
        sB[buf][ki][ni] = 0.0f;
      }
    }
  };

  int read_buf = 0;
  int write_buf = 1;

  load_tile_a(0, 0);
  load_tile_b(0, 0);
  __syncthreads();

  for (int64_t k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    if (k_tile + BLOCK_K < K) {
      load_tile_a(write_buf, k_tile + BLOCK_K);
      load_tile_b(write_buf, k_tile + BLOCK_K);
    }

    #pragma unroll
    for (int kk = 0; kk < BLOCK_K; ++kk) {
      #pragma unroll
      for (int i = 0; i < THREAD_ROWS; ++i) {
        int row_idx = thread_row_base + i;
        if (row_idx < BLOCK_ROWS) {
          regA[i] = sA[read_buf][kk][row_idx];
        } else {
          regA[i] = 0.0f;
        }
      }

      #pragma unroll
      for (int j = 0; j < THREAD_COLS; ++j) {
        int col_idx = thread_col_base + j;
        if (col_idx < BLOCK_COLS) {
          regB[j] = sB[read_buf][kk][col_idx];
        } else {
          regB[j] = 0.0f;
        }
      }

      #pragma unroll
      for (int i = 0; i < THREAD_ROWS; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_COLS; ++j) {
          accum[i][j] += regA[i] * regB[j];
        }
      }
    }

    __syncthreads();
    read_buf = write_buf;
    write_buf = 1 - write_buf;
  }

  #pragma unroll
  for (int i = 0; i < THREAD_ROWS; ++i) {
    #pragma unroll
    for (int j = 0; j < THREAD_COLS; ++j) {
      int64_t global_row = block_row + thread_row_base + i;
      int64_t global_col = block_col + thread_col_base + j;
      if (global_row < M && global_col < N) {
        float c_val = 0.0f;
        if (beta != 0.0f) {
          c_val = C[global_row * ldc + global_col];
        }
        D[global_row * ldd + global_col] = alpha * accum[i][j] + beta * c_val;
      }
    }
  }
}

template <typename fp8_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void fp8_to_float_kernel(
    const int64_t n,
    const fp8_t* __restrict__ src,
    float* __restrict__ dst) {
  CUDA_KERNEL_LOOP_TYPE(index, n, int64_t) {
    dst[index] = static_cast<float>(src[index]);
  }
}

template <typename fp8_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void float_to_fp8_kernel(
    const int64_t n,
    const float* __restrict__ src,
    fp8_t* __restrict__ dst) {
  CUDA_KERNEL_LOOP_TYPE(index, n, int64_t) {
    dst[index] = static_cast<fp8_t>(src[index]);
  }
}

template <typename fp8_t, bool TRANS_A, bool TRANS_B>
void launch_bgemm_kernel(
    const Tensor& result_f32,
    const Tensor& self_f32,
    const Tensor& batch1,
    const Tensor& batch2,
    float alpha, float beta,
    int64_t M, int64_t N, int64_t K,
    int64_t num_batches,
    cudaStream_t stream) {
  dim3 block(WARP_SIZE, 8, 1);
  dim3 grid(
      (N + BLOCK_COLS - 1) / BLOCK_COLS,
      (M + BLOCK_ROWS - 1) / BLOCK_ROWS,
      num_batches);

  fp8_bgemm_tiled_kernel<fp8_t, TRANS_A, TRANS_B><<<grid, block, 0, stream>>>(
      batch1.const_data_ptr<fp8_t>(),
      batch2.const_data_ptr<fp8_t>(),
      self_f32.const_data_ptr<float>(),
      result_f32.mutable_data_ptr<float>(),
      M, N, K, alpha, beta,
      batch1.stride(0), batch2.stride(0),
      self_f32.stride(0), result_f32.stride(0),
      batch1.stride(2), batch2.stride(2),
      self_f32.stride(2), result_f32.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename fp8_t, bool TRANS_A, bool TRANS_B>
void launch_gemm_single_kernel(
    const Tensor& result_f32,
    const Tensor& self_f32,
    const Tensor& batch1,
    const Tensor& batch2,
    float alpha, float beta,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream) {
  dim3 block(WARP_SIZE, 8, 1);
  dim3 grid(
      (N + BLOCK_COLS - 1) / BLOCK_COLS,
      (M + BLOCK_ROWS - 1) / BLOCK_ROWS,
      1);

  fp8_gemm_single_kernel<fp8_t, TRANS_A, TRANS_B><<<grid, block, 0, stream>>>(
      batch1.const_data_ptr<fp8_t>(),
      batch2.const_data_ptr<fp8_t>(),
      self_f32.const_data_ptr<float>(),
      result_f32.mutable_data_ptr<float>(),
      M, N, K, alpha, beta,
      batch1.stride(2), batch2.stride(2),
      self_f32.stride(2), result_f32.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename fp8_t>
void dispatch_transpose_combinations(
    const Tensor& result_f32,
    const Tensor& self_f32,
    const Tensor& batch1,
    const Tensor& batch2,
    float alpha, float beta,
    int64_t M, int64_t N, int64_t K,
    int64_t num_batches,
    bool transa, bool transb,
    cudaStream_t stream) {
  if (num_batches == 1) {
    if (!transa && !transb) {
      launch_gemm_single_kernel<fp8_t, false, false>(
          result_f32, self_f32, batch1, batch2, alpha, beta, M, N, K, stream);
    } else if (!transa && transb) {
      launch_gemm_single_kernel<fp8_t, false, true>(
          result_f32, self_f32, batch1, batch2, alpha, beta, M, N, K, stream);
    } else if (transa && !transb) {
      launch_gemm_single_kernel<fp8_t, true, false>(
          result_f32, self_f32, batch1, batch2, alpha, beta, M, N, K, stream);
    } else {
      launch_gemm_single_kernel<fp8_t, true, true>(
          result_f32, self_f32, batch1, batch2, alpha, beta, M, N, K, stream);
    }
  } else {
    if (!transa && !transb) {
      launch_bgemm_kernel<fp8_t, false, false>(
          result_f32, self_f32, batch1, batch2, alpha, beta, M, N, K, num_batches, stream);
    } else if (!transa && transb) {
      launch_bgemm_kernel<fp8_t, false, true>(
          result_f32, self_f32, batch1, batch2, alpha, beta, M, N, K, num_batches, stream);
    } else if (transa && !transb) {
      launch_bgemm_kernel<fp8_t, true, false>(
          result_f32, self_f32, batch1, batch2, alpha, beta, M, N, K, num_batches, stream);
    } else {
      launch_bgemm_kernel<fp8_t, true, true>(
          result_f32, self_f32, batch1, batch2, alpha, beta, M, N, K, num_batches, stream);
    }
  }
}

template <typename fp8_t>
void fp8_baddbmm_cuda(
    const Tensor& result,
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta_scalar,
    const Scalar& alpha_scalar) {
  auto result_sizes = result.sizes();
  int64_t num_batches = result_sizes[0];
  int64_t M = result_sizes[1];
  int64_t N = result_sizes[2];
  int64_t K = batch1.size(2);

  float alpha = alpha_scalar.to<float>();
  float beta = beta_scalar.to<float>();

  bool transa = false;
  bool transb = false;

  auto b1_strides = batch1.strides();
  auto b2_strides = batch2.strides();
  if (b1_strides[2] == 1 && b1_strides[1] > 1) {
    transa = false;
  } else if (b1_strides[1] == 1 && b1_strides[2] > 1) {
    transa = true;
  }
  if (b2_strides[2] == 1 && b2_strides[1] > 1) {
    transb = false;
  } else if (b2_strides[1] == 1 && b2_strides[2] > 1) {
    transb = true;
  }

  Tensor result_contiguous = result.is_contiguous() ? result : result.contiguous();
  Tensor self_contiguous = self.is_contiguous() ? self : self.contiguous();
  Tensor batch1_contiguous = batch1.is_contiguous() ? batch1 : batch1.contiguous();
  Tensor batch2_contiguous = batch2.is_contiguous() ? batch2 : batch2.contiguous();

  if (transa) {
    batch1_contiguous = batch1_contiguous.transpose(1, 2).contiguous();
    transa = false;
  }
  if (transb) {
    batch2_contiguous = batch2_contiguous.transpose(1, 2).contiguous();
    transb = false;
  }

  Tensor self_f32 = self_contiguous.to(at::ScalarType::Float);
  Tensor result_f32 = at::empty(result_contiguous.sizes(), result_contiguous.options().dtype(at::ScalarType::Float));

  auto stream = c10::cuda::getCurrentCUDAStream();

  dispatch_transpose_combinations<fp8_t>(
      result_f32, self_f32, batch1_contiguous, batch2_contiguous,
      alpha, beta, M, N, K, num_batches, transa, transb, stream);

  int64_t output_numel = result_contiguous.numel();
  float_to_fp8_kernel<fp8_t><<<GET_BLOCKS(output_numel), 1024, 0, stream>>>(
      output_numel,
      result_f32.const_data_ptr<float>(),
      result_contiguous.mutable_data_ptr<fp8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (!result.is_same(result_contiguous)) {
    result.copy_(result_contiguous);
  }
}

} // anonymous namespace

void fp8_baddbmm_dispatch(
    const Tensor& result,
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  if (batch1.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    fp8_baddbmm_cuda<c10::Float8_e4m3fn>(result, self, batch1, batch2, beta, alpha);
  } else if (batch1.scalar_type() == at::ScalarType::Float8_e5m2) {
    fp8_baddbmm_cuda<c10::Float8_e5m2>(result, self, batch1, batch2, beta, alpha);
  } else {
    TORCH_CHECK(false, "Unsupported FP8 type for baddbmm: ", batch1.scalar_type());
  }
}

} // namespace at::native
