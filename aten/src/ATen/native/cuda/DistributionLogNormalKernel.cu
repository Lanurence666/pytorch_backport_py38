#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#undef TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/Tensor.h>
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void log_normal_kernel(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
  if (isFloat8Type(iter.input_dtype())) {
    Tensor self = iter.tensor(0);
    Tensor self_hp = self.to(at::ScalarType::Float);
    auto hp_iter = TensorIterator::borrowing_nullary_op(self_hp);
    log_normal_kernel(hp_iter, mean, std, gen);
    self.copy_(self_hp.to(self.scalar_type()));
    return;
  }
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::log_normal_kernel(iter, mean, std, generator);
}

REGISTER_DISPATCH(log_normal_stub, &log_normal_kernel)

} // namespace at::native
