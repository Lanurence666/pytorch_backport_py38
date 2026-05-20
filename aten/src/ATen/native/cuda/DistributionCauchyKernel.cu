#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#undef TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/Tensor.h>
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void cauchy_kernel(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
  if (isFloat8Type(iter.input_dtype())) {
    Tensor self = iter.tensor(0);
    Tensor self_hp = self.to(at::ScalarType::Float);
    auto hp_iter = TensorIterator::borrowing_nullary_op(self_hp);
    cauchy_kernel(hp_iter, median, sigma, gen);
    self.copy_(self_hp.to(self.scalar_type()));
    return;
  }
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::cauchy_kernel(iter, median, sigma, generator);
}

REGISTER_DISPATCH(cauchy_stub, &cauchy_kernel)

} // namespace at::native

