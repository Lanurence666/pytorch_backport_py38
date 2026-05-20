#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#undef TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/Tensor.h>
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void exponential_kernel(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
  if (isFloat8Type(iter.input_dtype())) {
    Tensor self = iter.tensor(0);
    Tensor self_hp = self.to(at::ScalarType::Float);
    auto hp_iter = TensorIterator::borrowing_nullary_op(self_hp);
    exponential_kernel(hp_iter, lambda, gen);
    self.copy_(self_hp.to(self.scalar_type()));
    return;
  }
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::exponential_kernel(iter, lambda, generator);
}

REGISTER_DISPATCH(exponential_stub, &exponential_kernel)

} // namespace at::native
