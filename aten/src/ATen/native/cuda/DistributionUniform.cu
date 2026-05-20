#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#undef TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/Tensor.h>
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void uniform_kernel(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  if (isFloat8Type(iter.dtype())) {
    Tensor self = iter.tensor(0);
    Tensor self_hp = self.to(at::ScalarType::Float);
    auto hp_iter = TensorIterator::borrowing_nullary_op(self_hp);
    uniform_kernel(hp_iter, from, to, gen);
    self.copy_(self_hp.to(self.scalar_type()));
    return;
  }
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  templates::cuda::uniform_kernel(iter, from, to, generator);
}

REGISTER_DISPATCH(uniform_stub, &uniform_kernel)

} // namespace at::native
