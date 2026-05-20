#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#undef TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/Tensor.h>
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void normal_kernel(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
  if (isFloat8Type(self.scalar_type())) {
    Tensor self_t(self);
    Tensor self_hp = self_t.to(at::ScalarType::Float);
    normal_kernel(self_hp, mean, std, gen);
    self_t.copy_(self_hp.to(self_t.scalar_type()));
    return;
  }
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::normal_kernel(self, mean, std, generator);
}

REGISTER_DISPATCH(normal_stub, &normal_kernel)

} // namespace at::native
