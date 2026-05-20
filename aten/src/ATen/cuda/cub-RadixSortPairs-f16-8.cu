#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

#if CUB_HAS_SCAN_BY_KEY()
AT_INSTANTIATE_SORT_PAIRS(c10::BFloat16, 8)
#endif

} // namespace at::cuda::cub::detail
