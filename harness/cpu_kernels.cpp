// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/all.h>

namespace cpu_utils {
void rejection_greedy_sample_kernel_impl(torch::Tensor&, const torch::Tensor&,
                                         const torch::Tensor&,
                                         const torch::Tensor&,
                                         const torch::Tensor&,
                                         const std::optional<torch::Tensor>&,
                                         int64_t);
void rejection_random_sample_kernel_impl(
    torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const std::optional<torch::Tensor>&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const std::optional<torch::Tensor>&, int64_t, int64_t, bool);
}  // namespace cpu_utils

TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("rejection_greedy_sample_kernel_impl",
          &cpu_utils::rejection_greedy_sample_kernel_impl);
  ops.def("rejection_random_sample_kernel_impl",
          &cpu_utils::rejection_random_sample_kernel_impl);
}
