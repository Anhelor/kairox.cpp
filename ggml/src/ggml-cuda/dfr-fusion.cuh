#pragma once

#include "common.cuh"

struct ggml_cuda_dfr_update_args {
    bool  has_sum_cols{};
    bool  use_ema{};
    int   group_size{};
    int   n_groups{};
    float sparse_threshold{};
    float lambda{};
    float one_minus_lambda{};
    float normalizer{};
};

void ggml_cuda_op_dfr_update(ggml_backend_cuda_context &       ctx,
                             const ggml_tensor *               raw_sparse,
                             const ggml_tensor *               scores_in,
                             ggml_tensor *                     scores_out,
                             const ggml_cuda_dfr_update_args & args);

void ggml_cuda_op_dfr_mask(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         topk_idx,
                           float *                     group_mask,
                           ggml_tensor *               load_group,
                           ggml_tensor *               evict_group);
