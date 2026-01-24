#include "dfr-fusion.cuh"

constexpr int kairox_dfr_warps_per_block = 4;
constexpr int kairox_dfr_mask_threads    = 256;
constexpr int kairox_dfr_mask_words      = 1024 / 32;

template <bool HAS_SUM_COLS, bool USE_EMA, int N_COLS = 0>
static __launch_bounds__(kairox_dfr_warps_per_block * WARP_SIZE, 1) __global__ void kairox_dfr_update_f32_kernel(
    const float * __restrict__ raw_sparse,
    const float * __restrict__ old_scores,
    float * __restrict__ out_scores,
    int   ne0,
    int   ne1,
    int   group_size,
    int   n_groups,
    float sparse_threshold,
    float lambda,
    float one_minus_lambda,
    float normalizer) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int gid     = blockIdx.x * kairox_dfr_warps_per_block + warp_id;
    if (gid >= n_groups) {
        return;
    }

    const int row_begin = gid * group_size;
    int       local     = 0;

    if constexpr (HAS_SUM_COLS) {
        if constexpr (N_COLS > 0) {
#pragma unroll
            for (int c = 0; c < N_COLS; ++c) {
                const float * base = raw_sparse + c * ne0 + row_begin;
                for (int r = lane_id; r < group_size; r += WARP_SIZE) {
                    local += base[r] >= sparse_threshold;
                }
            }
        } else {
            for (int c = 0; c < ne1; ++c) {
                const float * base = raw_sparse + c * ne0 + row_begin;
                for (int r = lane_id; r < group_size; r += WARP_SIZE) {
                    local += base[r] >= sparse_threshold;
                }
            }
        }
    } else {
        const float * base = raw_sparse + row_begin;
        for (int r = lane_id; r < group_size; r += WARP_SIZE) {
            local += base[r] >= sparse_threshold;
        }
    }

    const int sum = warp_reduce_sum(local);

    if (lane_id == 0) {
        const float prev  = old_scores[gid];
        const float delta = float(sum) / normalizer;
        out_scores[gid]   = prev * lambda + (USE_EMA ? delta * one_minus_lambda : delta);
    }
}

void ggml_cuda_op_dfr_update(ggml_backend_cuda_context &       ctx,
                             const ggml_tensor *               raw_sparse,
                             const ggml_tensor *               scores_in,
                             ggml_tensor *                     scores_out,
                             const ggml_cuda_dfr_update_args & args) {
    GGML_ASSERT(ggml_is_contiguous(raw_sparse));
    GGML_ASSERT(ggml_is_contiguous(scores_in));
    GGML_ASSERT(ggml_is_contiguous(scores_out));

    GGML_ASSERT(raw_sparse->ne[2] == 1 && raw_sparse->ne[3] == 1);
    GGML_ASSERT(scores_in->ne[1] == 1 && scores_in->ne[2] == 1 && scores_in->ne[3] == 1);
    GGML_ASSERT(scores_out->ne[1] == 1 && scores_out->ne[2] == 1 && scores_out->ne[3] == 1);

    const int ne0 = (int) raw_sparse->ne[0];
    const int ne1 = (int) raw_sparse->ne[1];

    GGML_ASSERT(args.group_size > 0 && args.n_groups > 0);
    GGML_ASSERT(ne0 == args.group_size * args.n_groups);
    GGML_ASSERT(scores_in->ne[0] == args.n_groups);
    GGML_ASSERT(scores_out->ne[0] == args.n_groups);

    const float * raw_sparse_data = (const float *) raw_sparse->data;
    const float * old_scores      = (const float *) scores_in->data;
    float *       out_scores      = (float *) scores_out->data;
    const int     blocks          = (args.n_groups + kairox_dfr_warps_per_block - 1) / kairox_dfr_warps_per_block;
    const int     threads         = kairox_dfr_warps_per_block * WARP_SIZE;

    if (!args.has_sum_cols) {
        if (args.use_ema) {
            kairox_dfr_update_f32_kernel<false, true, 0><<<blocks, threads, 0, ctx.stream()>>>(
                raw_sparse_data, old_scores, out_scores, ne0, ne1, args.group_size, args.n_groups,
                args.sparse_threshold, args.lambda, args.one_minus_lambda, args.normalizer);
        } else {
            kairox_dfr_update_f32_kernel<false, false, 0><<<blocks, threads, 0, ctx.stream()>>>(
                raw_sparse_data, old_scores, out_scores, ne0, ne1, args.group_size, args.n_groups,
                args.sparse_threshold, args.lambda, args.one_minus_lambda, args.normalizer);
        }
        return;
    }

    switch (ne1) {
        case 1:
            if (args.use_ema) {
                kairox_dfr_update_f32_kernel<true, true, 1><<<blocks, threads, 0, ctx.stream()>>>(
                    raw_sparse_data, old_scores, out_scores, ne0, ne1, args.group_size, args.n_groups,
                    args.sparse_threshold, args.lambda, args.one_minus_lambda, args.normalizer);
            } else {
                kairox_dfr_update_f32_kernel<true, false, 1><<<blocks, threads, 0, ctx.stream()>>>(
                    raw_sparse_data, old_scores, out_scores, ne0, ne1, args.group_size, args.n_groups,
                    args.sparse_threshold, args.lambda, args.one_minus_lambda, args.normalizer);
            }
            break;
        case 2:
            if (args.use_ema) {
                kairox_dfr_update_f32_kernel<true, true, 2><<<blocks, threads, 0, ctx.stream()>>>(
                    raw_sparse_data, old_scores, out_scores, ne0, ne1, args.group_size, args.n_groups,
                    args.sparse_threshold, args.lambda, args.one_minus_lambda, args.normalizer);
            } else {
                kairox_dfr_update_f32_kernel<true, false, 2><<<blocks, threads, 0, ctx.stream()>>>(
                    raw_sparse_data, old_scores, out_scores, ne0, ne1, args.group_size, args.n_groups,
                    args.sparse_threshold, args.lambda, args.one_minus_lambda, args.normalizer);
            }
            break;
        case 4:
            if (args.use_ema) {
                kairox_dfr_update_f32_kernel<true, true, 4><<<blocks, threads, 0, ctx.stream()>>>(
                    raw_sparse_data, old_scores, out_scores, ne0, ne1, args.group_size, args.n_groups,
                    args.sparse_threshold, args.lambda, args.one_minus_lambda, args.normalizer);
            } else {
                kairox_dfr_update_f32_kernel<true, false, 4><<<blocks, threads, 0, ctx.stream()>>>(
                    raw_sparse_data, old_scores, out_scores, ne0, ne1, args.group_size, args.n_groups,
                    args.sparse_threshold, args.lambda, args.one_minus_lambda, args.normalizer);
            }
            break;
        default:
            if (args.use_ema) {
                kairox_dfr_update_f32_kernel<true, true, 0><<<blocks, threads, 0, ctx.stream()>>>(
                    raw_sparse_data, old_scores, out_scores, ne0, ne1, args.group_size, args.n_groups,
                    args.sparse_threshold, args.lambda, args.one_minus_lambda, args.normalizer);
            } else {
                kairox_dfr_update_f32_kernel<true, false, 0><<<blocks, threads, 0, ctx.stream()>>>(
                    raw_sparse_data, old_scores, out_scores, ne0, ne1, args.group_size, args.n_groups,
                    args.sparse_threshold, args.lambda, args.one_minus_lambda, args.normalizer);
            }
            break;
    }
}

template <bool SINGLE_TOKEN>
static __launch_bounds__(kairox_dfr_mask_threads, 1) __global__ void kairox_dfr_mask_f32_kernel(
    const int32_t * __restrict__ topk_idx,
    int topk_stride,
    int k,
    int n_tokens,
    int n_groups,
    float * __restrict__ group_mask,
    float * __restrict__ load_group,
    float * __restrict__ evict_group) {
    __shared__ unsigned int curr_mask_bits[kairox_dfr_mask_words];

    for (int i = threadIdx.x; i < kairox_dfr_mask_words; i += blockDim.x) {
        curr_mask_bits[i] = 0;
    }
    __syncthreads();

    if constexpr (SINGLE_TOKEN) {
        for (int j = threadIdx.x; j < k; j += blockDim.x) {
            const int g = topk_idx[j];
            if ((unsigned) g < (unsigned) n_groups) {
                atomicOr(&curr_mask_bits[g >> 5], 1u << (g & 31));
            }
        }
    } else {
        for (int t = 0; t < n_tokens; ++t) {
            const int32_t * token_topk = topk_idx + t * topk_stride;
            for (int j = threadIdx.x; j < k; j += blockDim.x) {
                const int g = token_topk[j];
                if ((unsigned) g < (unsigned) n_groups) {
                    atomicOr(&curr_mask_bits[g >> 5], 1u << (g & 31));
                }
            }
        }
    }
    __syncthreads();

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        const unsigned int curr = (curr_mask_bits[g >> 5] >> (g & 31)) & 1u;
        const unsigned int prev = group_mask[g] != 0.0f;
        const unsigned int diff = prev ^ curr;

        load_group[g]  = float(curr & diff);
        evict_group[g] = float(prev & diff);
        group_mask[g]  = curr ? 1.0f : 0.0f;
    }
}

void ggml_cuda_op_dfr_mask(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         topk_idx,
                           float *                     group_mask,
                           ggml_tensor *               load_group,
                           ggml_tensor *               evict_group) {
    GGML_ASSERT(group_mask != nullptr);

    GGML_ASSERT(ggml_is_contiguous(topk_idx));
    GGML_ASSERT(ggml_is_contiguous(load_group));
    GGML_ASSERT(ggml_is_contiguous(evict_group));

    GGML_ASSERT(topk_idx->ne[2] == 1 && topk_idx->ne[3] == 1);
    GGML_ASSERT(load_group->ne[1] == 1 && load_group->ne[2] == 1 && load_group->ne[3] == 1);
    GGML_ASSERT(evict_group->ne[1] == 1 && evict_group->ne[2] == 1 && evict_group->ne[3] == 1);

    const int k           = (int) topk_idx->ne[0];
    const int n_tokens    = (int) topk_idx->ne[1];
    const int topk_stride = (int) (topk_idx->nb[1] / sizeof(int32_t));
    const int n_groups    = (int) load_group->ne[0];

    GGML_ASSERT(n_groups > 0 && n_groups <= 1024);
    GGML_ASSERT(evict_group->ne[0] == n_groups);

    const int32_t * topk_data        = (const int32_t *) topk_idx->data;
    float *         load_group_data  = (float *) load_group->data;
    float *         evict_group_data = (float *) evict_group->data;
    if (n_tokens == 1) {
        kairox_dfr_mask_f32_kernel<true><<<1, kairox_dfr_mask_threads, 0, ctx.stream()>>>(
            topk_data, topk_stride, k, n_tokens, n_groups, group_mask, load_group_data, evict_group_data);
    } else {
        kairox_dfr_mask_f32_kernel<false><<<1, kairox_dfr_mask_threads, 0, ctx.stream()>>>(
            topk_data, topk_stride, k, n_tokens, n_groups, group_mask, load_group_data, evict_group_data);
    }
}
