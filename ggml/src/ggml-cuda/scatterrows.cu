#include "scatterrows.cuh"

template <int nvec> struct scatter_rows_vec_type;

template <> struct scatter_rows_vec_type<1> {
    using type = float;
};

template <> struct scatter_rows_vec_type<2> {
    using type = float2;
};

template <> struct scatter_rows_vec_type<4> {
    using type = float4;
};

template <int nvec> static inline bool scatter_rows_aligned_for(const void * p) {
    return ((uintptr_t) p % (sizeof(float) * nvec)) == 0;
}

template <int nvec>
static __global__ void scatter_rows_f32(const float * __restrict__ src,
                                        float * __restrict__ dst,
                                        const int32_t * __restrict__ gpu_neu_idx,
                                        const int ne00,
                                        const int ne01,
                                        const int ne0) {
    using vec_t = typename scatter_rows_vec_type<nvec>::type;

    const int i1     = (int) blockIdx.y;
    const int vec_i0 = (int) blockIdx.x * blockDim.x + threadIdx.x;
    const int n_vec0 = ne00 / nvec;

    if (i1 >= ne01 || vec_i0 >= n_vec0) {
        return;
    }

    const int i0      = vec_i0 * nvec;
    const int row_src = i1 * ne00;
    const int row_dst = i1 * ne0;
    const int src_idx = row_src + i0;

    vec_t v = *reinterpret_cast<const vec_t *>(src + src_idx);

    if constexpr (nvec == 1) {
        dst[row_dst + gpu_neu_idx[i0]] = v;
    } else if constexpr (nvec == 2) {
        const int2 idx       = *reinterpret_cast<const int2 *>(gpu_neu_idx + i0);
        dst[row_dst + idx.x] = v.x;
        dst[row_dst + idx.y] = v.y;
    } else {
        const int4 idx       = *reinterpret_cast<const int4 *>(gpu_neu_idx + i0);
        dst[row_dst + idx.x] = v.x;
        dst[row_dst + idx.y] = v.y;
        dst[row_dst + idx.z] = v.z;
        dst[row_dst + idx.w] = v.w;
    }
}

static __global__ void scatter_rows_f32_tail(const float * __restrict__ src,
                                             float * __restrict__ dst,
                                             const int32_t * __restrict__ gpu_neu_idx,
                                             const int ne00,
                                             const int ne01,
                                             const int ne0,
                                             const int i0_base) {
    const int i1 = (int) blockIdx.y;
    const int i0 = i0_base + (int) blockIdx.x * blockDim.x + threadIdx.x;

    if (i1 >= ne01 || i0 >= ne00) {
        return;
    }

    const int row_src = i1 * ne00;
    const int row_dst = i1 * ne0;

    dst[row_dst + gpu_neu_idx[i0]] = src[row_src + i0];
}

template <int nvec>
static __forceinline__ void scatter_rows_f32_cuda_vec(const float *   src,
                                                      float *         dst,
                                                      const int32_t * gpu_neu_idx,
                                                      const int       ne00,
                                                      const int       ne01,
                                                      const int       ne0,
                                                      cudaStream_t    stream) {
    constexpr int block_size = 256;

    const int n_vec0  = ne00 / nvec;
    const int i0_base = n_vec0 * nvec;

    if (n_vec0 > 0) {
        const dim3 block_dims(block_size, 1, 1);
        const dim3 block_nums((n_vec0 + block_size - 1) / block_size, ne01, 1);
        scatter_rows_f32<nvec><<<block_nums, block_dims, 0, stream>>>(src, dst, gpu_neu_idx, ne00, ne01, ne0);
    }

    if (i0_base < ne00) {
        const dim3 block_dims(block_size, 1, 1);
        const dim3 block_nums((ne00 - i0_base + block_size - 1) / block_size, ne01, 1);
        scatter_rows_f32_tail<<<block_nums, block_dims, 0, stream>>>(src, dst, gpu_neu_idx, ne00, ne01, ne0, i0_base);
    }
}

static __forceinline__ void scatter_rows_f32_cuda(const float *   src,
                                                  float *         dst,
                                                  const int32_t * gpu_neu_idx,
                                                  const int       ne00,
                                                  const int       ne01,
                                                  const int       ne0,
                                                  cudaStream_t    stream) {
    if ((ne00 % 4 == 0) && scatter_rows_aligned_for<4>(src) && scatter_rows_aligned_for<4>(gpu_neu_idx)) {
        scatter_rows_f32_cuda_vec<4>(src, dst, gpu_neu_idx, ne00, ne01, ne0, stream);
        return;
    }

    if ((ne00 % 2 == 0) && scatter_rows_aligned_for<2>(src) && scatter_rows_aligned_for<2>(gpu_neu_idx)) {
        scatter_rows_f32_cuda_vec<2>(src, dst, gpu_neu_idx, ne00, ne01, ne0, stream);
        return;
    }

    scatter_rows_f32_cuda_vec<1>(src, dst, gpu_neu_idx, ne00, ne01, ne0, stream);
}

void ggml_cuda_op_scatter_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0 != nullptr && src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1 != nullptr && src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne0  = dst->ne[0];
    const int64_t ne1  = dst->ne[1];

    GGML_ASSERT(ne10 == ne00);
    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne0 >= ne00);

    const float *   src0_d = (const float *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float *         dst_d  = (float *) dst->data;

    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), ctx.stream()));
    scatter_rows_f32_cuda(src0_d, dst_d, src1_d, (int) ne00, (int) ne01, (int) ne0, ctx.stream());
}
