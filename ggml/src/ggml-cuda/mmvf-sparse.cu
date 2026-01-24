#include "ggml.h"
#include "mmvf-sparse.cuh"

template <typename T, typename type_acc, int ncols_dst, int block_size>
static __global__ void mul_mat_vec_f_sparse(const T * __restrict__ x,
                                            const float * __restrict__ y,
                                            const float * __restrict__ sparse_idx,
                                            const int32_t * __restrict__ gpu_neu_idx,
                                            float * __restrict__ dst,
                                            const float sparse_threshold,
                                            const int   ncols2,
                                            const int   stride_row,
                                            const int   stride_col_y2,
                                            const int   stride_col_dst,
                                            const int   stride_channel_x,
                                            const int   stride_channel_y,
                                            const int   stride_channel_dst,
                                            const int   stride_sample_x,
                                            const int   stride_sample_y,
                                            const int   stride_sample_dst) {
    const int row         = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;
    const int tid         = threadIdx.x;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int row_dst = gpu_neu_idx ? gpu_neu_idx[row] : row;

    bool any_active = 0;
    bool active[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        active[j] = (sparse_idx[j * stride_col_dst + row_dst] >= sparse_threshold);
        any_active |= active[j];
    }
    if (!any_active) {
        return;
    }

    x += int64_t(sample_dst) * stride_sample_x + channel_dst * stride_channel_x + row * stride_row;
    y += int64_t(sample_dst) * stride_sample_y + channel_dst * stride_channel_y;
    dst += int64_t(sample_dst) * stride_sample_dst + channel_dst * stride_channel_dst;

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float *                buf_iw = (float *) data_mmv;

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = { 0.0f };

    if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                if (!active[j]) {
                    continue;
                }
                const float2 tmpy = y2[j * stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
            }
        }
    } else if constexpr (std::is_same_v<T, half>) {
        const half2 * x2 = (const half2 *) x;
        if (std::is_same_v<type_acc, float>) {
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    if (!active[j]) {
                        continue;
                    }
                    const float2 tmpy = y2[j * stride_col_y2 + col2];
                    ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
                }
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2[ncols_dst] = {
                { 0.0f, 0.0f }
            };
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const half2 tmpx = x2[col2];
#    pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    if (!active[j]) {
                        continue;
                    }
                    const float2 tmpy = y2[j * stride_col_y2 + col2];
                    sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);
                }
            }
#    pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                if (!active[j]) {
                    continue;
                }
                sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
            }
#else
            NO_DEVICE_CODE;
#endif  // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
#if defined(GGML_USE_HIP)
        const int * x2 = (const int *) x;
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const int tmpx = x2[col2];
#    pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                if (!active[j]) {
                    continue;
                }
                const float2 tmpy  = y2[j * stride_col_y2 + col2];
                const float  tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
                const float  tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
                ggml_cuda_mad(sumf[j], tmpx0, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx1, tmpy.y);
            }
        }
#else
        const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const nv_bfloat162 tmpx = x2[col2];
#    pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                if (!active[j]) {
                    continue;
                }
                const float2 tmpy = y2[j * stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
            }
        }
#endif
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if (block_size > warp_size) {
            buf_iw[tid / warp_size] = sumf[j];
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            }

            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    if (active[tid]) {
        dst[tid * stride_col_dst + row_dst] = sumf[tid];
    }
}

template <typename T, typename type_acc, int ncols_dst, int block_size>
static __forceinline__ void mul_mat_vec_f_sparse_launch(const T *          x,
                                                        const float *      y,
                                                        const float *      sparse_idx,
                                                        const int32_t *    gpu_neu_idx,
                                                        float *            dst,
                                                        const float        sparse_threshold,
                                                        const int64_t      ncols,
                                                        const int64_t      stride_row,
                                                        const int64_t      stride_col_y,
                                                        const int64_t      stride_col_dst,
                                                        const int          stride_channel_x,
                                                        const int          stride_channel_y,
                                                        const int          stride_channel_dst,
                                                        const int          stride_sample_x,
                                                        const int          stride_sample_y,
                                                        const int          stride_sample_dst,
                                                        const dim3 &       block_dims,
                                                        const dim3 &       block_nums,
                                                        const int          nbytes_shared,
                                                        const cudaStream_t stream) {
    mul_mat_vec_f_sparse<T, type_acc, ncols_dst, block_size><<<block_nums, block_dims, nbytes_shared, stream>>>(
        x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, stride_row, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst, stride_sample_x, stride_sample_y, stride_sample_dst);
}

template <typename T, typename type_acc, int ncols_dst>
static void launch_mul_mat_vec_f_sparse_cuda(const T *       x,
                                             const float *   y,
                                             const float *   sparse_idx,
                                             const int32_t * gpu_neu_idx,
                                             float *         dst,
                                             const float     sparse_threshold,
                                             const int64_t   ncols,
                                             const int64_t   nrows,
                                             const int64_t   nrows_launch,
                                             const int64_t   stride_row,
                                             const int64_t   stride_col_y,
                                             const int64_t   stride_col_dst,
                                             const int64_t   nchannels_dst,
                                             const int64_t   stride_channel_x,
                                             const int64_t   stride_channel_y,
                                             const int64_t   stride_channel_dst,
                                             const int64_t   nsamples_dst,
                                             const int64_t   stride_sample_x,
                                             const int64_t   stride_sample_y,
                                             const int64_t   stride_sample_dst,
                                             cudaStream_t    stream) {
    GGML_ASSERT(ncols % 2 == 0);
    GGML_ASSERT(stride_row % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);

    const int device    = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2 * warp_size - 1) / (2 * warp_size);
    int64_t max_block_size  = 256;
    if (ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD &&
        ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2 * warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2 * block_size - 1) / (2 * block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    GGML_UNUSED(nrows);
    const int  nbytes_shared = warp_size * sizeof(float);
    const dim3 block_nums(nrows_launch, nchannels_dst, nsamples_dst);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
        case 32:
            mul_mat_vec_f_sparse_launch<T, type_acc, ncols_dst, 32>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols / 2, stride_row, stride_col_y / 2,
                stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, stride_sample_x,
                stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
            break;
        case 64:
            mul_mat_vec_f_sparse_launch<T, type_acc, ncols_dst, 64>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols / 2, stride_row, stride_col_y / 2,
                stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, stride_sample_x,
                stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
            break;
        case 96:
            mul_mat_vec_f_sparse_launch<T, type_acc, ncols_dst, 96>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols / 2, stride_row, stride_col_y / 2,
                stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, stride_sample_x,
                stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
            break;
        case 128:
            mul_mat_vec_f_sparse_launch<T, type_acc, ncols_dst, 128>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols / 2, stride_row, stride_col_y / 2,
                stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, stride_sample_x,
                stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
            break;
        case 160:
            mul_mat_vec_f_sparse_launch<T, type_acc, ncols_dst, 160>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols / 2, stride_row, stride_col_y / 2,
                stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, stride_sample_x,
                stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
            break;
        case 192:
            mul_mat_vec_f_sparse_launch<T, type_acc, ncols_dst, 192>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols / 2, stride_row, stride_col_y / 2,
                stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, stride_sample_x,
                stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
            break;
        case 224:
            mul_mat_vec_f_sparse_launch<T, type_acc, ncols_dst, 224>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols / 2, stride_row, stride_col_y / 2,
                stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, stride_sample_x,
                stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
            break;
        case 256:
            mul_mat_vec_f_sparse_launch<T, type_acc, ncols_dst, 256>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols / 2, stride_row, stride_col_y / 2,
                stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, stride_sample_x,
                stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template <typename T, typename type_acc>
static void mul_mat_vec_f_sparse_cuda_switch_ncols_dst(const T *       x,
                                                       const float *   y,
                                                       const float *   sparse_idx,
                                                       const int32_t * gpu_neu_idx,
                                                       float *         dst,
                                                       const float     sparse_threshold,
                                                       const int64_t   ncols,
                                                       const int64_t   nrows,
                                                       const int64_t   nrows_launch,
                                                       const int64_t   ncols_dst,
                                                       const int64_t   stride_row,
                                                       const int64_t   stride_col_y,
                                                       const int64_t   stride_col_dst,
                                                       const int64_t   nchannels_dst,
                                                       const int64_t   stride_channel_x,
                                                       const int64_t   stride_channel_y,
                                                       const int64_t   stride_channel_dst,
                                                       const int64_t   nsamples_dst,
                                                       const int64_t   stride_sample_x,
                                                       const int64_t   stride_sample_y,
                                                       const int64_t   stride_sample_dst,
                                                       cudaStream_t    stream) {
    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_f_sparse_cuda<T, type_acc, 1>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch, stride_row,
                stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            break;
        case 2:
            launch_mul_mat_vec_f_sparse_cuda<T, type_acc, 2>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch, stride_row,
                stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            break;
        case 3:
            launch_mul_mat_vec_f_sparse_cuda<T, type_acc, 3>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch, stride_row,
                stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            break;
        case 4:
            launch_mul_mat_vec_f_sparse_cuda<T, type_acc, 4>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch, stride_row,
                stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            break;
        case 5:
            launch_mul_mat_vec_f_sparse_cuda<T, type_acc, 5>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch, stride_row,
                stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            break;
        case 6:
            launch_mul_mat_vec_f_sparse_cuda<T, type_acc, 6>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch, stride_row,
                stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            break;
        case 7:
            launch_mul_mat_vec_f_sparse_cuda<T, type_acc, 7>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch, stride_row,
                stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            break;
        case 8:
            launch_mul_mat_vec_f_sparse_cuda<T, type_acc, 8>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch, stride_row,
                stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template <typename T>
static void mul_mat_vec_f_sparse_cuda(const T *       x,
                                      const float *   y,
                                      const float *   sparse_idx,
                                      const int32_t * gpu_neu_idx,
                                      float *         dst,
                                      const float     sparse_threshold,
                                      const int64_t   ncols,
                                      const int64_t   nrows,
                                      const int64_t   nrows_launch,
                                      const int64_t   ncols_dst,
                                      const int64_t   stride_row,
                                      const int64_t   stride_col_y,
                                      const int       stride_col_dst,
                                      const int64_t   nchannels_dst,
                                      const int64_t   stride_channel_x,
                                      const int64_t   stride_channel_y,
                                      const int64_t   stride_channel_dst,
                                      const int64_t   nsamples_dst,
                                      const int64_t   stride_sample_x,
                                      const int64_t   stride_sample_y,
                                      const int64_t   stride_sample_dst,
                                      enum ggml_prec  prec,
                                      cudaStream_t    stream) {
    if constexpr (std::is_same_v<T, half>) {
        if (prec == GGML_PREC_DEFAULT) {
            mul_mat_vec_f_sparse_cuda_switch_ncols_dst<T, half>(
                x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch, ncols_dst, stride_row,
                stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
            return;
        }
    }
    mul_mat_vec_f_sparse_cuda_switch_ncols_dst<T, float>(
        x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch, ncols_dst, stride_row,
        stride_col_y, stride_col_dst, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
        nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
}

void ggml_cuda_mul_mat_vec_f_sparse(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor *         src0,
                                    const ggml_tensor *         src1,
                                    ggml_tensor *               dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(nb00 == ts_src0);
    GGML_ASSERT(nb10 == ts_src1);
    GGML_ASSERT(nb0 == ts_dst);

    const int            cc   = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) dst->data;

    GGML_ASSERT(dst->src[2] && dst->src[2]->data && "missing sparse_idx");
    const float *   sparse_idx       = (const float *) dst->src[2]->data;
    const int32_t * gpu_neu_idx      = dst->src[3] ? (const int32_t *) dst->src[3]->data : nullptr;
    const int64_t   nrows_launch     = gpu_neu_idx ? dst->src[3]->ne[0] : dst->ne[0];
    const float     sparse_threshold = ggml_get_op_params_f32(dst, 1);
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), ctx.stream()));

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  = dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  = dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  = dst->nb[3] / ts_dst;

    const int64_t ncols_dst          = ne1;
    const int64_t nchannels_dst      = ne2;
    const int64_t stride_col_dst     = s1;
    const int64_t stride_col_y       = s11;
    const int64_t stride_channel_dst = s2;
    const int64_t stride_channel_y   = s12;

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                const float * src0_d = (const float *) src0->data;
                mul_mat_vec_f_sparse_cuda(src0_d, src1_d, sparse_idx, gpu_neu_idx, dst_d, sparse_threshold, ne00, ne01,
                                          nrows_launch, ncols_dst, s01, stride_col_y, stride_col_dst, nchannels_dst,
                                          s02, stride_channel_y, stride_channel_dst, ne3, s03, s13, s3, prec,
                                          ctx.stream());
            }
            break;
        case GGML_TYPE_F16:
            {
                const half * src0_d = (const half *) src0->data;
                mul_mat_vec_f_sparse_cuda(src0_d, src1_d, sparse_idx, gpu_neu_idx, dst_d, sparse_threshold, ne00, ne01,
                                          nrows_launch, ncols_dst, s01, stride_col_y, stride_col_dst, nchannels_dst,
                                          s02, stride_channel_y, stride_channel_dst, ne3, s03, s13, s3, prec,
                                          ctx.stream());
            }
            break;
        case GGML_TYPE_BF16:
            {
                const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
                mul_mat_vec_f_sparse_cuda(src0_d, src1_d, sparse_idx, gpu_neu_idx, dst_d, sparse_threshold, ne00, ne01,
                                          nrows_launch, ncols_dst, s01, stride_col_y, stride_col_dst, nchannels_dst,
                                          s02, stride_channel_y, stride_channel_dst, ne3, s03, s13, s3, prec,
                                          ctx.stream());
            }
            break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}
