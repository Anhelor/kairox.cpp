#include "mmvq-sparse.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq,
                                  const block_q8_1 * __restrict__ bq8_1,
                                  const int & kbx,
                                  const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:
            return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q4_K:
            return vec_dot_q4_K_q8_1;
        default:
            return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:
            return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:
            return VDR_Q4_K_Q8_1_MMVQ;
        default:
            return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    return 1;
}

template <ggml_type type, int ncols_dst>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id()) * ggml_cuda_get_physical_warp_size(), 1) static
    __global__ void
    mul_mat_vec_q_sparse(const void * __restrict__ vx,
                         const void * __restrict__ vy,
                         const float * __restrict__ sparse_idx,
                         const int32_t * __restrict__ gpu_neu_idx,
                         float * __restrict__ dst,
                         const float    sparse_threshold,
                         const uint32_t ncols_x,
                         const uint3    nchannels_y,
                         const uint32_t stride_row_x,
                         const uint32_t stride_col_y,
                         const uint32_t stride_col_dst,
                         const uint3    channel_ratio,
                         const uint32_t stride_channel_x,
                         const uint32_t stride_channel_y,
                         const uint32_t stride_channel_dst,
                         const uint3    sample_ratio,
                         const uint32_t stride_sample_x,
                         const uint32_t stride_sample_y,
                         const uint32_t stride_sample_dst) {
    constexpr int                     qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int                     qi        = ggml_cuda_type_traits<type>::qi;
    constexpr int                     vdr       = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id  = get_device_table_id();
    constexpr int                     nwarps    = calc_nwarps(type, ncols_dst, table_id);
    constexpr int                     warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int     tid              = warp_size * threadIdx.y + threadIdx.x;
    const int     row              = blockIdx.x;
    const int     blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter  = vdr * nwarps * warp_size / qi;

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

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = channel_dst;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    float tmp[ncols_dst] = { 0.0f };

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y * stride_sample_y + channel_y * stride_channel_y;
    const int          kbx_offset = sample_x * stride_sample_x + channel_x * stride_channel_x + row * stride_row_x;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            if (!active[j]) {
                continue;
            }
            tmp[j] += vec_dot_q_cuda(vx, &y[j * stride_col_y + kby], kbx_offset + kbx, kqs);
        }
    }

    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            if (!active[j]) {
                continue;
            }
            tmp_shared[threadIdx.y - 1][j][threadIdx.x] = tmp[j];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row_dst;

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        if (!active[j]) {
            continue;
        }
#pragma unroll
        for (int l = 0; l < nwarps - 1; ++l) {
            tmp[j] += tmp_shared[l][j][threadIdx.x];
        }
        tmp[j] = warp_reduce_sum<warp_size>(tmp[j]);
        if (threadIdx.x == 0) {
            dst[j * stride_col_dst] = tmp[j];
        }
    }
}

template <ggml_type type>
static __forceinline__ std::pair<dim3, dim3> calc_launch_params(const int                     ncols_dst,
                                                                const int                     nrows_launch,
                                                                const int                     nchannels_dst,
                                                                const int                     nsamples_dst,
                                                                const int                     warp_size,
                                                                const mmvq_parameter_table_id table_id) {
    const dim3 block_nums(nrows_launch, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, calc_nwarps(type, ncols_dst, table_id), 1);
    return { block_nums, block_dims };
}

template <ggml_type type, int c_ncols_dst>
static void mul_mat_vec_q_sparse_switch_ncols_dst(const void *    vx,
                                                  const void *    vy,
                                                  const float *   sparse_idx,
                                                  const int32_t * gpu_neu_idx,
                                                  float *         dst,
                                                  const float     sparse_threshold,
                                                  const int       ncols_x,
                                                  const int       nrows_x,
                                                  const int       nrows_launch,
                                                  const int       ncols_dst,
                                                  const int       stride_row_x,
                                                  const int       stride_col_y,
                                                  const int       stride_col_dst,
                                                  const int       nchannels_x,
                                                  const int       nchannels_y,
                                                  const int       nchannels_dst,
                                                  const int       stride_channel_x,
                                                  const int       stride_channel_y,
                                                  const int       stride_channel_dst,
                                                  const int       nsamples_x,
                                                  const int       nsamples_dst,
                                                  const int       stride_sample_x,
                                                  const int       stride_sample_y,
                                                  const int       stride_sample_dst,
                                                  cudaStream_t    stream) {
    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);

    const uint3 nchannels_y_fd   = init_fastdiv_values(nchannels_y);
    const uint3 channel_ratio_fd = init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);

    const int                     device    = ggml_cuda_get_device();
    const int                     warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(ggml_cuda_info().devices[device].cc);

    GGML_UNUSED_VARS(nrows_x, ncols_dst);
    std::pair<dim3, dim3> dims =
        calc_launch_params<type>(c_ncols_dst, nrows_launch, nchannels_dst, nsamples_dst, warp_size, table_id);
    mul_mat_vec_q_sparse<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>(
        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y,
        stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
        stride_sample_x, stride_sample_y, stride_sample_dst);
}

static void mul_mat_vec_q_sparse_switch_type(const void *    vx,
                                             const ggml_type type_x,
                                             const void *    vy,
                                             const float *   sparse_idx,
                                             const int32_t * gpu_neu_idx,
                                             float *         dst,
                                             const float     sparse_threshold,
                                             const int       ncols_x,
                                             const int       nrows_x,
                                             const int       nrows_launch,
                                             const int       ncols_dst,
                                             const int       stride_row_x,
                                             const int       stride_col_y,
                                             const int       stride_col_dst,
                                             const int       nchannels_x,
                                             const int       nchannels_y,
                                             const int       nchannels_dst,
                                             const int       stride_channel_x,
                                             const int       stride_channel_y,
                                             const int       stride_channel_dst,
                                             const int       nsamples_x,
                                             const int       nsamples_dst,
                                             const int       stride_sample_x,
                                             const int       stride_sample_y,
                                             const int       stride_sample_dst,
                                             cudaStream_t    stream) {
    switch (type_x) {
        case GGML_TYPE_Q8_0:
            switch (ncols_dst) {
                case 1:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q8_0, 1>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 2:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q8_0, 2>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 3:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q8_0, 3>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 4:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q8_0, 4>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 5:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q8_0, 5>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 6:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q8_0, 6>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 7:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q8_0, 7>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 8:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q8_0, 8>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_TYPE_Q4_K:
            switch (ncols_dst) {
                case 1:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q4_K, 1>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 2:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q4_K, 2>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 3:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q4_K, 3>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 4:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q4_K, 4>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 5:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q4_K, 5>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 6:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q4_K, 6>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 7:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q4_K, 7>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                case 8:
                    mul_mat_vec_q_sparse_switch_ncols_dst<GGML_TYPE_Q4_K, 8>(
                        vx, vy, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols_x, nrows_x, nrows_launch,
                        ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                        stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                        stride_sample_x, stride_sample_y, stride_sample_dst, stream);
                    break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q_sparse(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor *         src0,
                                    const ggml_tensor *         src1,
                                    ggml_tensor *               dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_K);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(nb00 == ts_src0);
    GGML_ASSERT(nb10 == ts_src1);
    GGML_ASSERT(nb0 == ts_dst);

    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) dst->data;

    GGML_ASSERT(dst->src[2] && dst->src[2]->data && "missing sparse_idx");
    const float *   sparse_idx       = (const float *) dst->src[2]->data;
    const int32_t * gpu_neu_idx      = dst->src[3] ? (const int32_t *) dst->src[3]->data : nullptr;
    const int64_t   nrows_launch     = gpu_neu_idx ? dst->src[3]->ne[0] : dst->ne[0];
    const float     sparse_threshold = ggml_get_op_params_f32(dst, 1);
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), ctx.stream()));

    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, ctx.stream()));
        }
    }

    const int64_t              ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13 * ne12 * ne11 * ne10_padded * sizeof(block_q8_1) / QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11,
                               ne12, ne13, ctx.stream());
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  = dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  = dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  = dst->nb[3] / ts_dst;

    const int64_t s12 = ne11 * s11;
    const int64_t s13 = ne12 * s12;

    const int64_t ncols_dst          = ne1;
    const int64_t nchannels_y        = ne12;
    const int64_t nchannels_dst      = ne2;
    const int64_t stride_col_dst     = s1;
    const int64_t stride_col_y       = s11;
    const int64_t stride_channel_dst = s2;
    const int64_t stride_channel_y   = s12;

    mul_mat_vec_q_sparse_switch_type(src0->data, src0->type, src1_q8_1.get(), sparse_idx, gpu_neu_idx, dst_d,
                                     sparse_threshold, ne00, ne01, nrows_launch, ncols_dst, s01, stride_col_y,
                                     stride_col_dst, ne02, nchannels_y, nchannels_dst, s02, stride_channel_y,
                                     stride_channel_dst, ne03, ne3, s03, s13, s3, ctx.stream());
}
