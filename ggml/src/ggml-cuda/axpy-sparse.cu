#include "axpy-sparse.cuh"
#include "ggml.h"

template <typename T, int block_k, int vals_per_thread>
static __global__ void axpy_vec_f_sparse(const T * __restrict__ x,
                                         const float * __restrict__ y,
                                         const float * __restrict__ sparse_idx,
                                         const int32_t * __restrict__ gpu_neu_idx,
                                         float * __restrict__ dst,
                                         const float sparse_threshold,
                                         const int   ncols,
                                         const int   nrows,
                                         const int   nrows_launch) {
    const int     tid     = threadIdx.x;
    constexpr int block_n = WARP_SIZE * vals_per_thread;
    const int     col0    = blockIdx.x * block_n;
    const int     row0    = blockIdx.y * block_k;

    float sum[vals_per_thread] = { 0.0f };

#pragma unroll
    for (int i = 0; i < block_k; ++i) {
        const int row = row0 + i;
        if (row >= nrows_launch) {
            break;
        }
        const int row_dst = gpu_neu_idx ? gpu_neu_idx[row] : row;
        if (row_dst < 0 || row_dst >= nrows) {
            continue;
        }
        const float alpha = y[row_dst];
        if (alpha == 0.0f) {
            continue;
        }
        if (sparse_idx[row_dst] < sparse_threshold) {
            continue;
        }

        const T * row_x = x + row * ncols;
#pragma unroll
        for (int j = 0; j < vals_per_thread; ++j) {
            const int col = col0 + j * WARP_SIZE + tid;
            if (col < ncols) {
                float val;
                if constexpr (std::is_same_v<T, half>) {
                    val = __half2float(row_x[col]);
                } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
                    val = __bfloat162float(row_x[col]);
                } else {
                    val = row_x[col];
                }
                sum[j] += val * alpha;
            }
        }
    }

#pragma unroll
    for (int j = 0; j < vals_per_thread; ++j) {
        const int col = col0 + j * WARP_SIZE + tid;
        if (col < ncols && sum[j]) {
            atomicAdd(&dst[col], sum[j]);
        }
    }
}

template <typename T, int block_k, int ncols_dst, int block_size>
static __global__ void axpy_batch_f_sparse(const T * __restrict__ x,
                                           const float * __restrict__ y,
                                           const float * __restrict__ sparse_idx,
                                           const int32_t * __restrict__ gpu_neu_idx,
                                           float * __restrict__ dst,
                                           const float sparse_threshold,
                                           const int   ncols,
                                           const int   nrows,
                                           const int   ncols_y,
                                           const int   nrows_launch) {
    const int     lane    = threadIdx.x;
    const int     warp_id = threadIdx.y;
    constexpr int block_n = WARP_SIZE * ncols_dst;
    const int     col0    = blockIdx.x * block_n;
    const int     col_y0  = blockIdx.y * block_size;
    const int     row0    = blockIdx.z * block_k;
    const int     col_y   = col_y0 + warp_id;

    extern __shared__ unsigned char data_mmv[];
    T *                             buf_x = reinterpret_cast<T *>(data_mmv);

    const int tid        = warp_id * WARP_SIZE + lane;
    const int nt         = block_size * WARP_SIZE;
    const int tile_elems = block_k * block_n;

    for (int idx = tid; idx < tile_elems; idx += nt) {
        const int ik   = idx / block_n;
        const int icol = idx - ik * block_n;
        const int row  = row0 + ik;
        const int col  = col0 + icol;
        if (row < nrows_launch && col < ncols) {
            buf_x[idx] = x[row * ncols + col];
        } else {
            buf_x[idx] = T{};
        }
    }
    __syncthreads();
    if (col_y >= ncols_y) {
        return;
    }

    y += col_y * nrows;
    dst += col_y * ncols;
    sparse_idx += col_y * nrows;

    float sum[ncols_dst] = { 0.0f };

#pragma unroll
    for (int i = 0; i < block_k; ++i) {
        const int row = row0 + i;
        if (row >= nrows_launch) {
            break;
        }
        const int row_dst = gpu_neu_idx ? gpu_neu_idx[row] : row;
        if (row_dst < 0 || row_dst >= nrows) {
            continue;
        }
        float alpha = 0.0f;
        if (lane == 0) {
            alpha = y[row_dst];
        }
        alpha = __shfl_sync(0xffffffff, alpha, 0);
        if (alpha == 0.0f) {
            continue;
        }
        if (sparse_idx[row_dst] < sparse_threshold) {
            continue;
        }

        const T * row_x = buf_x + i * block_n;
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const int col = j * WARP_SIZE + lane;
            float     val;
            if constexpr (std::is_same_v<T, half>) {
                val = __half2float(row_x[col]);
            } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
                val = __bfloat162float(row_x[col]);
            } else {
                val = row_x[col];
            }
            sum[j] += val * alpha;
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        const int col = col0 + j * WARP_SIZE + lane;
        if (col < ncols && sum[j]) {
            atomicAdd(&dst[col], sum[j]);
        }
    }
}

template <typename T>
static void launch_axpy_vec_f_sparse_cuda(const T *       x,
                                          const float *   y,
                                          const float *   sparse_idx,
                                          const int32_t * gpu_neu_idx,
                                          float *         dst,
                                          const float     sparse_threshold,
                                          const int64_t   ncols,
                                          const int64_t   nrows,
                                          const int64_t   nrows_launch,
                                          cudaStream_t    stream) {
    constexpr int block_k         = 32;
    constexpr int vals_per_thread = 4;
    constexpr int block_n         = WARP_SIZE * vals_per_thread;

    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums((ncols + block_n - 1) / block_n, (nrows_launch + block_k - 1) / block_k, 1);

    axpy_vec_f_sparse<T, block_k, vals_per_thread><<<block_nums, block_dims, 0, stream>>>(
        x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch);
}

template <typename T>
static void launch_axpy_batch_f_sparse_cuda(const T *       x,
                                            const float *   y,
                                            const float *   sparse_idx,
                                            const int32_t * gpu_neu_idx,
                                            float *         dst,
                                            const float     sparse_threshold,
                                            const int64_t   ncols,
                                            const int64_t   nrows,
                                            const int64_t   ncols_y,
                                            const int64_t   nrows_launch,
                                            cudaStream_t    stream) {
    constexpr int block_k    = 16;
    constexpr int ncols_dst  = 8;
    constexpr int block_size = 8;
    constexpr int block_n    = WARP_SIZE * ncols_dst;

    const dim3 block_dims(WARP_SIZE, block_size, 1);
    const dim3 block_nums((ncols + block_n - 1) / block_n, (ncols_y + block_size - 1) / block_size,
                          (nrows_launch + block_k - 1) / block_k);

    const int nbytes_shared = block_k * block_n * sizeof(T);
    axpy_batch_f_sparse<T, block_k, ncols_dst, block_size><<<block_nums, block_dims, nbytes_shared, stream>>>(
        x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, ncols_y, nrows_launch);
}

template <typename T>
static void axpy_f_sparse_cuda(const T *       x,
                               const float *   y,
                               const float *   sparse_idx,
                               const int32_t * gpu_neu_idx,
                               float *         dst,
                               const float     sparse_threshold,
                               const int64_t   ncols,
                               const int64_t   nrows,
                               const int64_t   ncols_y,
                               const int64_t   nrows_launch,
                               enum ggml_prec  prec,
                               cudaStream_t    stream) {
    GGML_UNUSED(prec);
    if (ncols_y == 1) {
        launch_axpy_vec_f_sparse_cuda(x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, nrows_launch,
                                      stream);
        return;
    }
    launch_axpy_batch_f_sparse_cuda(x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, ncols_y,
                                    nrows_launch, stream);
}

void ggml_cuda_axpy_f_sparse(ggml_backend_cuda_context & ctx,
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

    GGML_ASSERT(src0->nb[1] == ne00 * ts_src0);
    GGML_ASSERT(src1->nb[1] == ne10 * ts_src1);
    GGML_ASSERT(dst->nb[1] == ne0 * ts_dst);

    const int            cc   = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) dst->data;

    GGML_ASSERT(dst->src[2] && dst->src[2]->data && "missing sparse_idx");
    const float *   sparse_idx       = (const float *) dst->src[2]->data;
    const int32_t * gpu_neu_idx      = dst->src[3] ? (const int32_t *) dst->src[3]->data : nullptr;
    const int64_t   nrows_launch     = gpu_neu_idx ? dst->src[3]->ne[0] : src1->ne[0];
    const float     sparse_threshold = ggml_get_op_params_f32(dst, 1);
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), ctx.stream()));

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                const float * src0_d = (const float *) src0->data;
                axpy_f_sparse_cuda(src0_d, src1_d, sparse_idx, gpu_neu_idx, dst_d, sparse_threshold, ne00, ne10, ne11,
                                   nrows_launch, prec, ctx.stream());
            }
            break;
        case GGML_TYPE_F16:
            {
                const half * src0_d = (const half *) src0->data;
                axpy_f_sparse_cuda(src0_d, src1_d, sparse_idx, gpu_neu_idx, dst_d, sparse_threshold, ne00, ne10, ne11,
                                   nrows_launch, prec, ctx.stream());
            }
            break;
        case GGML_TYPE_BF16:
            {
                const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
                axpy_f_sparse_cuda(src0_d, src1_d, sparse_idx, gpu_neu_idx, dst_d, sparse_threshold, ne00, ne10, ne11,
                                   nrows_launch, prec, ctx.stream());
            }
            break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}
