#include "axpyq-sparse.cuh"
#include "ggml.h"

static __device__ __forceinline__ void get_scale_min_k4_device(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static __device__ __forceinline__ float q8_0_get_f32(const block_q8_0 * bx, int idx, float alpha) {
    const float scale = alpha * __half2float(bx->d);
    return scale * (float) bx->qs[idx];
}

static __device__ __forceinline__ void q4_K_unpack_meta(const block_q4_K * __restrict__ bx,
                                                        float (&d8)[8],
                                                        float (&m8)[8]) {
    const float dall = __half2float(__low2half(bx->dm));
    const float dmin = __half2float(__high2half(bx->dm));
#pragma unroll
    for (int is = 0; is < 8; ++is) {
        uint8_t sc, m;
        get_scale_min_k4_device(is, bx->scales, sc, m);
        d8[is] = dall * (float) sc;
        m8[is] = dmin * (float) m;
    }
}

template <int block_qb>
static __device__ __forceinline__ void axpy_q4_K_block(const block_q4_K * __restrict__ bx,
                                                       float alpha,
                                                       int   col0,
                                                       int   ncols,
                                                       float (&sum)[block_qb][8],
                                                       int qb_local) {
    static_assert(QK_K == 256, "Q4_K fast path assumes QK_K == 256");

    const int lane      = threadIdx.x;
    const int ir        = lane >> 2;
    const int l         = lane & 3;
    const int base_byte = 4 * ir + l;

    const uint8_t q0 = bx->qs[0 + base_byte];
    const uint8_t q1 = bx->qs[32 + base_byte];
    const uint8_t q2 = bx->qs[64 + base_byte];
    const uint8_t q3 = bx->qs[96 + base_byte];

    float d8[8], m8[8];
    q4_K_unpack_meta(bx, d8, m8);

    const int c0 = col0 + lane;
    const int c1 = col0 + lane + 32;
    const int c2 = col0 + lane + 64;
    const int c3 = col0 + lane + 96;
    const int c4 = col0 + lane + 128;
    const int c5 = col0 + lane + 160;
    const int c6 = col0 + lane + 192;
    const int c7 = col0 + lane + 224;

    if (c0 < ncols) {
        sum[qb_local][0] += alpha * (d8[0] * (float) (q0 & 0xF) - m8[0]);
    }
    if (c1 < ncols) {
        sum[qb_local][1] += alpha * (d8[1] * (float) ((q0 >> 4) & 0xF) - m8[1]);
    }
    if (c2 < ncols) {
        sum[qb_local][2] += alpha * (d8[2] * (float) (q1 & 0xF) - m8[2]);
    }
    if (c3 < ncols) {
        sum[qb_local][3] += alpha * (d8[3] * (float) ((q1 >> 4) & 0xF) - m8[3]);
    }
    if (c4 < ncols) {
        sum[qb_local][4] += alpha * (d8[4] * (float) (q2 & 0xF) - m8[4]);
    }
    if (c5 < ncols) {
        sum[qb_local][5] += alpha * (d8[5] * (float) ((q2 >> 4) & 0xF) - m8[5]);
    }
    if (c6 < ncols) {
        sum[qb_local][6] += alpha * (d8[6] * (float) (q3 & 0xF) - m8[6]);
    }
    if (c7 < ncols) {
        sum[qb_local][7] += alpha * (d8[7] * (float) ((q3 >> 4) & 0xF) - m8[7]);
    }
}

template <int block_qb>
static __device__ __forceinline__ void axpy_q6_K_block(const block_q6_K * __restrict__ bx,
                                                       float alpha,
                                                       int   col0,
                                                       int   ncols,
                                                       float (&sum)[block_qb][8],
                                                       int qb_local) {
    static_assert(QK_K == 256, "Q6_K fast path assumes QK_K == 256");

    const int lane    = threadIdx.x;
    const int lane_hi = lane >> 4;

    const float d = (float) bx->d;
    const float s = alpha * d;

    const uint8_t   qh0 = bx->qh[lane];
    const uint8_t   qh1 = bx->qh[32 + lane];
    const int8_t *  sc0 = bx->scales + lane_hi;
    const int8_t *  sc1 = bx->scales + 8 + lane_hi;
    const uint8_t * ql0 = bx->ql + lane;
    const uint8_t * ql1 = bx->ql + 64 + lane;

    const int c0 = col0 + lane;
    const int c1 = col0 + lane + 32;
    const int c2 = col0 + lane + 64;
    const int c3 = col0 + lane + 96;
    const int c4 = col0 + lane + 128;
    const int c5 = col0 + lane + 160;
    const int c6 = col0 + lane + 192;
    const int c7 = col0 + lane + 224;

    const int qv0 = ((ql0[0] & 0xF) | (((qh0 >> 0) & 3) << 4)) - 32;
    const int qv1 = ((ql0[32] & 0xF) | (((qh0 >> 2) & 3) << 4)) - 32;
    const int qv2 = ((ql0[0] >> 4) | (((qh0 >> 4) & 3) << 4)) - 32;
    const int qv3 = ((ql0[32] >> 4) | (((qh0 >> 6) & 3) << 4)) - 32;
    const int qv4 = ((ql1[0] & 0xF) | (((qh1 >> 0) & 3) << 4)) - 32;
    const int qv5 = ((ql1[32] & 0xF) | (((qh1 >> 2) & 3) << 4)) - 32;
    const int qv6 = ((ql1[0] >> 4) | (((qh1 >> 4) & 3) << 4)) - 32;
    const int qv7 = ((ql1[32] >> 4) | (((qh1 >> 6) & 3) << 4)) - 32;

    if (c0 < ncols) {
        sum[qb_local][0] += s * (float) sc0[0] * (float) qv0;
    }
    if (c1 < ncols) {
        sum[qb_local][1] += s * (float) sc0[2] * (float) qv1;
    }
    if (c2 < ncols) {
        sum[qb_local][2] += s * (float) sc0[4] * (float) qv2;
    }
    if (c3 < ncols) {
        sum[qb_local][3] += s * (float) sc0[6] * (float) qv3;
    }
    if (c4 < ncols) {
        sum[qb_local][4] += s * (float) sc1[0] * (float) qv4;
    }
    if (c5 < ncols) {
        sum[qb_local][5] += s * (float) sc1[2] * (float) qv5;
    }
    if (c6 < ncols) {
        sum[qb_local][6] += s * (float) sc1[4] * (float) qv6;
    }
    if (c7 < ncols) {
        sum[qb_local][7] += s * (float) sc1[6] * (float) qv7;
    }
}

template <int block_qb, int block_y, int block_k>
static __global__ void axpy_q8_0_sparse(const block_q8_0 * __restrict__ x,
                                        const float * __restrict__ y,
                                        const float * __restrict__ sparse_idx,
                                        const int32_t * __restrict__ gpu_neu_idx,
                                        float * __restrict__ dst,
                                        const float sparse_threshold,
                                        const int   ncols,
                                        const int   nrows,
                                        const int   ncols_y,
                                        const int   nrows_launch,
                                        const int   qblocks_per_row) {
    static_assert(QK8_0 == 32, "Q8_0 kernel assumes QK8_0 == 32");

    const int lane  = threadIdx.x;
    const int ty    = threadIdx.y;
    const int col_y = blockIdx.y * block_y + ty;

    if (col_y >= ncols_y) {
        return;
    }

    const int qb0  = blockIdx.x * block_qb;
    const int row0 = blockIdx.z * block_k;
    const int row1 = min(row0 + block_k, nrows_launch);

    float sum[block_qb] = { 0.0f };

    y += col_y * nrows;
    dst += col_y * ncols;
    sparse_idx += col_y * nrows;

    for (int row = row0; row < row1; ++row) {
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

        const int row_x0 = row * qblocks_per_row;
#pragma unroll
        for (int i = 0; i < block_qb; ++i) {
            const int qb = qb0 + i;
            if (qb >= qblocks_per_row) {
                continue;
            }
            const int col = qb * QK8_0 + lane;
            if (col >= ncols) {
                continue;
            }
            sum[i] += q8_0_get_f32(x + row_x0 + qb, lane, alpha);
        }
    }

#pragma unroll
    for (int i = 0; i < block_qb; ++i) {
        const int qb = qb0 + i;
        if (qb >= qblocks_per_row) {
            continue;
        }
        const int col = qb * QK8_0 + lane;
        if (col < ncols) {
            atomicAdd(&dst[col], sum[i]);
        }
    }
}

template <int block_qb, int block_y, int block_k>
static __global__ void axpy_q4_K_sparse(const block_q4_K * __restrict__ x,
                                        const float * __restrict__ y,
                                        const float * __restrict__ sparse_idx,
                                        const int32_t * __restrict__ gpu_neu_idx,
                                        float * __restrict__ dst,
                                        const float sparse_threshold,
                                        const int   ncols,
                                        const int   nrows,
                                        const int   ncols_y,
                                        const int   nrows_launch,
                                        const int   qblocks_per_row) {
    static_assert(QK_K == 256, "Q4_K kernel assumes QK_K == 256");

    const int lane  = threadIdx.x;
    const int ty    = threadIdx.y;
    const int col_y = blockIdx.y * block_y + ty;

    if (col_y >= ncols_y) {
        return;
    }

    const int qb0  = blockIdx.x * block_qb;
    const int row0 = blockIdx.z * block_k;
    const int row1 = min(row0 + block_k, nrows_launch);

    float sum[block_qb][8] = { { 0.0f } };

    y += col_y * nrows;
    dst += col_y * ncols;
    sparse_idx += col_y * nrows;

    for (int row = row0; row < row1; ++row) {
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

        const int row_x0 = row * qblocks_per_row;
#pragma unroll
        for (int i = 0; i < block_qb; ++i) {
            const int qb = qb0 + i;
            if (qb >= qblocks_per_row) {
                continue;
            }
            const int col0 = qb * QK_K;
            if (col0 >= ncols) {
                continue;
            }
            axpy_q4_K_block<block_qb>(x + row_x0 + qb, alpha, col0, ncols, sum, i);
        }
    }

#pragma unroll
    for (int i = 0; i < block_qb; ++i) {
        const int qb = qb0 + i;
        if (qb >= qblocks_per_row) {
            continue;
        }

        const int col0 = qb * QK_K;
        const int c0   = col0 + lane;
        const int c1   = col0 + lane + 32;
        const int c2   = col0 + lane + 64;
        const int c3   = col0 + lane + 96;
        const int c4   = col0 + lane + 128;
        const int c5   = col0 + lane + 160;
        const int c6   = col0 + lane + 192;
        const int c7   = col0 + lane + 224;
        if (c0 < ncols) {
            atomicAdd(&dst[c0], sum[i][0]);
        }
        if (c1 < ncols) {
            atomicAdd(&dst[c1], sum[i][1]);
        }
        if (c2 < ncols) {
            atomicAdd(&dst[c2], sum[i][2]);
        }
        if (c3 < ncols) {
            atomicAdd(&dst[c3], sum[i][3]);
        }
        if (c4 < ncols) {
            atomicAdd(&dst[c4], sum[i][4]);
        }
        if (c5 < ncols) {
            atomicAdd(&dst[c5], sum[i][5]);
        }
        if (c6 < ncols) {
            atomicAdd(&dst[c6], sum[i][6]);
        }
        if (c7 < ncols) {
            atomicAdd(&dst[c7], sum[i][7]);
        }
    }
}

template <int block_qb, int block_y, int block_k>
static __global__ void axpy_q6_K_sparse(const block_q6_K * __restrict__ x,
                                        const float * __restrict__ y,
                                        const float * __restrict__ sparse_idx,
                                        const int32_t * __restrict__ gpu_neu_idx,
                                        float * __restrict__ dst,
                                        const float sparse_threshold,
                                        const int   ncols,
                                        const int   nrows,
                                        const int   ncols_y,
                                        const int   nrows_launch,
                                        const int   qblocks_per_row) {
    static_assert(QK_K == 256, "Q6_K kernel assumes QK_K == 256");

    const int lane  = threadIdx.x;
    const int ty    = threadIdx.y;
    const int col_y = blockIdx.y * block_y + ty;

    if (col_y >= ncols_y) {
        return;
    }

    const int qb0  = blockIdx.x * block_qb;
    const int row0 = blockIdx.z * block_k;
    const int row1 = min(row0 + block_k, nrows_launch);

    float sum[block_qb][8] = { { 0.0f } };

    y += col_y * nrows;
    dst += col_y * ncols;
    sparse_idx += col_y * nrows;

    for (int row = row0; row < row1; ++row) {
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

        const int row_x0 = row * qblocks_per_row;
#pragma unroll
        for (int i = 0; i < block_qb; ++i) {
            const int qb = qb0 + i;
            if (qb >= qblocks_per_row) {
                continue;
            }
            const int col0 = qb * QK_K;
            if (col0 >= ncols) {
                continue;
            }
            axpy_q6_K_block<block_qb>(x + row_x0 + qb, alpha, col0, ncols, sum, i);
        }
    }

#pragma unroll
    for (int i = 0; i < block_qb; ++i) {
        const int qb = qb0 + i;
        if (qb >= qblocks_per_row) {
            continue;
        }

        const int col0 = qb * QK_K;
        const int c0   = col0 + lane;
        const int c1   = col0 + lane + 32;
        const int c2   = col0 + lane + 64;
        const int c3   = col0 + lane + 96;
        const int c4   = col0 + lane + 128;
        const int c5   = col0 + lane + 160;
        const int c6   = col0 + lane + 192;
        const int c7   = col0 + lane + 224;
        if (c0 < ncols) {
            atomicAdd(&dst[c0], sum[i][0]);
        }
        if (c1 < ncols) {
            atomicAdd(&dst[c1], sum[i][1]);
        }
        if (c2 < ncols) {
            atomicAdd(&dst[c2], sum[i][2]);
        }
        if (c3 < ncols) {
            atomicAdd(&dst[c3], sum[i][3]);
        }
        if (c4 < ncols) {
            atomicAdd(&dst[c4], sum[i][4]);
        }
        if (c5 < ncols) {
            atomicAdd(&dst[c5], sum[i][5]);
        }
        if (c6 < ncols) {
            atomicAdd(&dst[c6], sum[i][6]);
        }
        if (c7 < ncols) {
            atomicAdd(&dst[c7], sum[i][7]);
        }
    }
}

template <int block_qb, int block_y, int block_k>
static void launch_axpy_q8_0_sparse(const block_q8_0 * x,
                                    const float *      y,
                                    const float *      sparse_idx,
                                    const int32_t *    gpu_neu_idx,
                                    float *            dst,
                                    const float        sparse_threshold,
                                    const int64_t      ncols,
                                    const int64_t      nrows,
                                    const int64_t      ncols_y,
                                    const int64_t      nrows_launch,
                                    cudaStream_t       stream) {
    const int  qblocks_per_row = (ncols + QK8_0 - 1) / QK8_0;
    const dim3 block_dims(WARP_SIZE, block_y, 1);
    const dim3 block_nums((qblocks_per_row + block_qb - 1) / block_qb, (ncols_y + block_y - 1) / block_y,
                          (nrows_launch + block_k - 1) / block_k);
    axpy_q8_0_sparse<block_qb, block_y, block_k><<<block_nums, block_dims, 0, stream>>>(
        x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, ncols_y, nrows_launch, qblocks_per_row);
}

template <int block_qb, int block_y, int block_k>
static void launch_axpy_q4_K_sparse(const block_q4_K * x,
                                    const float *      y,
                                    const float *      sparse_idx,
                                    const int32_t *    gpu_neu_idx,
                                    float *            dst,
                                    const float        sparse_threshold,
                                    const int64_t      ncols,
                                    const int64_t      nrows,
                                    const int64_t      ncols_y,
                                    const int64_t      nrows_launch,
                                    cudaStream_t       stream) {
    const int  qblocks_per_row = (ncols + QK_K - 1) / QK_K;
    const dim3 block_dims(WARP_SIZE, block_y, 1);
    const dim3 block_nums((qblocks_per_row + block_qb - 1) / block_qb, (ncols_y + block_y - 1) / block_y,
                          (nrows_launch + block_k - 1) / block_k);
    axpy_q4_K_sparse<block_qb, block_y, block_k><<<block_nums, block_dims, 0, stream>>>(
        x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, ncols_y, nrows_launch, qblocks_per_row);
}

template <int block_qb, int block_y, int block_k>
static void launch_axpy_q6_K_sparse(const block_q6_K * x,
                                    const float *      y,
                                    const float *      sparse_idx,
                                    const int32_t *    gpu_neu_idx,
                                    float *            dst,
                                    const float        sparse_threshold,
                                    const int64_t      ncols,
                                    const int64_t      nrows,
                                    const int64_t      ncols_y,
                                    const int64_t      nrows_launch,
                                    cudaStream_t       stream) {
    const int  qblocks_per_row = (ncols + QK_K - 1) / QK_K;
    const dim3 block_dims(WARP_SIZE, block_y, 1);
    const dim3 block_nums((qblocks_per_row + block_qb - 1) / block_qb, (ncols_y + block_y - 1) / block_y,
                          (nrows_launch + block_k - 1) / block_k);
    axpy_q6_K_sparse<block_qb, block_y, block_k><<<block_nums, block_dims, 0, stream>>>(
        x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, ncols_y, nrows_launch, qblocks_per_row);
}

static void axpy_q8_0_sparse_cuda(const block_q8_0 * x,
                                  const float *      y,
                                  const float *      sparse_idx,
                                  const int32_t *    gpu_neu_idx,
                                  float *            dst,
                                  const float        sparse_threshold,
                                  const int64_t      ncols,
                                  const int64_t      nrows,
                                  const int64_t      ncols_y,
                                  const int64_t      nrows_launch,
                                  cudaStream_t       stream) {
    if (ncols_y >= 4) {
        launch_axpy_q8_0_sparse<8, 8, 64>(x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, ncols_y,
                                          nrows_launch, stream);
    } else {
        launch_axpy_q8_0_sparse<8, 4, 32>(x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, ncols_y,
                                          nrows_launch, stream);
    }
}

static void axpy_q4_K_sparse_cuda(const block_q4_K * x,
                                  const float *      y,
                                  const float *      sparse_idx,
                                  const int32_t *    gpu_neu_idx,
                                  float *            dst,
                                  const float        sparse_threshold,
                                  const int64_t      ncols,
                                  const int64_t      nrows,
                                  const int64_t      ncols_y,
                                  const int64_t      nrows_launch,
                                  cudaStream_t       stream) {
    launch_axpy_q4_K_sparse<2, 4, 32>(x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, ncols_y,
                                      nrows_launch, stream);
}

static void axpy_q6_K_sparse_cuda(const block_q6_K * x,
                                  const float *      y,
                                  const float *      sparse_idx,
                                  const int32_t *    gpu_neu_idx,
                                  float *            dst,
                                  const float        sparse_threshold,
                                  const int64_t      ncols,
                                  const int64_t      nrows,
                                  const int64_t      ncols_y,
                                  const int64_t      nrows_launch,
                                  cudaStream_t       stream) {
    launch_axpy_q6_K_sparse<2, 4, 32>(x, y, sparse_idx, gpu_neu_idx, dst, sparse_threshold, ncols, nrows, ncols_y,
                                      nrows_launch, stream);
}

void ggml_cuda_axpy_q_sparse(ggml_backend_cuda_context & ctx,
                             const ggml_tensor *         src0,
                             const ggml_tensor *         src1,
                             ggml_tensor *               dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

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
    const int64_t   nrows_launch     = gpu_neu_idx ? dst->src[3]->ne[0] : src1->ne[0];
    const float     sparse_threshold = ggml_get_op_params_f32(dst, 1);
    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), ctx.stream()));

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            {
                const block_q8_0 * src0_d = (const block_q8_0 *) src0->data;
                axpy_q8_0_sparse_cuda(src0_d, src1_d, sparse_idx, gpu_neu_idx, dst_d, sparse_threshold, ne00, ne10,
                                      ne11, nrows_launch, ctx.stream());
            }
            break;
        case GGML_TYPE_Q4_K:
            {
                const block_q4_K * src0_d = (const block_q4_K *) src0->data;
                axpy_q4_K_sparse_cuda(src0_d, src1_d, sparse_idx, gpu_neu_idx, dst_d, sparse_threshold, ne00, ne10,
                                      ne11, nrows_launch, ctx.stream());
            }
            break;
        case GGML_TYPE_Q6_K:
            {
                const block_q6_K * src0_d = (const block_q6_K *) src0->data;
                axpy_q6_K_sparse_cuda(src0_d, src1_d, sparse_idx, gpu_neu_idx, dst_d, sparse_threshold, ne00, ne10,
                                      ne11, nrows_launch, ctx.stream());
            }
            break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}
