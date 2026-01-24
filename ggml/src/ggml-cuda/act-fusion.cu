#include "act-fusion.cuh"
#include "unary.cuh"

static __device__ __forceinline__ uint32_t kairox_row_bcast(const uint32_t row,
                                                          const uint32_t nrows,
                                                          const uint32_t nrows_src,
                                                          const uint3    nrows_src_fd) {
    return nrows_src == 1 ? 0u : (nrows_src == nrows ? row : fastmodulo(row, nrows_src_fd));
}

static __global__ void kairox_add_add_fatrelu_mul_vec4_kernel(const float * __restrict__ up0,
                                                            const float * __restrict__ up1,
                                                            const float * __restrict__ gate0,
                                                            const float * __restrict__ gate1,
                                                            float * __restrict__ dst,
                                                            const int64_t k4,
                                                            const uint3   nc4_fd,
                                                            const int64_t up0_o,
                                                            const int64_t up1_o,
                                                            const int64_t gate0_o,
                                                            const int64_t gate1_o,
                                                            const int64_t dst_o,
                                                            const float   threshold) {
    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k4) {
        return;
    }

    const uint2   rc   = fast_div_modulo(i, nc4_fd);
    const int64_t row  = rc.x;
    const int64_t col4 = rc.y;

    const float4 up0v   = reinterpret_cast<const float4 *>(up0 + row * up0_o)[col4];
    const float4 up1v   = reinterpret_cast<const float4 *>(up1 + row * up1_o)[col4];
    const float4 gate0v = reinterpret_cast<const float4 *>(gate0 + row * gate0_o)[col4];
    const float4 gate1v = reinterpret_cast<const float4 *>(gate1 + row * gate1_o)[col4];

    float4 upv;
    upv.x = up0v.x + up1v.x;
    upv.y = up0v.y + up1v.y;
    upv.z = up0v.z + up1v.z;
    upv.w = up0v.w + up1v.w;

    float4 gatev;
    gatev.x = gate0v.x + gate1v.x;
    gatev.y = gate0v.y + gate1v.y;
    gatev.z = gate0v.z + gate1v.z;
    gatev.w = gate0v.w + gate1v.w;
    gatev.x = gatev.x > threshold ? gatev.x : 0.0f;
    gatev.y = gatev.y > threshold ? gatev.y : 0.0f;
    gatev.z = gatev.z > threshold ? gatev.z : 0.0f;
    gatev.w = gatev.w > threshold ? gatev.w : 0.0f;

    float4 outv;
    outv.x                                              = gatev.x * upv.x;
    outv.y                                              = gatev.y * upv.y;
    outv.z                                              = gatev.z * upv.z;
    outv.w                                              = gatev.w * upv.w;
    reinterpret_cast<float4 *>(dst + row * dst_o)[col4] = outv;
}

static __global__ void kairox_add_add_relu_relu_mul_vec4_kernel(const float * __restrict__ up0,
                                                              const float * __restrict__ up1,
                                                              const float * __restrict__ gate0,
                                                              const float * __restrict__ gate1,
                                                              float * __restrict__ dst,
                                                              const int64_t k4,
                                                              const uint3   nc4_fd,
                                                              const int64_t up0_o,
                                                              const int64_t up1_o,
                                                              const int64_t gate0_o,
                                                              const int64_t gate1_o,
                                                              const int64_t dst_o) {
    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k4) {
        return;
    }

    const uint2   rc   = fast_div_modulo(i, nc4_fd);
    const int64_t row  = rc.x;
    const int64_t col4 = rc.y;

    const float4 up0v   = reinterpret_cast<const float4 *>(up0 + row * up0_o)[col4];
    const float4 up1v   = reinterpret_cast<const float4 *>(up1 + row * up1_o)[col4];
    const float4 gate0v = reinterpret_cast<const float4 *>(gate0 + row * gate0_o)[col4];
    const float4 gate1v = reinterpret_cast<const float4 *>(gate1 + row * gate1_o)[col4];

    float4 upv;
    upv.x = up0v.x + up1v.x;
    upv.y = up0v.y + up1v.y;
    upv.z = up0v.z + up1v.z;
    upv.w = up0v.w + up1v.w;
    upv.x = fmaxf(upv.x, 0.0f);
    upv.y = fmaxf(upv.y, 0.0f);
    upv.z = fmaxf(upv.z, 0.0f);
    upv.w = fmaxf(upv.w, 0.0f);

    float4 gatev;
    gatev.x = gate0v.x + gate1v.x;
    gatev.y = gate0v.y + gate1v.y;
    gatev.z = gate0v.z + gate1v.z;
    gatev.w = gate0v.w + gate1v.w;
    gatev.x = fmaxf(gatev.x, 0.0f);
    gatev.y = fmaxf(gatev.y, 0.0f);
    gatev.z = fmaxf(gatev.z, 0.0f);
    gatev.w = fmaxf(gatev.w, 0.0f);

    float4 outv;
    outv.x                                              = gatev.x * upv.x;
    outv.y                                              = gatev.y * upv.y;
    outv.z                                              = gatev.z * upv.z;
    outv.w                                              = gatev.w * upv.w;
    reinterpret_cast<float4 *>(dst + row * dst_o)[col4] = outv;
}

static __global__ void kairox_add_relu_vec4_kernel(const float * __restrict__ src0,
                                                 const float * __restrict__ src1,
                                                 float * __restrict__ dst,
                                                 const int64_t  k4,
                                                 const uint3    nc4_fd,
                                                 const int64_t  src0_o,
                                                 const int64_t  src1_o,
                                                 const int64_t  dst_o,
                                                 const uint32_t nrows,
                                                 const uint32_t nrows_src1,
                                                 const uint3    nrows_src1_fd) {
    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k4) {
        return;
    }

    const uint2    rc   = fast_div_modulo(i, nc4_fd);
    const uint32_t row  = rc.x;
    const uint32_t col4 = rc.y;
    const uint32_t row1 = kairox_row_bcast(row, nrows, nrows_src1, nrows_src1_fd);

    const float4 src0v = reinterpret_cast<const float4 *>(src0 + int64_t(row) * src0_o)[col4];
    const float4 src1v = reinterpret_cast<const float4 *>(src1 + int64_t(row1) * src1_o)[col4];
    float4       outv;
    outv.x                                                       = fmaxf(src0v.x + src1v.x, 0.0f);
    outv.y                                                       = fmaxf(src0v.y + src1v.y, 0.0f);
    outv.z                                                       = fmaxf(src0v.z + src1v.z, 0.0f);
    outv.w                                                       = fmaxf(src0v.w + src1v.w, 0.0f);
    reinterpret_cast<float4 *>(dst + int64_t(row) * dst_o)[col4] = outv;
}

static __global__ void kairox_add_add_relu_vec4_kernel(const float * __restrict__ a00,
                                                     const float * __restrict__ a01,
                                                     const float * __restrict__ a1x,
                                                     float * __restrict__ dst,
                                                     const int64_t  k4,
                                                     const uint3    nc4_fd,
                                                     const int64_t  a00_o,
                                                     const int64_t  a01_o,
                                                     const int64_t  a1x_o,
                                                     const int64_t  dst_o,
                                                     const uint32_t nrows,
                                                     const uint32_t nrows_a01,
                                                     const uint32_t nrows_a1x,
                                                     const uint3    nrows_a01_fd,
                                                     const uint3    nrows_a1x_fd) {
    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k4) {
        return;
    }

    const uint2    rc    = fast_div_modulo(i, nc4_fd);
    const uint32_t row   = rc.x;
    const uint32_t col4  = rc.y;
    const uint32_t row01 = kairox_row_bcast(row, nrows, nrows_a01, nrows_a01_fd);
    const uint32_t row1x = kairox_row_bcast(row, nrows, nrows_a1x, nrows_a1x_fd);

    const float4 a00v = reinterpret_cast<const float4 *>(a00 + int64_t(row) * a00_o)[col4];
    const float4 a01v = reinterpret_cast<const float4 *>(a01 + int64_t(row01) * a01_o)[col4];
    const float4 a1xv = reinterpret_cast<const float4 *>(a1x + int64_t(row1x) * a1x_o)[col4];
    float4       outv;
    outv.x                                                       = fmaxf(a00v.x + a01v.x + a1xv.x, 0.0f);
    outv.y                                                       = fmaxf(a00v.y + a01v.y + a1xv.y, 0.0f);
    outv.z                                                       = fmaxf(a00v.z + a01v.z + a1xv.z, 0.0f);
    outv.w                                                       = fmaxf(a00v.w + a01v.w + a1xv.w, 0.0f);
    reinterpret_cast<float4 *>(dst + int64_t(row) * dst_o)[col4] = outv;
}

void ggml_cuda_op_kairox_add_add_fatrelu_mul(ggml_backend_cuda_context & ctx,
                                           ggml_tensor *               add0,
                                           ggml_tensor *               add1,
                                           ggml_tensor *               fatrelu,
                                           ggml_tensor *               mul) {
    const ggml_tensor * gate_add  = fatrelu->src[0];
    const ggml_tensor * up_add    = gate_add == add0 ? add1 : add0;
    const int64_t       nc        = mul->ne[0] / 4;
    const int64_t       k4        = ggml_nrows(mul) * nc;
    const int64_t       nblocks   = (k4 + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const float         threshold = ggml_get_op_params_f32(fatrelu, 0);
    const int64_t       up0_o     = up_add->src[0]->nb[1] / ggml_element_size(up_add->src[0]);
    const int64_t       up1_o     = up_add->src[1]->nb[1] / ggml_element_size(up_add->src[1]);
    const int64_t       gate0_o   = gate_add->src[0]->nb[1] / ggml_element_size(gate_add->src[0]);
    const int64_t       gate1_o   = gate_add->src[1]->nb[1] / ggml_element_size(gate_add->src[1]);
    const int64_t       dst_o     = mul->nb[1] / ggml_element_size(mul);
    const uint3         nc_fd     = init_fastdiv_values(nc);

    kairox_add_add_fatrelu_mul_vec4_kernel<<<nblocks, CUDA_GLU_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) up_add->src[0]->data, (const float *) up_add->src[1]->data,
        (const float *) gate_add->src[0]->data, (const float *) gate_add->src[1]->data, (float *) mul->data, k4, nc_fd,
        up0_o, up1_o, gate0_o, gate1_o, dst_o, threshold);
}

void ggml_cuda_op_kairox_add_add_relu_relu_mul(ggml_backend_cuda_context & ctx,
                                             ggml_tensor *               add0,
                                             ggml_tensor *               add1,
                                             ggml_tensor *               relu0,
                                             ggml_tensor *               relu1,
                                             ggml_tensor *               mul) {
    GGML_ASSERT(relu0->src[0] == add0 || relu0->src[0] == add1);
    GGML_ASSERT(relu1->src[0] == add0 || relu1->src[0] == add1);

    const ggml_tensor * up_add   = relu0->src[0];
    const ggml_tensor * gate_add = relu1->src[0];
    const int64_t       nc       = mul->ne[0] / 4;
    const int64_t       k4       = ggml_nrows(mul) * nc;
    const int64_t       nblocks  = (k4 + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const int64_t       up0_o    = up_add->src[0]->nb[1] / ggml_element_size(up_add->src[0]);
    const int64_t       up1_o    = up_add->src[1]->nb[1] / ggml_element_size(up_add->src[1]);
    const int64_t       gate0_o  = gate_add->src[0]->nb[1] / ggml_element_size(gate_add->src[0]);
    const int64_t       gate1_o  = gate_add->src[1]->nb[1] / ggml_element_size(gate_add->src[1]);
    const int64_t       dst_o    = mul->nb[1] / ggml_element_size(mul);
    const uint3         nc_fd    = init_fastdiv_values(nc);

    kairox_add_add_relu_relu_mul_vec4_kernel<<<nblocks, CUDA_GLU_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) up_add->src[0]->data, (const float *) up_add->src[1]->data,
        (const float *) gate_add->src[0]->data, (const float *) gate_add->src[1]->data, (float *) mul->data, k4, nc_fd,
        up0_o, up1_o, gate0_o, gate1_o, dst_o);
}

void ggml_cuda_op_kairox_add_add_relu(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor *         add0,
                                    const ggml_tensor *         add1,
                                    ggml_tensor *               relu) {
    const ggml_tensor * add1_other = add1->src[0] == add0 ? add1->src[1] : add1->src[0];
    const int64_t       nc         = relu->ne[0] / 4;
    const int64_t       k4         = ggml_nrows(relu) * nc;
    const int64_t       nblocks    = (k4 + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    const int64_t       a00_o      = add0->src[0]->nb[1] / ggml_element_size(add0->src[0]);
    const int64_t       a01_o      = add0->src[1]->nb[1] / ggml_element_size(add0->src[1]);
    const int64_t       a1x_o      = add1_other->nb[1] / ggml_element_size(add1_other);
    const int64_t       dst_o      = relu->nb[1] / ggml_element_size(relu);
    const uint3         nc_fd      = init_fastdiv_values(nc);
    const uint32_t      nrows      = (uint32_t) ggml_nrows(relu);
    const uint32_t      nrows_a01  = (uint32_t) ggml_nrows(add0->src[1]);
    const uint32_t      nrows_a1x  = (uint32_t) ggml_nrows(add1_other);

    kairox_add_add_relu_vec4_kernel<<<nblocks, CUDA_RELU_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) add0->src[0]->data, (const float *) add0->src[1]->data, (const float *) add1_other->data,
        (float *) relu->data, k4, nc_fd, a00_o, a01_o, a1x_o, dst_o, nrows, nrows_a01, nrows_a1x,
        init_fastdiv_values(nrows_a01), init_fastdiv_values(nrows_a1x));
}

void ggml_cuda_op_kairox_add_relu(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * relu) {
    const int64_t  nc         = relu->ne[0] / 4;
    const int64_t  k4         = ggml_nrows(relu) * nc;
    const int64_t  nblocks    = (k4 + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    const int64_t  src0_o     = add->src[0]->nb[1] / ggml_element_size(add->src[0]);
    const int64_t  src1_o     = add->src[1]->nb[1] / ggml_element_size(add->src[1]);
    const int64_t  dst_o      = relu->nb[1] / ggml_element_size(relu);
    const uint32_t nrows      = (uint32_t) ggml_nrows(relu);
    const uint32_t nrows_src1 = (uint32_t) ggml_nrows(add->src[1]);

    kairox_add_relu_vec4_kernel<<<nblocks, CUDA_RELU_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) add->src[0]->data, (const float *) add->src[1]->data, (float *) relu->data, k4,
        init_fastdiv_values(nc), src0_o, src1_o, dst_o, nrows, nrows_src1, init_fastdiv_values(nrows_src1));
}
