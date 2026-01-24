#include "common.cuh"

void ggml_cuda_axpy_f_sparse(ggml_backend_cuda_context & ctx,
                             const ggml_tensor *         src0,
                             const ggml_tensor *         src1,
                             ggml_tensor *               dst);
