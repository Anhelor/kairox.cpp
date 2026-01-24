#pragma once

#include "common.cuh"

void ggml_cuda_op_kairox_add_add_fatrelu_mul(ggml_backend_cuda_context & ctx,
                                           ggml_tensor *               add0,
                                           ggml_tensor *               add1,
                                           ggml_tensor *               fatrelu,
                                           ggml_tensor *               mul);

void ggml_cuda_op_kairox_add_add_relu_relu_mul(ggml_backend_cuda_context & ctx,
                                             ggml_tensor *               add0,
                                             ggml_tensor *               add1,
                                             ggml_tensor *               relu0,
                                             ggml_tensor *               relu1,
                                             ggml_tensor *               mul);

void ggml_cuda_op_kairox_add_add_relu(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor *         add0,
                                    const ggml_tensor *         add1,
                                    ggml_tensor *               relu);

void ggml_cuda_op_kairox_add_relu(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * relu);
