#pragma once

#include "ggml-kairox.hpp"
#include "llama.h"

struct kairox_cache_manager {
    std::vector<kairox_layer_cache *> layer_caches;
    std::vector<ggml_tensor *>            reorder_perms;

    ggml_context *        ctx_cpu     = nullptr;
    ggml_context *        ctx_gpu     = nullptr;
    ggml_backend_t        backend_cpu = nullptr;
    ggml_backend_t        backend_gpu = nullptr;
    ggml_backend_buffer_t buf_cpu     = nullptr;
    ggml_backend_buffer_t buf_gpu     = nullptr;

    ggml_tensor * group_identity   = nullptr;
    float         sparse_threshold = 0.5f;

    kairox_cache_manager(llama_model * model, const char * kairox_ms_path, int64_t vram_budget);
    ~kairox_cache_manager();
};
