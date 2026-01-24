#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<std::vector<llama_token_data>> dists;

    struct common_sampler * smpl = nullptr;
};

struct speculative_bench_run_data {
    int32_t n_prompt = 0;
    int32_t n_decode = 0;
    int64_t t_prompt_us = 0;
    int64_t t_decode_us = 0;
};

static std::vector<std::string> load_bench_prompts_from_file(const std::string & path) {
    std::vector<std::string> prompts;
    std::ifstream file(path);
    if (!file) {
        LOG_ERR("failed to open benchmark prompt file: '%s'\n", path.c_str());
        return prompts;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            prompts.push_back(std::move(line));
        }
    }

    return prompts;
}

static bool decode_or_log(llama_context * ctx, llama_batch batch, const char * where) {
    const int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        LOG_ERR("%s: llama_decode failed, ret = %d\n", where, ret);
        return false;
    }

    return true;
}

static bool speculative_bench_run_one(
               llama_model *                   model_tgt,
               llama_model *                   model_dft,
             llama_context *                 ctx_tgt,
             llama_context *                 ctx_dft,
             llama_memory_t                  mem_tgt,
             llama_memory_t                  mem_dft,
       const llama_vocab *                   vocab_tgt,
             common_params &                 params,
                      int                    n_seq_dft,
                    float                    p_draft_split,
    std::default_random_engine &             rng,
    std::uniform_real_distribution<> &       u_dist,
      const std::string &                    prompt,
    speculative_bench_run_data &             out) {
    llama_memory_clear(mem_tgt, true);
    llama_memory_clear(mem_dft, true);
    llama_perf_context_reset(ctx_tgt);
    llama_perf_context_reset(ctx_dft);

    std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return false;
    }

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    if (!decode_or_log(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1), __func__)) {
        return false;
    }
    if (!decode_or_log(ctx_tgt, llama_batch_get_one(&inp.back(),           1), __func__)) {
        return false;
    }
    if (!decode_or_log(ctx_dft, llama_batch_get_one( inp.data(), n_input), __func__)) {
        return false;
    }

    const auto t_enc_end = ggml_time_us();

    int n_predict = 0;
    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();
    const int n_ctx_tgt = llama_n_ctx(ctx_tgt);
    const int n_ctx_dft = llama_n_ctx(ctx_dft);

    bool has_eos = false;

    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);
    std::vector<seq_draft> drafts(n_seq_dft);

    for (int s = 0; s < n_seq_dft; ++s) {
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us();

    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    while (true) {
        std::set<int> active_seqs = {};

        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }
            active_seqs.insert(s);
        }

        int i_dft  = 0;
        int s_keep = 0;

        llama_token token_id;
        while (true) {
            {
                bool accept = false;
                if (params.sampling.temp > 0) {
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);

                    auto & dist_tgt = *common_sampler_get_candidates(smpl, true);

                    float p_tgt = 0.0f;
                    float p_dft = 0.0f;

                    while (active_seqs.size() > 0) {
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        if (accept) {
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }

                        float r = u_dist(rng);
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        if (r <= p_tgt / p_dft) {
                            s_keep = s;
                            accept = true;
                            token_id = drafts[s].tokens[i_dft];
                            common_sampler_accept(smpl, token_id, true);
                            break;
                        } else {
                            drafts[s].active = false;

                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);

                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f;

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }
                                sum_probs += dist_tgt.data[i].p;
                            }

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }

                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        active_seqs.erase(s);
                        for (int i = 0; i < n_seq_dft; i++) {
                            if (i == s) {
                                continue;
                            }
                            if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                drafts[i].active = drafts[i].active && accept;
                                if (!drafts[i].active) {
                                    active_seqs.erase(s);
                                }
                            }
                        }
                    }

                    if (!accept) {
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        std::discrete_distribution<> dist(probs.begin(), probs.end());
                        const int idx = dist(rng);
                        token_id = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                    }
                } else {
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);
                    common_sampler_accept(smpl, token_id, true);

                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) {
                            continue;
                        }

                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            s_keep = s;
                            accept = true;
                        } else {
                            drafts[s].active = false;
                        }
                    }
                }

                if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict;

                if (accept) {
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;
                    continue;
                } else {
                    break;
                }
            }
        }

        {
            {
                llama_memory_seq_keep(mem_dft, s_keep);
                llama_memory_seq_cp  (mem_dft, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_dft, 0);

                llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);
                llama_memory_seq_keep(mem_tgt, s_keep);
                llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_tgt, 0);
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            if (n_past_dft >= n_ctx_dft) {
                LOG_WRN("%s: benchmark draft context full => stopping run\n", __func__);
                break;
            }

            common_batch_clear(batch_dft);
            common_batch_add  (batch_dft, token_id, n_past_dft, { 0 }, true);

            llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
            if (!decode_or_log(ctx_dft, batch_dft, __func__)) {
                return false;
            }

            ++n_past_dft;
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        if (n_past_tgt >= n_ctx_tgt) {
            LOG_WRN("%s: benchmark context full => stopping run\n", __func__);
            break;
        }

        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        for (int i = 0; i < params.speculative.n_max; ++i) {
            if (n_past_tgt + i + 1 >= n_ctx_tgt || n_past_cur >= n_ctx_dft) {
                break;
            }

            batch_dft.n_tokens = 0;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);
                std::vector<int> sa(1, s);

                for (int f = 1; f < 8; ++f) {
                    if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                        llama_memory_seq_rm(mem_dft,    n_seq_cur, -1, -1);
                        llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);

                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;
                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].dists       = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;
                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);
                        sa.push_back(n_seq_cur);
                        n_seq_cur++;
                    } else {
                        break;
                    }
                }

                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;
                    const int s = sa[is];
                    common_sampler_accept(drafts[s].smpl, id, true);
                    drafts[s].tokens.push_back(id);
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);
                    common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);
                    drafts[s].i_batch_dft = batch_dft.n_tokens;
                    common_batch_add(batch_dft, id, n_past_cur, { s }, true);
                    if (batch_tgt.n_tokens > params.speculative.n_max) {
                        drafts[s].drafting = false;
                    }
                }
            }

            if (batch_dft.n_tokens == 0) {
                break;
            }

            if (!decode_or_log(ctx_dft, batch_dft, __func__)) {
                return false;
            }
            ++n_past_cur;

            if (batch_tgt.n_tokens > params.speculative.n_max) {
                break;
            }
        }

        {
            if (batch_tgt.n_tokens == 0) {
                LOG_WRN("%s: benchmark context full => stopping run\n", __func__);
                break;
            }

            llama_memory_seq_keep(mem_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
            }
            if (!decode_or_log(ctx_tgt, batch_tgt, __func__)) {
                return false;
            }
            ++n_past_tgt;
        }

        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }
            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }
    }

    const auto t_dec_end = ggml_time_us();

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);
    llama_batch_free(batch_tgt);

    out.n_prompt    = n_input;
    out.n_decode    = n_predict;
    out.t_prompt_us = t_enc_end - t_enc_start;
    out.t_decode_us = t_dec_end - t_dec_start;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    if (params.speculative.mparams_dft.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    const float p_draft_split = params.speculative.p_split;

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    // load the draft model
    params.devices = params.speculative.devices;
    params.model = params.speculative.mparams_dft;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    params.tensor_buft_overrides     = params.speculative.tensor_buft_overrides;
    const std::string kairox_ms_path   = params.kairox_ms_path;
    params.kairox_ms_path              = "";

    auto llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft->model();
    ctx_dft   = llama_init_dft->context();

    kairox_init_from_model_and_ctx(model_tgt, ctx_tgt, model_dft, ctx_dft, kairox_ms_path.c_str(), params.vram_budget);

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);

    const bool bench_mode = !params.bench_prompt_file.empty() || params.bench_runs > 1 || params.bench_warmup > 0;
    if (bench_mode) {
        auto bench_exit = [&](int code) {
            llama_backend_free();
            return code;
        };

        if (params.n_predict < 0) {
            LOG_ERR("%s: benchmark mode requires --n-predict >= 0\n", __func__);
            return bench_exit(1);
        }

        std::vector<std::string> bench_prompts;
        if (!params.bench_prompt_file.empty()) {
            bench_prompts = load_bench_prompts_from_file(params.bench_prompt_file);
            if (bench_prompts.empty()) {
                LOG_ERR("no prompts loaded from benchmark prompt file: '%s'\n", params.bench_prompt_file.c_str());
                return bench_exit(1);
            }
        } else {
            bench_prompts.push_back(params.prompt);
        }

        size_t target_runs = 0;
        if (params.bench_prompt_file.empty()) {
            target_runs = (size_t) params.bench_runs;
        } else if (params.bench_runs_set) {
            target_runs = (size_t) params.bench_runs;
        } else {
            target_runs = bench_prompts.size();
        }

        const int n_warmup = params.bench_warmup;
        LOG_INF("benchmark mode: warmup = %d, target measured runs = %zu, prompts = %zu\n",
                n_warmup, target_runs, bench_prompts.size());

        for (int i = 0; i < n_warmup; ++i) {
            speculative_bench_run_data warmup_data;
            const std::string & prompt_run = bench_prompts[(size_t) i % bench_prompts.size()];
            if (!speculative_bench_run_one(model_tgt, model_dft, ctx_tgt, ctx_dft, mem_tgt, mem_dft, vocab_tgt, params, n_seq_dft, p_draft_split, rng, u_dist, prompt_run, warmup_data)) {
                return bench_exit(1);
            }
            LOG_INF("benchmark warmup %d/%d complete\n", i + 1, n_warmup);
        }

        std::vector<double> prefill_tps_values;
        std::vector<double> decode_tps_values;
        prefill_tps_values.reserve(target_runs);
        decode_tps_values.reserve(target_runs);
        const size_t max_attempts_without_progress = 10 * std::max(target_runs, bench_prompts.size());
        size_t n_attempted = 0;
        size_t n_attempted_since_include = 0;
        size_t n_filtered = 0;
        size_t n_included = 0;

        while (n_included < target_runs) {
            speculative_bench_run_data run_data;
            const std::string & prompt_run = bench_prompts[n_attempted % bench_prompts.size()];
            if (!speculative_bench_run_one(model_tgt, model_dft, ctx_tgt, ctx_dft, mem_tgt, mem_dft, vocab_tgt, params, n_seq_dft, p_draft_split, rng, u_dist, prompt_run, run_data)) {
                return bench_exit(1);
            }
            ++n_attempted;

            const double prefill_tps = run_data.t_prompt_us > 0 ? (1e6 * run_data.n_prompt / (double) run_data.t_prompt_us) : 0.0;
            const double decode_tps  = run_data.t_decode_us > 0 ? (1e6 * run_data.n_decode / (double) run_data.t_decode_us) : 0.0;
            const bool include_in_stats = run_data.n_decode >= 16;
            if (!include_in_stats) {
                ++n_filtered;
                ++n_attempted_since_include;
            } else {
                ++n_included;
                n_attempted_since_include = 0;
            }

            LOG("bench run attempt %zu (included %zu/%zu): prompt = %d tok, decode = %d tok, prefill = %.2f t/s, decode = %.2f t/s%s\n",
                n_attempted, n_included, target_runs, run_data.n_prompt, run_data.n_decode, prefill_tps, decode_tps,
                include_in_stats ? "" : " [filtered: decode < 16]");

            if (include_in_stats) {
                prefill_tps_values.push_back(prefill_tps);
                decode_tps_values.push_back(decode_tps);
            }

            if (n_included < target_runs && n_attempted_since_include >= max_attempts_without_progress) {
                LOG_WRN("benchmark stopped early after %zu consecutive filtered attempts without collecting a new measured run\n",
                        n_attempted_since_include);
                break;
            }
        }

        LOG("\nbenchmark summary:\n");
        LOG("  target measured runs: %zu\n", target_runs);
        LOG("  attempted runs:       %zu\n", n_attempted);
        LOG("  filtered runs (decode < 16): %zu\n", n_filtered);
        LOG("  included runs:        %zu\n", n_included);

        if (n_included < target_runs) {
            LOG("  note: stopped before reaching the target because no new measured runs were collected for too many attempts\n");
        }

        if (decode_tps_values.empty()) {
            LOG("  no runs passed the decode token filter\n");
            return bench_exit(0);
        }

        const double mean_prefill_tps = std::accumulate(prefill_tps_values.begin(), prefill_tps_values.end(), 0.0) / prefill_tps_values.size();
        const double mean_decode_tps  = std::accumulate(decode_tps_values.begin(), decode_tps_values.end(), 0.0) / decode_tps_values.size();


        LOG("  prefill mean:   %.2f t/s\n", mean_prefill_tps);
        LOG("  decode mean:    %.2f t/s\n", mean_decode_tps);

        return bench_exit(0);
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1));
    llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input));

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

    // how many tokens to draft each time
    int n_draft = params.speculative.n_max;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context (reuse the llama_context's sampling instance)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    for (int s = 0; s < n_seq_dft; ++s) {
        // allocate llama_sampler for each draft sequence
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    while (true) {
        std::set<int> active_seqs = {};

        // print current draft sequences
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            active_seqs.insert(s);
            const auto & tokens = drafts[s].tokens;

            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
        }

        int i_dft  = 0;
        int s_keep = 0;

        llama_token token_id;
        std::string token_str;

        // loop until we fail to accept a drafted token or we run out of drafted tokens
        while (true) {

            // check if the target token matches any of the drafts
            // for stochastic sampling, attempt to match the token with the drafted tokens
            {
                bool accept = false;
                if (params.sampling.temp > 0) {
                    // stochastic verification
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);

                    auto & dist_tgt = *common_sampler_get_candidates(smpl, true);

                    float p_tgt = 0.0f;
                    float p_dft = 0.0f;

                    while (active_seqs.size() > 0) {
                        // randomly select a sequence to verify from active sequences
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        if (accept) {
                            // if we already accepted a token, we can skip the rest
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }

                        LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int) active_seqs.size());
                        float r = u_dist(rng);
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        //GGML_ASSERT(dist_tgt.size <= dist_dft.size);

                        // acquire the token probabilities assigned by the draft and target models
                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);
                        if (r <= p_tgt / p_dft) {
                            s_keep = s;
                            accept = true;
                            token_id = drafts[s].tokens[i_dft];
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                            common_sampler_accept(smpl, token_id, true);

                            LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break;
                        } else {
                            LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false;

                            // calculate residual probability
                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);

                            // sort dist by id
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f;

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }

                                sum_probs += dist_tgt.data[i].p;
                            }

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }

                            // sort dist_tgt by p desc
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        active_seqs.erase(s);
                        for (int i = 0; i < n_seq_dft; i++) {
                            if (i == s) {
                                continue;
                            }
                            if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                // synchronize active status for sequences with the same drafted token
                                drafts[i].active = drafts[i].active && accept;
                                if (!drafts[i].active) {
                                    active_seqs.erase(s);
                                }
                            }
                        }
                    }

                    if (!accept) {
                        // all drafted tokens were rejected
                        // sample from the target model
                        LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        std::discrete_distribution<> dist(probs.begin(), probs.end());

                        const int idx = dist(rng);

                        token_id = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                    }
                } else {
                    // greedy verification

                    // sample from the target model
                    LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);

                    common_sampler_accept(smpl, token_id, true);

                    token_str = common_token_to_piece(ctx_tgt, token_id);

                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) {
                            continue;
                        }

                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());

                            s_keep = s;
                            accept = true;
                        } else {
                            drafts[s].active = false;
                        }
                    }
                }

                if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict;

                if (accept) {
                    ++n_accept;
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;
                    if (params.use_color) {
                        // Color token according to its origin sequence
                        LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    continue;
                } else {
                    LOG("%s", token_str.c_str());
                    break;
                }
            }
        }

        {
            LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

            // TODO: simplify
            {
                LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                llama_memory_seq_keep(mem_dft, s_keep);
                llama_memory_seq_cp  (mem_dft, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_dft, 0);

                llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);
                llama_memory_seq_keep(mem_tgt, s_keep);
                llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_tgt, 0);
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            common_batch_clear(batch_dft);
            common_batch_add  (batch_dft, token_id, n_past_dft, { 0 }, true);

            llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
            // LOG_DBG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
            llama_decode(ctx_dft, batch_dft);

            ++n_past_dft;
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);

                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);

                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                std::vector<int> sa(1, s);

                // attempt to split the branch if the probability is high enough
                for (int f = 1; f < 8; ++f) {
                    if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                        LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur);

                        llama_memory_seq_rm(mem_dft,    n_seq_cur, -1, -1);
                        llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);

                        // all previous tokens from this branch are now also part of the new branch
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        // copy the draft state
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;

                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].dists       = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);

                        sa.push_back(n_seq_cur);

                        n_seq_cur++;
                    } else {
                        break;
                    }
                }

                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;

                    const int s = sa[is];

                    common_sampler_accept(drafts[s].smpl, id, true);

                    drafts[s].tokens.push_back(id);
                    // save cur_p.data into drafts[s].dists
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                    // add unique drafted tokens to the target batch
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                    common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                    // add the token to the batch for batched decoding with the draft model
                    drafts[s].i_batch_dft = batch_dft.n_tokens;

                    common_batch_add(batch_dft, id, n_past_cur, { s }, true);

                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                }
            }

            // no sequence is drafting anymore
            if (batch_dft.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch_dft);
            ++n_past_cur;
            ++n_drafted;

            if (batch_tgt.n_tokens > n_draft) {
                break;
            }
        }

        // evaluate the target model on the drafted tokens
        {
            llama_memory_seq_keep(mem_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
            }

            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            llama_decode(ctx_tgt, batch_tgt);
            ++n_past_tgt;
        }

        // the first token is always proposed by the target model before the speculation loop so we erase it here
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    // TODO: print sampling/grammar timings for all drafts
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
