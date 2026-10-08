/* Compare actual quantized MoE layers across CPU/GPU placement, including
 * padded down rows, multi-token batches and graph-owned host buffer reuse. */
#define DS4_TEST_HOOKS
#include "../ds4.c"
#include <assert.h>

static void compare(const float *a, const float *b, uint64_t n, const char *name) {
    float worst = 0.0f, scale = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        assert(isfinite(a[i]) && isfinite(b[i]));
        const float d = fabsf(a[i] - b[i]);
        if (d > worst) worst = d;
        if (fabsf(a[i]) > scale) scale = fabsf(a[i]);
    }
    printf("%s: max_abs=%g scale=%g\n", name, (double)worst, (double)scale);
    assert(worst <= 2e-5f + 2e-4f * scale);
}

static void count_emitted(void *ud, int token) {
    assert(token >= 0);
    (*(int *)ud)++;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && strcmp(argv[2], "resident"))) {
        fprintf(stderr, "usage: %s MODEL.gguf [resident]\n", argv[0]); return 2;
    }
    const bool resident = argc == 3;
    ds4_engine_options opt = {.model_path=argv[1], .backend=DS4_BACKEND_CUDA,
        .context_size=128, .prefill_chunk=3, .n_threads=4, .cpu_experts=true};
    if (resident) {
        opt.gpu_memory_budget_bytes = 12ull * 1073741824ull;
        opt.gpu_expert_layers = 8;
        opt.gpu_expert_layers_set = true;
        opt.glm_mtp = true;
    }
    ds4_engine *e = NULL;
    assert(ds4_engine_open(&e, &opt) == 0);
    enum { MAX_T=9 };
    ds4_qwen4_gpu_graph g;
    assert(qwen4_graph_alloc(&g, &e->weights, 128, MAX_T, false, NULL, NULL, NULL, 0));
    assert(qwen4_graph_configure_placement(&g, e));
    if (resident) {
        assert(e->qwen4_resident_ready && e->qwen4_gpu_expert_mask);
        const ds4_tensor *t = e->weights.layer[0].hc_ffn_down;
        assert(ds4_gpu_cache_model_range(e->model.map,e->model.size,t->abs_offset,t->bytes,"resident probe"));
        assert(ds4_gpu_qwen4_release_layer_weights());
        const uint64_t available = ds4_gpu_qwen4_resident_available();
        assert(ds4_gpu_cache_model_range(e->model.map,e->model.size,t->abs_offset,t->bytes,"resident probe"));
        /* The second lookup must reuse the copy after layer-cache release. */
        assert(ds4_gpu_qwen4_resident_available() == available);
        assert(!ds4_gpu_qwen4_cache_resident(e->model.map,e->model.size,0,UINT64_MAX));
    }
    const uint32_t E=DS4_N_EMBD, HC=DS4_N_HC;
    float *x=xmalloc(MAX_T*E*sizeof(float));
    float *residual=xmalloc(MAX_T*E*HC*sizeof(float));
    float *inj=xmalloc(MAX_T*HC*HC*8u*sizeof(float));
    float *gpu_mid=xmalloc((uint64_t)MAX_T*(DS4_N_EXPERT_USED+1u)*DS4_N_FF_EXP*sizeof(float));
    float *gpu_part=xmalloc((uint64_t)MAX_T*(DS4_N_EXPERT_USED+1u)*E*sizeof(float));
    float *gpu=xmalloc(MAX_T*E*sizeof(float)), *cpu=xmalloc(MAX_T*E*sizeof(float));
    float *gpu_r=xmalloc(MAX_T*E*HC*sizeof(float)), *cpu_r=xmalloc(MAX_T*E*HC*sizeof(float));
    for (uint32_t i=0;i<MAX_T*E;i++) x[i]=1.5f*sinf((float)i*0.173f);
    for (uint32_t i=0;i<MAX_T*E*HC;i++) residual[i]=0.1f*cosf((float)i*0.31f);
    for (uint32_t i=0;i<MAX_T*HC*HC*8u;i++) inj[i]=0.05f*sinf((float)i*0.23f);
    const uint32_t layers[]={0,46,48};
    const uint32_t batches[]={1,2,3,MAX_T};
    for (uint32_t li=0;li<(resident ? 3u : 2u);li++) for (uint32_t ti=0;ti<4;ti++) {
        const uint32_t T=batches[ti];
        /* Repeated routing must exercise groups split at four slots. Keep
         * the first three inputs different for ordinary multi-token checks. */
        if (T == MAX_T) for (uint32_t t=4;t<MAX_T;t++) memcpy(x+t*E,x,E*sizeof(float));
        printf("layer=%u T=%u\n",layers[li],T);
        for (int hybrid=0;hybrid<2;hybrid++) {
            g.cpu_experts=hybrid!=0;
            g.gpu_expert_mask=0; /* Compare the same layer on both devices. */
            assert(ds4_gpu_tensor_write(g.mixed,0,x,(uint64_t)T*E*sizeof(float)));
            assert(ds4_gpu_tensor_write(g.R,0,residual,(uint64_t)T*E*HC*sizeof(float)));
            assert(ds4_gpu_tensor_write(g.inj,0,inj,(uint64_t)T*HC*HC*8u*sizeof(float)));
            assert(qwen4_graph_moe(&g,&e->model,&e->weights.layer[layers[li]],T));
            if (!hybrid && layers[li] == 48u) {
                assert(ds4_gpu_tensor_read(g.mid,0,gpu_mid,(uint64_t)T*(DS4_N_EXPERT_USED+1u)*DS4_N_FF_EXP*sizeof(float)));
                assert(ds4_gpu_tensor_read(g.part,0,gpu_part,(uint64_t)T*(DS4_N_EXPERT_USED+1u)*E*sizeof(float)));
            }
            assert(ds4_gpu_tensor_read(g.blk,0,hybrid?cpu:gpu,(uint64_t)T*E*sizeof(float)));
            assert(ds4_gpu_tensor_read(g.R,0,hybrid?cpu_r:gpu_r,(uint64_t)T*E*HC*sizeof(float)));
            assert(ds4_gpu_qwen4_release_layer_weights());
        }
        if (layers[li] == 48u && T == 1u) {
            printf("MTP gate type=%u down type=%u\n",e->weights.layer[48].ffn_gate_exps->type,e->weights.layer[48].ffn_down_exps->type);
            compare(gpu_mid,g.cpu_expert_mid,(uint64_t)DS4_N_EXPERT_USED*DS4_N_FF_EXP,"MTP mid");
            compare(gpu_part,g.cpu_expert_part,(uint64_t)DS4_N_EXPERT_USED*E,"MTP part");
        }
        compare(gpu,cpu,(uint64_t)T*E,"MoE output");
        compare(gpu_r,cpu_r,(uint64_t)T*E*HC,"wide residual");
    }
    free(x);free(residual);free(inj);free(gpu);free(cpu);free(gpu_r);free(cpu_r);
    free(gpu_mid);free(gpu_part);
    qwen4_graph_free(&g);
    /* Greedy CLI generation must not silently take the GPU-only helper. */
    ds4_tokens prompt = {0};
    ds4_tokenize_text(e, "Hello", &prompt);
    const uint64_t before = ds4_test_qwen4_cpu_expert_calls;
    int emitted = 0;
    assert(ds4_engine_generate_argmax(e, &prompt, 2, 128,
               count_emitted, NULL, &emitted, NULL, NULL) == 0);
    assert(emitted > 0 && ds4_test_qwen4_cpu_expert_calls > before);
    printf("greedy generation: emitted=%d CPU expert calls=%llu\n", emitted,
           (unsigned long long)(ds4_test_qwen4_cpu_expert_calls - before));
    ds4_tokens_free(&prompt);ds4_engine_close(e);
    puts("Qwen CPU expert placement: ok");
    return 0;
}
