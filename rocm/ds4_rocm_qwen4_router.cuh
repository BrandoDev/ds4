/* Qwen router softmax and stable top-k. One thread owns each token so ties
 * select the smallest expert ID, matching the reference ordering. */
__global__ static void qwen4_rocm_router_kernel(
        int *selected, float *weights, const float *logits, const float *x,
        const char *gate, float *shared_gate, uint32_t NE, uint32_t NS,
        uint32_t K, uint32_t gate_type) {
    const uint32_t t = blockIdx.x;
    if (threadIdx.x) return;
    float p[512];
    float mx = -FLT_MAX;
    for (uint32_t e = 0; e < NE; e++) mx = fmaxf(mx, logits[(uint64_t)t * NE + e]);
    float total = 0.0f;
    for (uint32_t e = 0; e < NE; e++) {
        p[e] = expf(logits[(uint64_t)t * NE + e] - mx);
        total += p[e];
    }
    for (uint32_t e = 0; e < NE; e++) p[e] /= total;
    if (K) {
        float v = 0.0f;
        for (uint32_t i = 0; i < K; i++)
            v += qwen4_rocm_weight_at(gate, gate_type, i) * x[(uint64_t)t * K + i];
        shared_gate[t] = v;
    }
    float chosen_total = 0.0f;
    for (uint32_t s = 0; s < NS; s++) {
        uint32_t best_id = 0;
        float best = -1.0f;
        for (uint32_t e = 0; e < NE; e++) {
            if (p[e] > best) { best = p[e]; best_id = e; }
        }
        selected[(uint64_t)t * NS + s] = (int)best_id;
        weights[(uint64_t)t * NS + s] = best;
        chosen_total += best;
        p[best_id] = -1.0f;
    }
    for (uint32_t s = 0; s < NS; s++) weights[(uint64_t)t * NS + s] /= chosen_total;
}

extern "C" int ds4_gpu_qwen4_router_topk_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *x, const void *map, uint64_t size, uint64_t gate_offset,
        uint32_t gate_type, uint32_t K, ds4_gpu_tensor *shared_gate,
        uint32_t T, uint32_t NE, uint32_t NS) {
    if (!T || !NE || NE > 512u || !NS || NS > NE || NS > 32u ||
        !selected || !weights || !logits ||
        selected->bytes < (uint64_t)T * NS * 4u ||
        weights->bytes < (uint64_t)T * NS * 4u ||
        logits->bytes < (uint64_t)T * NE * 4u) return 0;
    const uint64_t rb = K ? qwen4_rocm_row_bytes(gate_type, K) : 0u;
    if (K && (!rb || !x || !shared_gate || !map ||
              x->bytes < (uint64_t)T * K * 4u || shared_gate->bytes < (uint64_t)T * 4u ||
              gate_offset > size || rb > size - gate_offset)) return 0;
    const char *w = K ? cuda_model_range_ptr(map, gate_offset, rb, "Qwen shared gate") : NULL;
    if (K && !w) return 0;
    qwen4_rocm_router_kernel<<<T, 1>>>(
        (int *)selected->ptr, (float *)weights->ptr, (const float *)logits->ptr,
        K ? (const float *)x->ptr : NULL, w,
        K ? (float *)shared_gate->ptr : NULL, NE, NS, K, gate_type);
    return cuda_ok(cudaGetLastError(), "Qwen router");
}
