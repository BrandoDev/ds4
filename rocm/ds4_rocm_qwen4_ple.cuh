/* PLE and MTP operations shared by the Qwen ROCm graph. */
__global__ static void qwen4_rocm_ple_gate_kernel(
        float *gated, float *normed, const float *R, const float *key,
        const float *value, const float *gk, const float *gq, const float *gc,
        uint32_t E, uint32_t hc, float eps) {
    const uint32_t s = blockIdx.x, t = blockIdx.y;
    if (threadIdx.x) return;
    const uint64_t base = ((uint64_t)t * hc + s) * E;
    float sk = 0.0f, sq = 0.0f;
    for (uint32_t i = 0; i < E; i++) {
        sk += key[base + i] * key[base + i];
        sq += R[base + i] * R[base + i];
    }
    const float ik = rsqrtf(sk / E + eps), iq = rsqrtf(sq / E + eps);
    float dot = 0.0f;
    for (uint32_t i = 0; i < E; i++)
        dot += (key[base + i] * ik * gk[s * E + i]) *
               (R[base + i] * iq * gq[s * E + i]);
    const float a = dot * rsqrtf((float)E);
    const float mag = sqrtf(fmaxf(fabsf(a), 1e-6f));
    const float gate = qwen4_rocm_sigmoid(a > 0.0f ? mag : a < 0.0f ? -mag : 0.0f);
    float ss = 0.0f;
    for (uint32_t i = 0; i < E; i++) {
        const float v = gate * value[(uint64_t)t * E + i];
        gated[base + i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(ss / E + eps);
    for (uint32_t i = 0; i < E; i++) normed[base + i] = gated[base + i] * inv * gc[s * E + i];
}

__global__ static void qwen4_rocm_ple_conv_kernel(
        float *R, const float *gated, const float *normed, float *history,
        const char *w, uint32_t type, uint32_t T, uint32_t C, uint32_t K,
        uint32_t dilation, float *snap, uint32_t st, float *snap2, uint32_t st2) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const uint32_t H = (K - 1u) * dilation;
    float hist[9], taps[4];
    for (uint32_t i = 0; i < H; i++) hist[i] = history[(uint64_t)i * C + c];
    for (uint32_t i = 0; i < K; i++) taps[i] = qwen4_rocm_weight_at(w, type, (uint64_t)c * K + i);
    for (uint32_t t = 0; t < T; t++) {
        const uint64_t p = (uint64_t)t * C + c;
        const float cur = normed[p];
        float v = taps[K - 1u] * cur;
        for (uint32_t i = 0; i + 1u < K; i++) v += taps[i] * hist[i * dilation];
        for (uint32_t i = 0; i + 1u < H; i++) hist[i] = hist[i + 1u];
        hist[H - 1u] = cur;
        R[p] += gated[p] + qwen4_rocm_silu(v);
        if (snap && t == st) for (uint32_t i = 0; i < H; i++) snap[(uint64_t)i * C + c] = hist[i];
        if (snap2 && t == st2) for (uint32_t i = 0; i < H; i++) snap2[(uint64_t)i * C + c] = hist[i];
    }
    for (uint32_t i = 0; i < H; i++) history[(uint64_t)i * C + c] = hist[i];
}

extern "C" int ds4_gpu_qwen4_ple_gate_tensor(
        ds4_gpu_tensor *gated, ds4_gpu_tensor *normed, const ds4_gpu_tensor *R,
        const ds4_gpu_tensor *key, const ds4_gpu_tensor *value,
        const void *map, uint64_t size, uint64_t ko, uint64_t qo, uint64_t co,
        uint32_t T, uint32_t E, uint32_t hc, float eps) {
    const uint64_t bytes = (uint64_t)T * E * hc * 4u, wb = (uint64_t)E * hc * 4u;
    if (!T || !E || !hc || hc > 4u || !gated || !normed || !R || !key || !value ||
        gated->bytes < bytes || normed->bytes < bytes || R->bytes < bytes || key->bytes < bytes ||
        value->bytes < (uint64_t)T * E * 4u || !map ||
        ko > size || wb > size - ko || qo > size || wb > size - qo ||
        co > size || wb > size - co) return 0;
    const float *gk = (const float *)cuda_model_range_ptr(map, ko, wb, "Qwen PLE key gamma");
    const float *gq = (const float *)cuda_model_range_ptr(map, qo, wb, "Qwen PLE query gamma");
    const float *gc = (const float *)cuda_model_range_ptr(map, co, wb, "Qwen PLE conv gamma");
    if (!gk || !gq || !gc) return 0;
    qwen4_rocm_ple_gate_kernel<<<dim3(hc, T), 1>>>(
        (float *)gated->ptr, (float *)normed->ptr, (const float *)R->ptr,
        (const float *)key->ptr, (const float *)value->ptr, gk, gq, gc, E, hc, eps);
    return cuda_ok(cudaGetLastError(), "Qwen PLE gate");
}

extern "C" int ds4_gpu_qwen4_ple_conv_tensor(
        ds4_gpu_tensor *R, const ds4_gpu_tensor *gated, const ds4_gpu_tensor *normed,
        ds4_gpu_tensor *history, const void *map, uint64_t size, uint64_t off,
        uint32_t type, uint32_t T, uint32_t C, uint32_t K, uint32_t dilation,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    const uint64_t bytes = (uint64_t)T * C * 4u;
    const uint64_t hb = (uint64_t)(K ? K - 1u : 0u) * dilation * C * 4u;
    const uint64_t wb = (uint64_t)C * K * (type == 0u ? 4u : 2u);
    if (!T || !C || K < 2u || K > 4u || !dilation || dilation > 3u ||
        (type != 0u && type != 1u) || !R || !gated || !normed || !history ||
        R->bytes < bytes || gated->bytes < bytes || normed->bytes < bytes ||
        history->bytes < hb || (snap && (st >= T || snap->bytes < hb)) ||
        (snap2 && (st2 >= T || snap2->bytes < hb)) ||
        !map || off > size || wb > size - off) return 0;
    const char *w = cuda_model_range_ptr(map, off, wb, "Qwen PLE conv");
    if (!w) return 0;
    qwen4_rocm_ple_conv_kernel<<<(C + 255u) / 256u, 256>>>(
        (float *)R->ptr, (const float *)gated->ptr, (const float *)normed->ptr,
        (float *)history->ptr, w, type, T, C, K, dilation,
        snap ? (float *)snap->ptr : NULL, st, snap2 ? (float *)snap2->ptr : NULL, st2);
    return cuda_ok(cudaGetLastError(), "Qwen PLE conv");
}

__global__ static void qwen4_rocm_mtp_stage_kernel(
        float *cat, const float *e, const float *R, const float *ge,
        const float *gh, uint32_t E, uint32_t hc, float eps) {
    const uint32_t row = blockIdx.x;
    if (threadIdx.x) return;
    const bool emb = row == 0u;
    const float *src = emb ? e : R + (uint64_t)(row - 1u) * E;
    const float *g = emb ? ge : gh + (uint64_t)(row - 1u) * E;
    float ss = 0.0f;
    const float *norm_src = emb ? e : R;
    const uint32_t n = emb ? E : E * hc;
    for (uint32_t j = 0; j < n; j++) ss += norm_src[j] * norm_src[j];
    const float inv = rsqrtf(ss / n + eps);
    float *out = cat + (uint64_t)row * 2u * E;
    for (uint32_t i = 0; i < E; i++) {
        out[(emb ? 0u : E) + i] = src[i] * inv * g[i];
        out[(emb ? E : 0u) + i] = 0.0f;
    }
}

__global__ static void qwen4_rocm_mtp_combine_kernel(
        float *out, const float *proj, uint32_t E, uint32_t hc) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (uint64_t)E * hc) out[i] = proj[i % E] + proj[E + i];
}

extern "C" int ds4_gpu_qwen4_mtp_stage_tensor(
        ds4_gpu_tensor *cat, const ds4_gpu_tensor *e, const ds4_gpu_tensor *R,
        const void *map, uint64_t size, uint64_t eo, uint64_t ho,
        uint32_t E, uint32_t hc, float eps) {
    if (!E || !hc || hc > 4u || !cat || !e || !R || !map ||
        cat->bytes < (uint64_t)(hc + 1u) * 2u * E * 4u ||
        e->bytes < (uint64_t)E * 4u || R->bytes < (uint64_t)hc * E * 4u ||
        eo > size || (uint64_t)E * 4u > size - eo ||
        ho > size || (uint64_t)hc * E * 4u > size - ho) return 0;
    const float *ge = (const float *)cuda_model_range_ptr(map, eo, (uint64_t)E * 4u, "Qwen MTP embedding gamma");
    const float *gh = (const float *)cuda_model_range_ptr(map, ho, (uint64_t)hc * E * 4u, "Qwen MTP hidden gamma");
    if (!ge || !gh) return 0;
    qwen4_rocm_mtp_stage_kernel<<<hc + 1u, 1>>>(
        (float *)cat->ptr, (const float *)e->ptr, (const float *)R->ptr, ge, gh, E, hc, eps);
    return cuda_ok(cudaGetLastError(), "Qwen MTP stage");
}

extern "C" int ds4_gpu_qwen4_mtp_combine_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *proj, uint32_t E, uint32_t hc) {
    if (!E || !hc || hc > 4u || !out || !proj ||
        out->bytes < (uint64_t)hc * E * 4u ||
        proj->bytes < (uint64_t)(hc + 1u) * E * 4u) return 0;
    qwen4_rocm_mtp_combine_kernel<<<((uint64_t)hc * E + 255u) / 256u, 256>>>(
        (float *)out->ptr, (const float *)proj->ptr, E, hc);
    return cuda_ok(cudaGetLastError(), "Qwen MTP combine");
}
