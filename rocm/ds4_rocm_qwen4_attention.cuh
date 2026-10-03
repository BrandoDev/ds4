/* Correctness-first Qwen attention and RoPE. The sparse token IDs are the
 * output of the existing ROCm index score/select/expand pipeline. */
static qwen4_rocm_rope_args qwen4_rocm_make_rope(uint32_t nrot, float base) {
    qwen4_rocm_rope_args a = {};
    a.scale = qwen4_rocm_rope_scale;
    a.custom = qwen4_rocm_rope_custom ? 1u : 0u;
    for (uint32_t i = 0; i < nrot / 2u; i++)
        a.freq[i] = qwen4_rocm_rope_custom ? qwen4_rocm_rope_freq[i] : powf(base, -2.0f * i / nrot);
    return a;
}

__device__ static void qwen4_rocm_apply_rope(float *row, const uint32_t *pos,
                                              qwen4_rocm_rope_args a, uint32_t nrot) {
    for (uint32_t i = 0; i < nrot / 2u; i++) {
        const float theta = (float)pos[i % 3u] * a.freq[i];
        const float c = cosf(theta) * a.scale, s = sinf(theta) * a.scale;
        const float u = row[i], v = row[i + nrot / 2u];
        row[i] = u * c - v * s;
        row[i + nrot / 2u] = u * s + v * c;
    }
}

__global__ static void qwen4_rocm_attn_prep_kernel(
        float *qout, float *gate, __half *kc, __half *vc, float *iqout,
        float *ikc, const float *qg, const float *kp, const float *vp,
        const float *iq, const float *ik, const uint32_t *pos3,
        const float *gq, const float *gk, const float *giq,
        uint32_t H, uint32_t Hkv, uint32_t D, uint32_t Hi, uint32_t Di,
        uint32_t pos0, uint32_t nrot, float eps, qwen4_rocm_rope_args rp) {
    if (threadIdx.x) return;
    const uint32_t slot = blockIdx.x, t = blockIdx.y, pos = pos0 + t;
    if (slot == H + Hkv + Hi) {
        for (uint32_t i = 0; i < Di; i++) ikc[(uint64_t)pos * Di + i] = ik[(uint64_t)t * Di + i];
        return;
    }
    const bool isq = slot < H, isk = !isq && slot < H + Hkv;
    const uint32_t h = isq ? slot : isk ? slot - H : slot - H - Hkv;
    const uint32_t dim = isq || isk ? D : Di;
    const float *src = isq ? qg + ((uint64_t)t * H + h) * 2u * D :
                       isk ? kp + ((uint64_t)t * Hkv + h) * D :
                             iq + ((uint64_t)t * Hi + h) * Di;
    const float *gamma = isq ? gq : isk ? gk : giq;
    float row[256];
    float ss = 0.0f;
    for (uint32_t i = 0; i < dim; i++) ss += src[i] * src[i];
    const float inv = rsqrtf(ss / dim + eps);
    for (uint32_t i = 0; i < dim; i++) row[i] = src[i] * inv * gamma[i];
    qwen4_rocm_apply_rope(row, pos3 + (uint64_t)pos * 4u, rp, nrot);
    for (uint32_t i = 0; i < dim; i++) {
        if (isq) {
            qout[((uint64_t)t * H + h) * D + i] = row[i];
            gate[((uint64_t)t * H + h) * D + i] = src[D + i];
        } else if (isk) {
            kc[((uint64_t)pos * Hkv + h) * D + i] = __float2half_rn(row[i]);
            vc[((uint64_t)pos * Hkv + h) * D + i] =
                __float2half_rn(vp[((uint64_t)t * Hkv + h) * D + i]);
        } else iqout[((uint64_t)t * Hi + h) * Di + i] = row[i];
    }
}

extern "C" int ds4_gpu_qwen4_attn_prep_tensor(
        ds4_gpu_tensor *q, ds4_gpu_tensor *gate, ds4_gpu_tensor *kc, ds4_gpu_tensor *vc,
        ds4_gpu_tensor *iqout, ds4_gpu_tensor *ikc, const ds4_gpu_tensor *qg,
        const ds4_gpu_tensor *kp, const ds4_gpu_tensor *vp, const ds4_gpu_tensor *iq,
        const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3,
        const void *map, uint64_t size, uint64_t qo, uint64_t ko, uint64_t io,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t nrot,
        uint32_t Hi, uint32_t Di, uint32_t pos0, uint32_t cap, float base, float eps) {
    const uint64_t qb = (uint64_t)T * H * D * 4u, kb = (uint64_t)T * Hkv * D * 4u;
    const uint64_t ib = (uint64_t)T * Hi * Di * 4u;
    if (!T || !H || !Hkv || H % Hkv || !Hi || D < 32u || D > 256u || D % 32u ||
        Di < 32u || Di > 128u || Di % 32u || nrot > 64u || nrot > D || nrot > Di || nrot % 2u ||
        (uint64_t)pos0 + T > cap || !q || !gate || !kc || !vc || !iqout || !ikc ||
        !qg || !kp || !vp || !iq || !ik || !pos3 || !map ||
        q->bytes < qb || gate->bytes < qb || qg->bytes < qb * 2u ||
        kp->bytes < kb || vp->bytes < kb || iq->bytes < ib || iqout->bytes < ib ||
        ik->bytes < (uint64_t)T * Di * 4u || ikc->bytes < (uint64_t)cap * Di * 4u ||
        kc->bytes < (uint64_t)cap * Hkv * D * 2u || vc->bytes < (uint64_t)cap * Hkv * D * 2u ||
        pos3->bytes < (uint64_t)cap * 16u ||
        qo > size || (uint64_t)D * 4u > size - qo ||
        ko > size || (uint64_t)D * 4u > size - ko ||
        io > size || (uint64_t)Di * 4u > size - io) return 0;
    const float *gq = (const float *)cuda_model_range_ptr(map, qo, D * 4u, "Qwen query gamma");
    const float *gk = (const float *)cuda_model_range_ptr(map, ko, D * 4u, "Qwen key gamma");
    const float *giq = (const float *)cuda_model_range_ptr(map, io, Di * 4u, "Qwen index query gamma");
    if (!gq || !gk || !giq) return 0;
    qwen4_rocm_attn_prep_kernel<<<dim3(H + Hkv + Hi + 1u, T), 1>>>(
        (float *)q->ptr, (float *)gate->ptr, (__half *)kc->ptr, (__half *)vc->ptr,
        (float *)iqout->ptr, (float *)ikc->ptr, (const float *)qg->ptr,
        (const float *)kp->ptr, (const float *)vp->ptr, (const float *)iq->ptr,
        (const float *)ik->ptr, (const uint32_t *)pos3->ptr, gq, gk, giq,
        H, Hkv, D, Hi, Di, pos0, nrot, eps, qwen4_rocm_make_rope(nrot, base));
    return cuda_ok(cudaGetLastError(), "Qwen attention prep");
}

__global__ static void qwen4_rocm_attn_decode_kernel(
        float *out, const float *q, const float *gate, const __half *kc,
        const __half *vc, const int *sel, const uint32_t *counts,
        uint32_t H, uint32_t Hkv, uint32_t D, uint32_t pos0,
        bool sparse, uint32_t stride, float scale) {
    if (threadIdx.x) return;
    const uint32_t h = blockIdx.x, t = blockIdx.y, kh = h / (H / Hkv);
    const uint32_t n = sparse ? counts[t] : pos0 + t + 1u;
    float acc[256] = {};
    float mx = -3e38f, denom = 0.0f;
    const float *qh = q + ((uint64_t)t * H + h) * D;
    for (uint32_t j = 0; j < n; j++) {
        const uint32_t p = sparse ? (uint32_t)sel[(uint64_t)t * stride + j] : j;
        if (p > pos0 + t) continue;
        const __half *k = kc + ((uint64_t)p * Hkv + kh) * D;
        const __half *v = vc + ((uint64_t)p * Hkv + kh) * D;
        float score = 0.0f;
        for (uint32_t d = 0; d < D; d++) score += qh[d] * scale * __half2float(k[d]);
        const float nm = fmaxf(mx, score);
        const float correction = expf(mx - nm), w = expf(score - nm);
        denom = denom * correction + w;
        for (uint32_t d = 0; d < D; d++) acc[d] = acc[d] * correction + w * __half2float(v[d]);
        mx = nm;
    }
    const uint64_t off = ((uint64_t)t * H + h) * D;
    for (uint32_t d = 0; d < D; d++)
        out[off + d] = (denom > 0.0f ? acc[d] / denom : 0.0f) * qwen4_rocm_sigmoid(gate[off + d]);
}

extern "C" int ds4_gpu_qwen4_attn_decode_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *kc, const ds4_gpu_tensor *vc,
        const ds4_gpu_tensor *sel, const ds4_gpu_tensor *counts, ds4_gpu_tensor *part,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D,
        uint32_t pos0, bool sparse, uint32_t stride, float scale) {
    (void)part;
    const uint64_t bytes = (uint64_t)T * H * D * 4u;
    if (!T || !H || !Hkv || H % Hkv || !D || D > 256u || !out || !q || !gate || !kc || !vc ||
        out->bytes < bytes || q->bytes < bytes || gate->bytes < bytes ||
        kc->bytes < (uint64_t)(pos0 + T) * Hkv * D * 2u ||
        vc->bytes < (uint64_t)(pos0 + T) * Hkv * D * 2u ||
        (sparse && (!sel || !counts || sel->bytes < (uint64_t)T * stride * 4u ||
                    counts->bytes < (uint64_t)T * 4u))) return 0;
    qwen4_rocm_attn_decode_kernel<<<dim3(H, T), 1>>>(
        (float *)out->ptr, (const float *)q->ptr, (const float *)gate->ptr,
        (const __half *)kc->ptr, (const __half *)vc->ptr,
        sparse ? (const int *)sel->ptr : NULL, sparse ? (const uint32_t *)counts->ptr : NULL,
        H, Hkv, D, pos0, sparse, stride, scale);
    return cuda_ok(cudaGetLastError(), "Qwen attention decode");
}

extern "C" int ds4_gpu_qwen4_vision_encode(
        float *out, const float *patches, const float *pos_embed,
        uint32_t n_patches, uint32_t grid_w, const void *model_map,
        uint64_t model_size, const ds4_qwen4_vision_weights *w) {
    (void)out; (void)patches; (void)pos_embed; (void)n_patches;
    (void)grid_w; (void)model_map; (void)model_size; (void)w;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "Qwen vision encoder is unavailable on ROCm\n");
    return 0;
}
