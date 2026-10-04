/* Correctness-first routed Qwen MoE for Q2_K and IQ3_XXS, plus dense/Q8
 * reference formats. Each output row owns its reduction; activations stay
 * on device, including the logical 640-value IQ3/Q2 down input. */
static uint64_t qwen4_rocm_expert_row_bytes(uint32_t type, uint32_t K) {
    if (type == 10u) return K % 32u ? 0u : ((uint64_t)K + 255u) / 256u * 84u;
    if (type == 12u) return K % 256u ? 0u : (uint64_t)K / 256u * 144u;
    if (type == 16u) return K % 256u ? 0u : (uint64_t)K / 256u * 66u;
    if (type == 18u) return ((uint64_t)K + 255u) / 256u * 98u;
    if (type == 39u) return K % 32u ? 0u : (uint64_t)K / 32u * 17u;
    return qwen4_rocm_row_bytes(type, K);
}

/* Stage the union of routed experts, keeping their rows byte-identical.
 * Reading IDs waits for the router (and the preceding projection); the
 * default-stream copy orders reuse of the shared temporary buffer after its
 * previous kernel. Mid and down therefore need no persistent weight cache. */
static int qwen4_rocm_selected_weights(
        const ds4_gpu_tensor *sel, const void *map, uint64_t off, uint64_t up_off,
        uint32_t NE, uint32_t T, uint32_t NS, uint64_t expert_bytes, bool pair,
        const char **weights, const char **up, const int **remap) {
    *remap = NULL;
    /* Already mapped/resident weights need no host-to-device staging. */
    if (NE > 512u || g_model_range_mapping_supported || cuda_model_image_range_ptr(map, off, NE * expert_bytes) ||
        getenv("DS4_QWEN4_FULL_EXPERT_COPIES")) {
        *weights = cuda_model_range_ptr(map, off, NE * expert_bytes, "Qwen expert reference");
        if (pair) *up = cuda_model_range_ptr(map, up_off, NE * expert_bytes, "Qwen expert up reference");
        return *weights && (!pair || *up);
    }
    int indices[512];
    for (uint32_t e = 0; e < NE; e++) indices[e] = -1;
    const size_t n_ids = (size_t)T * NS;
    int *ids = (int *)malloc(n_ids * sizeof(int));
    if (!ids) return 0;
    if (!cuda_ok(cudaMemcpy(ids, sel->ptr, n_ids * sizeof(int),
                            cudaMemcpyDeviceToHost), "Qwen selected expert IDs")) {
        free(ids);
        return 0;
    }
    for (size_t i = 0; i < n_ids; i++) {
        const int id = ids[i];
        if (id < 0 || (uint32_t)id >= NE) { free(ids); return 0; }
        indices[id] = 0;
    }
    free(ids);
    uint32_t count = 0;
    for (uint32_t e = 0; e < NE; e++) if (indices[e] == 0) indices[e] = (int)count++;
    const uint64_t bytes = count * expert_bytes;
    const uint64_t indices_offset = (bytes * (pair ? 2u : 1u) + 3u) & ~3ull;
    /* Even a fully populated union fits exactly where the old full tables
     * fit; no assumptions about token count or routing distribution. */
    char *packed = (char *)cuda_tmp_alloc(indices_offset + NE * sizeof(int),
                                         "Qwen selected expert weights");
    if (!packed) return 0;
    for (uint32_t e = 0; e < NE;) {
        if (indices[e] < 0) { e++; continue; }
        const uint32_t begin = e;
        while (e < NE && indices[e] >= 0) e++;
        const uint64_t n = (e - begin) * expert_bytes;
        char *dst = packed + (uint64_t)indices[begin] * expert_bytes;
        if (!cuda_ok(cudaMemcpy(dst, (const char *)map + off + begin * expert_bytes,
                                n, cudaMemcpyHostToDevice), "Qwen selected expert weights")) return 0;
        if (pair && !cuda_ok(cudaMemcpy(dst + bytes,
                (const char *)map + up_off + begin * expert_bytes,
                n, cudaMemcpyHostToDevice), "Qwen selected expert up weights")) return 0;
    }
    int *device_indices = (int *)(packed + indices_offset);
    if (!cuda_ok(cudaMemcpy(device_indices, indices, NE * sizeof(int),
                            cudaMemcpyHostToDevice), "Qwen compact expert indices")) return 0;
    *weights = packed;
    if (pair) *up = packed + bytes;
    *remap = device_indices;
    return 1;
}

__device__ static float qwen4_rocm_expert_dot(
        const char *row, const float *x, uint32_t type, uint32_t K) {
    if (type == 18u) return qwen4_rocm_iq3_dot((const uint8_t *)row, x, K, ((K + 255u) / 256u) * 256u);
    float sum = 0.0f;
    if (type == 16u) {
        for (uint32_t b = 0; b < K / 256u; b++) {
            const uint8_t *p = (const uint8_t *)row + (uint64_t)b * 66u;
            const float d = __half2float(*reinterpret_cast<const __half *>(p));
            for (uint32_t g = 0; g < 8u; g++) {
                const uint8_t *q = p + 2u + g * 8u;
                const uint32_t aux = (uint32_t)q[4] | ((uint32_t)q[5] << 8u) |
                                     ((uint32_t)q[6] << 16u) | ((uint32_t)q[7] << 24u);
                const float scale = 0.125f * d * (float)(2u * (aux >> 28u) + 1u);
                for (uint32_t pair = 0; pair < 4u; pair++) {
                    const uint64_t grid = cuda_iq2xxs_grid[q[pair]];
                    const uint8_t signs = cuda_ksigns_iq2xs[(aux >> (7u * pair)) & 127u];
                    for (uint32_t j = 0; j < 8u; j++) {
                        const uint32_t k = b * 256u + g * 32u + pair * 8u + j;
                        const float v = (float)((grid >> (j * 8u)) & 255u);
                        sum += scale * ((signs & (1u << j)) ? -v : v) * x[k];
                    }
                }
            }
        }
    } else if (type == 10u) {
        for (uint32_t i = 0; i < K; i++) {
            const cuda_block_q2_K *b = ((const cuda_block_q2_K *)row) + i / 256u;
            const uint32_t j = i & 255u, s = b->scales[j / 16u];
            const uint32_t q = (b->qs[(j / 128u) * 32u + j % 32u] >>
                                (2u * ((j % 128u) / 32u))) & 3u;
            const float d = __half2float(*reinterpret_cast<const __half *>(&b->d));
            const float dm = __half2float(*reinterpret_cast<const __half *>(&b->dmin));
            sum += (d * (s & 15u) * q - dm * (s >> 4u)) * x[i];
        }
    } else if (type == 12u) {
        for (uint32_t i = 0; i < K; i++) {
            const uint8_t *p = (const uint8_t *)row + (uint64_t)(i / 256u) * 144u;
            const uint32_t j = i & 255u, g = j / 32u;
            const uint8_t *sc = p + 4u;
            uint32_t s, mn;
            if (g < 4u) {
                s = sc[g] & 63u;
                mn = sc[g + 4u] & 63u;
            } else {
                s = (sc[g + 4u] & 15u) | ((sc[g - 4u] & 0xc0u) >> 2u);
                mn = (sc[g + 4u] >> 4u) | ((sc[g] & 0xc0u) >> 2u);
            }
            const uint32_t q = (p[16u + (g >> 1u) * 32u + (j & 31u)] >> ((g & 1u) * 4u)) & 15u;
            const float d = __half2float(*reinterpret_cast<const __half *>(p));
            const float dm = __half2float(*reinterpret_cast<const __half *>(p + 2u));
            sum += (d * (float)s * (float)q - dm * (float)mn) * x[i];
        }
    } else if (type == 39u) {
        for (uint32_t i = 0; i < K; i++) {
            const uint8_t *p = (const uint8_t *)row + (uint64_t)(i / 32u) * 17u;
            const uint32_t j = i & 31u;
            const uint8_t packed = p[1u + (j & 15u)];
            const uint32_t code = j < 16u ? packed & 15u : packed >> 4u;
            const uint32_t mag_code = code & 7u;
            const float mag = mag_code <= 4u ? 0.5f * (float)mag_code :
                              mag_code == 5u ? 3.0f : mag_code == 6u ? 4.0f : 6.0f;
            const float v = (code & 8u) ? -mag : mag;
            sum += dev_e8m0_to_f32(p[0]) * v * x[i];
        }
    } else {
        for (uint32_t i = 0; i < K; i++) sum += qwen4_rocm_weight_at(row, type, i) * x[i];
    }
    return sum;
}

__global__ static void qwen4_rocm_moe_mid_kernel(
        float *mid, const float *x, const int *sel, const int *remap,
        const char *gate, const char *up, const char *sh_gate, const char *sh_up,
        uint32_t type, uint32_t shared_type, uint32_t NE, uint32_t T,
        uint32_t NS, uint32_t K, uint32_t M, uint64_t rb, uint64_t srb) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t slot = blockIdx.y, t = blockIdx.z;
    if (row >= M) return;
    const bool shared = slot == NS;
    const uint32_t stride = NS + (shared_type != UINT32_MAX);
    const char *wg, *wu;
    uint32_t wt;
    if (shared) {
        wg = sh_gate + (uint64_t)row * srb;
        wu = sh_up + (uint64_t)row * srb;
        wt = shared_type;
    } else {
        const int expert = sel[(uint64_t)t * NS + slot];
        if (expert < 0 || (uint32_t)expert >= NE) return;
        const uint32_t packed = remap ? (uint32_t)remap[expert] : (uint32_t)expert;
        wg = gate + ((uint64_t)packed * M + row) * rb;
        wu = up + ((uint64_t)packed * M + row) * rb;
        wt = type;
    }
    const float *xt = x + (uint64_t)t * K;
    const float a = qwen4_rocm_expert_dot(wg, xt, wt, K);
    const float b = qwen4_rocm_expert_dot(wu, xt, wt, K);
    mid[((uint64_t)t * stride + slot) * M + row] = qwen4_rocm_silu(a) * b;
}

__global__ static void qwen4_rocm_moe_down_kernel(
        float *part, const float *mid, const int *sel, const int *remap,
        const char *down, const char *sh_down, uint32_t type,
        uint32_t shared_type, uint32_t NE, uint32_t NS, uint32_t K,
        uint32_t M, uint64_t rb, uint64_t srb) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t slot = blockIdx.y, t = blockIdx.z;
    if (row >= M) return;
    const bool shared = slot == NS;
    const uint32_t stride = NS + (shared_type != UINT32_MAX);
    const char *w;
    uint32_t wt;
    if (shared) {
        w = sh_down + (uint64_t)row * srb;
        wt = shared_type;
    } else {
        const int expert = sel[(uint64_t)t * NS + slot];
        if (expert < 0 || (uint32_t)expert >= NE) return;
        const uint32_t packed = remap ? (uint32_t)remap[expert] : (uint32_t)expert;
        w = down + ((uint64_t)packed * M + row) * rb;
        wt = type;
    }
    const float *x = mid + ((uint64_t)t * stride + slot) * K;
    part[((uint64_t)t * stride + slot) * M + row] = qwen4_rocm_expert_dot(w, x, wt, K);
}

extern "C" int ds4_gpu_qwen4_moe_mid_tensor(
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *sel,
        const void *map, uint64_t size, uint64_t go, uint64_t uo,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t K, uint32_t M, uint64_t sgo, uint64_t suo, uint32_t st) {
    const uint32_t stride = NS + (st != UINT32_MAX);
    const uint64_t rb = qwen4_rocm_expert_row_bytes(type, K);
    const uint64_t srb = st == UINT32_MAX ? 0u : qwen4_rocm_expert_row_bytes(st, K);
    const uint64_t bytes = (uint64_t)NE * M * rb, sb = (uint64_t)M * srb;
    if (!T || !NE || !NS || !K || !M || !rb || (st != UINT32_MAX && !srb) ||
        !mid || !x || !sel || !map ||
        mid->bytes < (uint64_t)T * stride * M * 4u || x->bytes < (uint64_t)T * K * 4u ||
        sel->bytes < (uint64_t)T * NS * 4u ||
        go > size || bytes > size - go || uo > size || bytes > size - uo ||
        (st != UINT32_MAX && (sgo > size || sb > size - sgo || suo > size || sb > size - suo))) return 0;
    const char *g = NULL, *u = NULL;
    const int *remap = NULL;
    if (!qwen4_rocm_selected_weights(sel, map, go, uo, NE, T, NS, M * rb, true,
                                     &g, &u, &remap)) return 0;
    const char *sg = st == UINT32_MAX ? NULL : cuda_model_range_ptr(map, sgo, sb, "Qwen shared gate");
    const char *su = st == UINT32_MAX ? NULL : cuda_model_range_ptr(map, suo, sb, "Qwen shared up");
    if (!g || !u || (st != UINT32_MAX && (!sg || !su))) return 0;
    qwen4_rocm_moe_mid_kernel<<<dim3((M + 63u) / 64u, stride, T), 64>>>(
        (float *)mid->ptr, (const float *)x->ptr, (const int *)sel->ptr, remap,
        g, u, sg, su, type, st, NE, T, NS, K, M, rb, srb);
    return cuda_ok(cudaGetLastError(), "Qwen MoE mid");
}

extern "C" int ds4_gpu_qwen4_moe_down_tensor(
        ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *sel,
        const void *map, uint64_t size, uint64_t off, uint32_t type,
        uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t so, uint32_t st) {
    const uint32_t stride = NS + (st != UINT32_MAX);
    const uint64_t rb = qwen4_rocm_expert_row_bytes(type, K);
    const uint64_t srb = st == UINT32_MAX ? 0u : qwen4_rocm_expert_row_bytes(st, K);
    const uint64_t bytes = (uint64_t)NE * M * rb, sb = (uint64_t)M * srb;
    if (!T || !NE || !NS || !K || !M || !rb || (st != UINT32_MAX && !srb) ||
        !part || !mid || !sel || !map ||
        part->bytes < (uint64_t)T * stride * M * 4u || mid->bytes < (uint64_t)T * stride * K * 4u ||
        sel->bytes < (uint64_t)T * NS * 4u ||
        off > size || bytes > size - off ||
        (st != UINT32_MAX && (so > size || sb > size - so))) return 0;
    const char *w = NULL;
    const int *remap = NULL;
    if (!qwen4_rocm_selected_weights(sel, map, off, 0u, NE, T, NS, M * rb, false,
                                     &w, NULL, &remap)) return 0;
    const char *sw = st == UINT32_MAX ? NULL : cuda_model_range_ptr(map, so, sb, "Qwen shared down");
    if (!w || (st != UINT32_MAX && !sw)) return 0;
    qwen4_rocm_moe_down_kernel<<<dim3((M + 63u) / 64u, stride, T), 64>>>(
        (float *)part->ptr, (const float *)mid->ptr, (const int *)sel->ptr, remap,
        w, sw, type, st, NE, NS, K, M, rb, srb);
    return cuda_ok(cudaGetLastError(), "Qwen MoE down");
}

__global__ static void qwen4_rocm_moe_reduce_kernel(
        float *out, float *R, const float *inj, const float *part,
        const float *weights, const float *gate, const float *shared,
        uint32_t NS, uint32_t stride, uint32_t D, uint32_t hc) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (d >= D) return;
    float v = 0.0f;
    for (uint32_t s = 0; s < NS; s++)
        v += weights[(uint64_t)t * NS + s] * part[((uint64_t)t * stride + s) * D + d];
    if (gate) v += qwen4_rocm_sigmoid(gate[t]) *
                   (shared ? shared[(uint64_t)t * D + d] : part[((uint64_t)t * stride + NS) * D + d]);
    out[(uint64_t)t * D + d] = v;
    for (uint32_t s = 0; s < hc; s++) {
        float a = 0.0f;
        const float *it = inj + (uint64_t)t * hc * hc * 8u;
        for (uint32_t i = 0; i < hc * 8u; i++) a += it[i * hc + s];
        R[((uint64_t)t * hc + s) * D + d] += 2.0f * qwen4_rocm_sigmoid(a / hc) * v;
    }
}

extern "C" int ds4_gpu_qwen4_moe_reduce_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *part, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *sg, const ds4_gpu_tensor *sh, ds4_gpu_tensor *R,
        const ds4_gpu_tensor *inj, uint32_t T, uint32_t NS, uint32_t stride,
        uint32_t D, uint32_t hc) {
    if (!T || !NS || !D || stride < NS + (sg && !sh) || hc > 4u ||
        !out || !part || !weights || out->bytes < (uint64_t)T * D * 4u ||
        part->bytes < (uint64_t)T * stride * D * 4u ||
        weights->bytes < (uint64_t)T * NS * 4u ||
        (sg && sg->bytes < (uint64_t)T * 4u) ||
        (sh && sh->bytes < (uint64_t)T * D * 4u) ||
        (hc && (!R || !inj || R->bytes < (uint64_t)T * hc * D * 4u ||
                inj->bytes < (uint64_t)T * hc * hc * 8u * 4u))) return 0;
    qwen4_rocm_moe_reduce_kernel<<<dim3((D + 255u) / 256u, T), 256>>>(
        (float *)out->ptr, hc ? (float *)R->ptr : NULL, hc ? (const float *)inj->ptr : NULL,
        (const float *)part->ptr, (const float *)weights->ptr,
        sg ? (const float *)sg->ptr : NULL, sh ? (const float *)sh->ptr : NULL,
        NS, stride, D, hc);
    return cuda_ok(cudaGetLastError(), "Qwen MoE reduce");
}

__global__ static void qwen4_rocm_q8_weights_kernel(
        float *out, const char *w, const float *x, uint32_t K, uint32_t M, uint64_t rb) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) out[row] = qwen4_rocm_expert_dot(w + (uint64_t)row * rb, x, 8u, K);
}

extern "C" int ds4_gpu_qwen4_matmul_q8_0_weights_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *w, uint32_t K,
        uint32_t M, const ds4_gpu_tensor *x) {
    const uint64_t rb = qwen4_rocm_row_bytes(8u, K);
    if (!K || !M || !rb || !out || !w || !x ||
        out->bytes < (uint64_t)M * 4u || w->bytes < rb * M ||
        x->bytes < (uint64_t)K * 4u) return 0;
    qwen4_rocm_q8_weights_kernel<<<(M + 63u) / 64u, 64>>>(
        (float *)out->ptr, (const char *)w->ptr, (const float *)x->ptr, K, M, rb);
    return cuda_ok(cudaGetLastError(), "Qwen Q8 tensor weights");
}

__global__ static void qwen4_rocm_moe_lists_kernel(
        int *lists, int *counts, const int *sel, uint32_t pairs,
        uint32_t NE, uint32_t cap) {
    if (threadIdx.x) return;
    for (uint32_t e = 0; e < NE; e++) counts[e] = 0;
    for (uint32_t p = 0; p < pairs; p++) {
        const int e = sel[p];
        if (e < 0 || (uint32_t)e >= NE) continue;
        const uint32_t n = (uint32_t)counts[e];
        if (n < cap) { lists[(uint64_t)e * cap + n] = (int)p; counts[e] = (int)(n + 1u); }
    }
}

extern "C" int ds4_gpu_qwen4_moe_build_lists_tensor(
        ds4_gpu_tensor *lists, ds4_gpu_tensor *counts, const ds4_gpu_tensor *sel,
        uint32_t T, uint32_t NS, uint32_t NE, uint32_t cap) {
    if (!T || !NS || !NE || NE > 512u || NS > NE || cap < T ||
        !lists || !counts || !sel ||
        lists->bytes < (uint64_t)NE * cap * 4u ||
        counts->bytes < (uint64_t)NE * 4u ||
        sel->bytes < (uint64_t)T * NS * 4u) return 0;
    qwen4_rocm_moe_lists_kernel<<<1, 1>>>(
        (int *)lists->ptr, (int *)counts->ptr, (const int *)sel->ptr, T * NS, NE, cap);
    return cuda_ok(cudaGetLastError(), "Qwen MoE expert lists");
}

__global__ static void qwen4_rocm_moe_mm_kernel(
        float *out, const float *x, const int *lists, const int *counts,
        const char *w0, const char *w1, uint32_t type, uint32_t NE,
        uint32_t NS, uint32_t NO, uint32_t K, uint32_t M,
        uint32_t cap, uint64_t rb, bool down) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x, e = blockIdx.y;
    if (row >= M || e >= NE) return;
    const char *a = w0 + ((uint64_t)e * M + row) * rb;
    const char *b = down ? NULL : w1 + ((uint64_t)e * M + row) * rb;
    const uint32_t n = (uint32_t)counts[e];
    for (uint32_t i = 0; i < n && i < cap; i++) {
        const uint32_t pair = (uint32_t)lists[(uint64_t)e * cap + i];
        const uint32_t t = pair / NS, slot = pair % NS;
        const float *xt = x + (down ? ((uint64_t)t * NO + slot) : t) * K;
        const float v = qwen4_rocm_expert_dot(a, xt, type, K);
        out[((uint64_t)t * NO + slot) * M + row] =
            down ? v : qwen4_rocm_silu(v) * qwen4_rocm_expert_dot(b, xt, type, K);
    }
}

static int qwen4_rocm_moe_mm(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts,
        const void *map, uint64_t size, uint64_t o0, uint64_t o1,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t NO,
        uint32_t K, uint32_t M, uint32_t cap, bool down) {
    const uint64_t rb = qwen4_rocm_expert_row_bytes(type, K);
    const uint64_t bytes = (uint64_t)NE * M * rb;
    if (!T || !NE || !NS || NS > NE || NO < NS || !K || !M || cap < T || !rb ||
        !out || !x || !lists || !counts || !map ||
        out->bytes < (uint64_t)T * NO * M * 4u ||
        x->bytes < (uint64_t)T * (down ? NO : 1u) * K * 4u ||
        lists->bytes < (uint64_t)NE * cap * 4u ||
        counts->bytes < (uint64_t)NE * 4u ||
        o0 > size || bytes > size - o0 ||
        (!down && (o1 > size || bytes > size - o1))) return 0;
    const char *w0 = cuda_model_range_ptr(map, o0, bytes, "Qwen MoE grouped weight");
    const char *w1 = down ? NULL : cuda_model_range_ptr(map, o1, bytes, "Qwen MoE grouped up");
    if (!w0 || (!down && !w1)) return 0;
    qwen4_rocm_moe_mm_kernel<<<dim3((M + 63u) / 64u, NE), 64>>>(
        (float *)out->ptr, (const float *)x->ptr, (const int *)lists->ptr,
        (const int *)counts->ptr, w0, w1, type, NE, NS, NO, K, M, cap, rb, down);
    return cuda_ok(cudaGetLastError(), "Qwen MoE grouped matmul");
}

extern "C" int ds4_gpu_qwen4_moe_mm_mid_tensor(
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *lists,
        const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t go, uint64_t uo, uint32_t type, uint32_t NE, uint32_t T,
        uint32_t NS, uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    return qwen4_rocm_moe_mm(mid, x, lists, counts, map, size,
                             go, uo, type, NE, T, NS, NO, K, M, cap, false);
}

extern "C" int ds4_gpu_qwen4_moe_mm_down_tensor(
        ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *lists,
        const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t NE, uint32_t T,
        uint32_t NS, uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    return qwen4_rocm_moe_mm(part, mid, lists, counts, map, size,
                             off, 0u, type, NE, T, NS, NO, K, M, cap, true);
}

extern "C" int ds4_gpu_qwen4_release_layer_weights(void) {
    if (!cuda_ok(cudaDeviceSynchronize(), "Qwen layer weight release sync")) return 0;
    cuda_model_range_release_ranges_only();
    return 1;
}

extern "C" void ds4_gpu_qwen4_use_device_weight_copies(void) {
    /* HIP 7.2 aborts while mapping ranges from this large file-backed GGUF
     * ("Memobj map does not have ptr"). The generic FD staging path also
     * pins host buffers. Copy each layer's ranges into VRAM, then release
     * them at the synchronized layer boundary instead. */
    g_model_fd = -1;
    g_model_range_mapping_supported = 0;
    static bool reported = false;
    if (!reported) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "Qwen weights: device copies per layer; host mapping disabled\n");
        reported = true;
    }
}
