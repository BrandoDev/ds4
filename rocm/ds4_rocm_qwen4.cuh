/* Qwen3.8 Flash Next - initial ROCm adapter layer.
 *
 * Start by reusing backend primitives that already have equivalent semantics.
 * Model-specific kernels are added separately as they are ported.
 */

extern "C" int ds4_gpu_qwen4_decode_fusions_enabled(void) {
    return 0;
}

extern "C" uint64_t ds4_gpu_qwen4_attn_part_floats(
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim) {
    return (uint64_t)n_tokens * n_head * 64u * (head_dim + 2u);
}

extern "C" int ds4_gpu_qwen4_argmax_tensor(
        ds4_gpu_tensor *out_idx,
        ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *logits,
        uint32_t n_vocab) {
    (void)scratch;
    return ds4_gpu_argmax_tensor(out_idx, logits, n_vocab);
}

extern "C" int ds4_gpu_qwen4_q8_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if ((out0_dim & 1u) || (out1_dim & 1u)) return 0;

    return ds4_gpu_matmul_q8_0_tensor(
               out0, model_map, model_size, weight0_offset,
               in_dim, out0_dim, x, n_tok) &&
           ds4_gpu_matmul_q8_0_tensor(
               out1, model_map, model_size, weight1_offset,
               in_dim, out1_dim, x, n_tok);
}


/* Qwen Q8_0 small-batch matvec.
 *
 * Preserve the accumulation geometry used by the native Qwen CUDA path:
 * four contiguous K values per lane, 32-lane XOR reduction.
 * This is numerically relevant to GDN alpha/beta projections.
 */
__device__ static inline float qwen4_rocm_wave32_sum_xor(float v) {
    for (uint32_t d = 16u; d != 0u; d >>= 1u) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        v += __shfl_xor(v, d, 32);
#else
        v += __shfl_xor_sync(0xffffffffu, v, d, 32);
#endif
    }
    return v;
}

__global__ static void qwen4_rocm_q8_small_batch_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t T,
        uint32_t K,
        uint32_t M,
        uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x * 4u + wave;

    if (row >= M) return;

    const unsigned char *wr =
        w + (uint64_t)row * row_bytes;

    float acc[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };

    for (uint32_t i = lane * 4u; i < K; i += 128u) {
        const unsigned char *blk =
            wr + (uint64_t)(i >> 5u) * 34u;

        const float scale =
            __half2float(*(const __half *)blk);

        const int8_t *q =
            (const int8_t *)(blk + 2u + (i & 31u));

        for (uint32_t t = 0; t < T; ++t) {
            const float *v =
                (const float *)x +
                (uint64_t)t * K + i;

            float part = (float)q[0] * v[0];
            part += (float)q[1] * v[1];
            part += (float)q[2] * v[2];
            part += (float)q[3] * v[3];

            acc[t] += part * scale;
        }
    }

    for (uint32_t t = 0; t < T; ++t) {
        const float sum =
            qwen4_rocm_wave32_sum_xor(acc[t]);

        if (lane == 0u) {
            out[(uint64_t)t * M + row] = sum;
        }
    }
}

extern "C" int ds4_gpu_qwen4_dense_mm_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t weight_type,
        uint32_t n_tokens,
        uint32_t in_dim,
        uint32_t out_rows) {

    /* Qwen F16 small-batch FP32-activation path.
     *
     * The generic ROCm F16 GEMM converts activations to FP16 for n_tokens > 1.
     * Qwen keeps FP32 activations for small batches; this matters for the
     * downstream HC gate numerical tolerance.
     */
    if (weight_type == 1u && n_tokens <= 8u) {
        if (!out || !x || !model_map ||
            !n_tokens || !in_dim || !out_rows) {
            return 0;
        }

        const uint64_t x_bytes =
            (uint64_t)n_tokens * in_dim * sizeof(float);

        const uint64_t out_bytes =
            (uint64_t)n_tokens * out_rows * sizeof(float);

        const uint64_t weight_bytes =
            (uint64_t)out_rows * in_dim * sizeof(uint16_t);

        if (x->bytes < x_bytes ||
            out->bytes < out_bytes ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            return 0;
        }

        const __half *w =
            (const __half *)cuda_model_range_ptr(
                model_map,
                weight_offset,
                weight_bytes,
                "Qwen F16 small-batch");

        if (!w) return 0;

        const dim3 grid(
            out_rows,
            n_tokens,
            1u);

        matmul_f16_kernel<<<grid, 256>>>(
            (float *)out->ptr,
            w,
            (const float *)x->ptr,
            in_dim,
            out_rows,
            n_tokens);

        return cuda_ok(
            cudaGetLastError(),
            "Qwen F16 small-batch FP32-activation matmul");
    }


    /* Qwen Q8 small-batch accumulation path.
     * Match the Qwen-specific reduction order instead of the generic ROCm
     * Q8 GEMM.  GDN decay is sensitive to projection rounding.
     */
    if (weight_type == 8u &&
        n_tokens <= 8u &&
        (in_dim % 32u) == 0u) {

        if (!out || !x || !model_map ||
            !n_tokens || !in_dim || !out_rows ||
            n_tokens > 8u) {
            return 0;
        }

        const uint64_t row_bytes =
            ((uint64_t)in_dim / 32u) * 34u;

        const uint64_t weight_bytes =
            row_bytes * out_rows;

        const uint64_t x_bytes =
            (uint64_t)n_tokens *
            in_dim * sizeof(float);

        const uint64_t out_bytes =
            (uint64_t)n_tokens *
            out_rows * sizeof(float);

        if (x->bytes < x_bytes ||
            out->bytes < out_bytes ||
            weight_offset > model_size ||
            weight_bytes > model_size - weight_offset) {
            return 0;
        }

        const unsigned char *w =
            (const unsigned char *)cuda_model_range_ptr(
                model_map,
                weight_offset,
                weight_bytes,
                "Qwen Q8 small-batch");

        if (!w) return 0;

        qwen4_rocm_q8_small_batch_kernel<<<
            (out_rows + 3u) / 4u,
            128
        >>>(
            (float *)out->ptr,
            w,
            (const float *)x->ptr,
            n_tokens,
            in_dim,
            out_rows,
            row_bytes);

        return cuda_ok(
            cudaGetLastError(),
            "Qwen Q8 small-batch matvec");
    }

    switch (weight_type) {
    case 0u:
        return ds4_gpu_matmul_f32_tensor(
            out, model_map, model_size, weight_offset,
            in_dim, out_rows, x, n_tokens);

    case 1u:
        return ds4_gpu_matmul_f16_tensor(
            out, model_map, model_size, weight_offset,
            in_dim, out_rows, x, n_tokens);

    case 8u:
        return ds4_gpu_matmul_q8_0_tensor(
            out, model_map, model_size, weight_offset,
            in_dim, out_rows, x, n_tokens);

    case 12u:
        return ds4_gpu_matmul_q4_K_tensor(
            out, model_map, model_size, weight_offset,
            in_dim, out_rows, x, n_tokens);

    case 30u:
        return ds4_gpu_glm53_matmul_bf16(
            out, model_map, model_size, weight_offset,
            in_dim, out_rows, x, n_tokens);

    default:
        return 0;
    }
}

extern "C" int ds4_gpu_qwen4_multi_gemv_tensor(
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t in_dim,
        uint32_t n_out,
        ds4_gpu_tensor *const *outs,
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint32_t *types,
        const uint32_t *out_rows) {
    if (!n_out || n_out > 4u ||
        !outs || !offsets || !types || !out_rows) {
        return 0;
    }

    for (uint32_t i = 0; i < n_out; ++i) {
        if (!ds4_gpu_qwen4_dense_mm_tensor(
                outs[i], x,
                model_map, model_size,
                offsets[i], types[i],
                n_tokens, in_dim, out_rows[i])) {
            return 0;
        }
    }

    return 1;
}

extern "C" int ds4_gpu_qwen4_batch_mm_q8_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *x,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_tokens,
        uint32_t in_dim,
        uint32_t out_rows) {
    return ds4_gpu_matmul_q8_0_tensor(
        out, model_map, model_size, weight_offset,
        in_dim, out_rows, x, n_tokens);
}


/* -------------------------------------------------------------------------
 * Qwen3.8 hyper-connections -- ROCm correctness path.
 * ------------------------------------------------------------------------- */

__device__ static inline float qwen4_rocm_sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

__device__ static inline float qwen4_rocm_silu(float x) {
    return x * qwen4_rocm_sigmoid(x);
}

static inline uint64_t qwen4_rocm_row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
    case 0u: return n * 4u;
    case 1u: return n * 2u;
    case 8u: return n % 32u ? 0u : n / 32u * 34u;
    default: return 0u;
    }
}

__device__ static inline float qwen4_rocm_weight_at(
        const char *row, uint32_t type, uint64_t i) {
    if (type == 0u) {
        return ((const float *)row)[i];
    }

    if (type == 1u) {
        return __half2float(((const __half *)row)[i]);
    }

    if (type == 8u) {
        const unsigned char *b =
            (const unsigned char *)row + (i >> 5u) * 34u;
        const float d = __half2float(*(const __half *)b);
        const int8_t q = ((const int8_t *)(b + 2u))[i & 31u];
        return d * (float)q;
    }

    return 0.0f;
}

__global__ static void qwen4_rocm_hc_norm_kernel(
        float *xn,
        float *inj,
        const float *R,
        const float *gamma,
        const char *wi,
        uint32_t type,
        uint32_t E,
        uint32_t hc,
        uint32_t ni,
        float eps) {
    const uint32_t stream = blockIdx.x / 8u;
    const uint32_t chunk  = blockIdx.x & 7u;
    const uint32_t tok    = blockIdx.y;
    const uint32_t tid    = threadIdx.x;
    const uint32_t dim    = E * hc;

    const uint64_t base =
        ((uint64_t)tok * hc + stream) * E;

    __shared__ float red[128];

    float ss = 0.0f;
    for (uint32_t i = tid; i < E; i += blockDim.x) {
        const float v = R[base + i];
        ss += v * v;
    }

    red[tid] = ss;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1u;
         stride != 0u;
         stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }

    const float inv = rsqrtf(red[0] / (float)E + eps);

    const uint32_t per = (E + 7u) / 8u;
    const uint32_t begin = chunk * per;
    const uint32_t end = min(E, begin + per);

    float acc[4] = {0, 0, 0, 0};

    for (uint32_t i = begin + tid; i < end; i += blockDim.x) {
        const float v =
            R[base + i] * inv * gamma[(uint64_t)stream * E + i];

        xn[base + i] = v;

        for (uint32_t j = 0; j < ni; ++j) {
            acc[j] += qwen4_rocm_weight_at(
                wi, type,
                (uint64_t)j * dim + (uint64_t)stream * E + i) * v;
        }
    }

    for (uint32_t j = 0; j < ni; ++j) {
        red[tid] = acc[j];
        __syncthreads();

        for (uint32_t stride = blockDim.x >> 1u;
             stride != 0u;
             stride >>= 1u) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }

        if (tid == 0u) {
            inj[
                ((uint64_t)tok * hc * 8u +
                 (uint64_t)stream * 8u +
                 chunk) * ni + j
            ] = red[0];
        }

        __syncthreads();
    }
}

__global__ static void qwen4_rocm_hc_gate_mix_kernel(
        float *out,
        const float *xn,
        const float *lo,
        const char *up,
        uint64_t row_bytes,
        uint32_t type,
        uint32_t E,
        uint32_t hc,
        uint32_t rank) {
    const uint32_t d =
        blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tok = blockIdx.y;

    if (d >= E) return;

    float result = 0.0f;

    for (uint32_t stream = 0; stream < hc; ++stream) {
        const char *row =
            up + ((uint64_t)stream * E + d) * row_bytes;

        float acc = 0.0f;

        for (uint32_t r = 0; r < rank; ++r) {
            const float a =
                qwen4_rocm_silu(
                    lo[(uint64_t)tok * rank + r] / (float)hc);

            acc += qwen4_rocm_weight_at(row, type, r) * a;
        }

        result += qwen4_rocm_sigmoid(acc) *
            xn[((uint64_t)tok * hc + stream) * E + d];
    }

    out[(uint64_t)tok * E + d] = result / (float)hc;
}

__global__ static void qwen4_rocm_hc_combine_kernel(
        float *R,
        const float *out,
        const float *inj,
        uint32_t E,
        uint32_t hc) {
    const uint32_t d =
        blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tok = blockIdx.y;

    if (d >= E) return;

    const float ov = out[(uint64_t)tok * E + d];
    const float *base =
        inj + (uint64_t)tok * hc * hc * 8u;

    for (uint32_t stream = 0; stream < hc; ++stream) {
        float s = 0.0f;

        for (uint32_t i = 0; i < hc * 8u; ++i) {
            s += base[(uint64_t)i * hc + stream];
        }

        const float w =
            2.0f * qwen4_rocm_sigmoid(s / (float)hc);

        R[((uint64_t)tok * hc + stream) * E + d] += w * ov;
    }
}

__global__ static void qwen4_rocm_hc_lo_kernel(
        float *dst,
        const float *src,
        uint64_t n,
        uint32_t hc) {
    const uint64_t i =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        dst[i] = qwen4_rocm_silu(src[i] / (float)hc);
    }
}

__global__ static void qwen4_rocm_hc_rows_kernel(
        float *dst,
        const float *up,
        const float *xn,
        uint32_t E,
        uint32_t hc) {
    const uint32_t d =
        blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tok = blockIdx.y;

    if (d >= E) return;

    float v = 0.0f;

    for (uint32_t stream = 0; stream < hc; ++stream) {
        const uint64_t i =
            ((uint64_t)tok * hc + stream) * E + d;

        v += qwen4_rocm_sigmoid(up[i]) * xn[i];
    }

    dst[(uint64_t)tok * E + d] = v / (float)hc;
}

extern "C" int ds4_gpu_qwen4_hc_norm_tensor(
        ds4_gpu_tensor *xn,
        ds4_gpu_tensor *inj,
        const ds4_gpu_tensor *R,
        const void *model_map,
        uint64_t model_size,
        uint64_t gamma_offset,
        uint64_t inject_offset,
        uint32_t weight_type,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_inject,
        float eps) {
    if (!xn || !R || !model_map ||
        !n_tokens || !n_embd || !n_hc ||
        n_hc > 8u || n_inject > 4u ||
        (weight_type != 0u &&
         weight_type != 1u &&
         weight_type != 8u)) {
        return 0;
    }

    const uint64_t n =
        (uint64_t)n_tokens * n_embd * n_hc;

    if (xn->bytes < n * sizeof(float) ||
        R->bytes < n * sizeof(float)) {
        return 0;
    }

    if (n_inject &&
        (!inj ||
         inj->bytes <
             (uint64_t)n_tokens * n_hc * 8u *
             n_inject * sizeof(float))) {
        return 0;
    }

    const uint64_t gamma_bytes =
        (uint64_t)n_embd * n_hc * sizeof(float);

    const uint64_t dim =
        (uint64_t)n_embd * n_hc;

    const uint64_t rb =
        qwen4_rocm_row_bytes(weight_type, dim);

    if (!rb) return 0;

    const float *gamma =
        (const float *)cuda_model_range_ptr(
            model_map, gamma_offset, gamma_bytes,
            "Qwen HC gamma");

    const char *wi = gamma
        ? (n_inject
            ? cuda_model_range_ptr(
                model_map, inject_offset,
                rb * n_inject,
                "Qwen HC injection")
            : (const char *)gamma)
        : NULL;

    if (!gamma || !wi) return 0;

    qwen4_rocm_hc_norm_kernel<<<
        dim3(n_hc * 8u, n_tokens),
        128,
        0,
        0
    >>>(
        (float *)xn->ptr,
        n_inject ? (float *)inj->ptr : NULL,
        (const float *)R->ptr,
        gamma,
        wi,
        weight_type,
        n_embd,
        n_hc,
        n_inject,
        eps);

    return cuda_ok(
        cudaGetLastError(),
        "Qwen ROCm HC norm");
}

extern "C" int ds4_gpu_qwen4_hc_gate_mix_tensor(
        ds4_gpu_tensor *mixed,
        const ds4_gpu_tensor *xn,
        const ds4_gpu_tensor *lo,
        const void *model_map,
        uint64_t model_size,
        uint64_t up_offset,
        uint32_t weight_type,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rank) {
    if (!mixed || !xn || !lo || !model_map ||
        !n_tokens || !n_embd || !n_hc ||
        n_hc > 4u || !n_rank ||
        (weight_type != 0u &&
         weight_type != 1u &&
         weight_type != 8u)) {
        return 0;
    }

    if (mixed->bytes <
            (uint64_t)n_tokens * n_embd * sizeof(float) ||
        xn->bytes <
            (uint64_t)n_tokens * n_embd * n_hc * sizeof(float) ||
        lo->bytes <
            (uint64_t)n_tokens * n_rank * sizeof(float)) {
        return 0;
    }

    const uint64_t rb =
        qwen4_rocm_row_bytes(weight_type, n_rank);

    if (!rb) return 0;

    const uint64_t weight_bytes =
        rb * n_embd * n_hc;

    const char *up =
        cuda_model_range_ptr(
            model_map, up_offset, weight_bytes,
            "Qwen HC up");

    if (!up) return 0;

    qwen4_rocm_hc_gate_mix_kernel<<<
        dim3((n_embd + 255u) / 256u, n_tokens),
        256,
        0,
        0
    >>>(
        (float *)mixed->ptr,
        (const float *)xn->ptr,
        (const float *)lo->ptr,
        up,
        rb,
        weight_type,
        n_embd,
        n_hc,
        n_rank);

    return cuda_ok(
        cudaGetLastError(),
        "Qwen ROCm HC gate mix");
}

extern "C" int ds4_gpu_qwen4_hc_combine_tensor(
        ds4_gpu_tensor *R,
        const ds4_gpu_tensor *out,
        const ds4_gpu_tensor *inj,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    if (!R || !out || !inj ||
        !n_tokens || !n_embd || !n_hc ||
        n_hc > 4u) {
        return 0;
    }

    if (R->bytes <
            (uint64_t)n_tokens * n_hc *
            n_embd * sizeof(float) ||
        out->bytes <
            (uint64_t)n_tokens *
            n_embd * sizeof(float) ||
        inj->bytes <
            (uint64_t)n_tokens * n_hc *
            n_hc * 8u * sizeof(float)) {
        return 0;
    }

    qwen4_rocm_hc_combine_kernel<<<
        dim3((n_embd + 255u) / 256u, n_tokens),
        256,
        0,
        0
    >>>(
        (float *)R->ptr,
        (const float *)out->ptr,
        (const float *)inj->ptr,
        n_embd,
        n_hc);

    return cuda_ok(
        cudaGetLastError(),
        "Qwen ROCm HC combine");
}

extern "C" int ds4_gpu_qwen4_hc_lo_act_tensor(
        ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src,
        uint32_t n_tokens,
        uint32_t n_hc,
        uint32_t n_rank) {
    const uint64_t n =
        (uint64_t)n_tokens * n_rank;

    if (!dst || !src || !n || !n_hc ||
        dst->bytes < n * sizeof(float) ||
        src->bytes < n * sizeof(float)) {
        return 0;
    }

    qwen4_rocm_hc_lo_kernel<<<
        (unsigned)((n + 255u) / 256u),
        256,
        0,
        0
    >>>(
        (float *)dst->ptr,
        (const float *)src->ptr,
        n,
        n_hc);

    return cuda_ok(
        cudaGetLastError(),
        "Qwen ROCm HC lo");
}

extern "C" int ds4_gpu_qwen4_hc_mix_rows_tensor(
        ds4_gpu_tensor *mixed,
        const ds4_gpu_tensor *up,
        const ds4_gpu_tensor *xn,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    if (!mixed || !up || !xn ||
        !n_tokens || !n_embd || !n_hc ||
        n_hc > 4u) {
        return 0;
    }

    const uint64_t hc_elems =
        (uint64_t)n_tokens * n_hc * n_embd;

    if (mixed->bytes <
            (uint64_t)n_tokens * n_embd * sizeof(float) ||
        up->bytes < hc_elems * sizeof(float) ||
        xn->bytes < hc_elems * sizeof(float)) {
        return 0;
    }

    qwen4_rocm_hc_rows_kernel<<<
        dim3((n_embd + 255u) / 256u, n_tokens),
        256,
        0,
        0
    >>>(
        (float *)mixed->ptr,
        (const float *)up->ptr,
        (const float *)xn->ptr,
        n_embd,
        n_hc);

    return cuda_ok(
        cudaGetLastError(),
        "Qwen ROCm HC rows");
}

extern "C" int ds4_gpu_qwen4_hc_combine_norm_tensor(
        ds4_gpu_tensor *next_R,
        const ds4_gpu_tensor *blk,
        const ds4_gpu_tensor *old_inj,
        ds4_gpu_tensor *xn,
        ds4_gpu_tensor *inj_part,
        const ds4_gpu_tensor *R,
        const void *model_map,
        uint64_t model_size,
        uint64_t gamma_offset,
        uint64_t inject_offset,
        uint32_t weight_type,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_inject,
        float eps) {
    const uint64_t bytes =
        (uint64_t)n_tokens * n_embd *
        n_hc * sizeof(float);

    if (!next_R || !R ||
        next_R == R ||
        !old_inj || !inj_part ||
        old_inj == inj_part ||
        next_R->bytes < bytes ||
        R->bytes < bytes) {
        return 0;
    }

    if (!cuda_ok(
            cudaMemcpyAsync(
                next_R->ptr,
                R->ptr,
                bytes,
                cudaMemcpyDeviceToDevice,
                0),
            "Qwen ROCm HC copy")) {
        return 0;
    }

    return
        ds4_gpu_qwen4_hc_combine_tensor(
            next_R, blk, old_inj,
            n_tokens, n_embd, n_hc) &&
        ds4_gpu_qwen4_hc_norm_tensor(
            xn, inj_part, next_R,
            model_map, model_size,
            gamma_offset, inject_offset,
            weight_type, n_tokens,
            n_embd, n_hc,
            n_inject, eps);
}


/* Qwen3.8 GDN - ROCm correctness path. */

__device__ static inline float qwen4_rocm_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ static void qwen4_rocm_conv_kernel(
        float *x,
        float *history,
        const float *w,
        uint32_t T,
        uint32_t C,
        uint32_t K,
        bool activate,
        float *snap,
        uint32_t snap_t,
        float *snap2,
        uint32_t snap2_t) {
    const uint32_t c =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= C) return;

    float win[3] = {0.0f, 0.0f, 0.0f};
    float taps[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint32_t i = 0; i + 1u < K; ++i) {
        win[i] = history[(uint64_t)i * C + c];
    }

    for (uint32_t i = 0; i < K; ++i) {
        taps[i] = w[(uint64_t)c * K + i];
    }

    for (uint32_t t = 0; t < T; ++t) {
        const uint64_t pos = (uint64_t)t * C + c;
        const float raw = x[pos];

        float v = taps[K - 1u] * raw;

        for (uint32_t i = 0; i + 1u < K; ++i) {
            v += taps[i] * win[i];
        }

        for (uint32_t i = 0; i + 2u < K; ++i) {
            win[i] = win[i + 1u];
        }

        win[K - 2u] = raw;

        x[pos] = activate ? qwen4_rocm_silu(v) : v;

        if (snap && t == snap_t) {
            for (uint32_t i = 0; i + 1u < K; ++i) {
                snap[(uint64_t)i * C + c] = win[i];
            }
        }

        if (snap2 && t == snap2_t) {
            for (uint32_t i = 0; i + 1u < K; ++i) {
                snap2[(uint64_t)i * C + c] = win[i];
            }
        }
    }

    for (uint32_t i = 0; i + 1u < K; ++i) {
        history[(uint64_t)i * C + c] = win[i];
    }
}

__global__ static void qwen4_rocm_gdn_prep_kernel(
        float *qkv,
        float *a,
        float *b,
        const float *ssm_a,
        const float *dt_bias,
        uint32_t Hk,
        uint32_t Hv,
        uint32_t D) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;

    const uint32_t C = (2u * Hk + Hv) * D;

    float *q =
        qkv + (uint64_t)t * C + (uint64_t)h * D;

    float *k =
        qkv + (uint64_t)t * C +
        (uint64_t)Hk * D +
        (uint64_t)h * D;

    __shared__ float qred[128];
    __shared__ float kred[128];

    float qv = 0.0f;
    float kv = 0.0f;

    if (tid < D) {
        qv = q[tid];
        kv = k[tid];
    }

    qred[tid] = qv * qv;
    kred[tid] = kv * kv;
    __syncthreads();

    for (uint32_t stride = 64u;
         stride != 0u;
         stride >>= 1u) {
        if (tid < stride) {
            qred[tid] += qred[tid + stride];
            kred[tid] += kred[tid + stride];
        }
        __syncthreads();
    }

    if (tid < D) {
        const float qscale =
            rsqrtf(qred[0] + 1e-6f) *
            rsqrtf((float)D);

        const float kscale =
            rsqrtf(kred[0] + 1e-6f);

        q[tid] *= qscale;
        k[tid] *= kscale;
    }

    if (h == 0u) {
        for (uint32_t j = tid; j < Hv; j += blockDim.x) {
            const uint64_t p = (uint64_t)t * Hv + j;

            a[p] = expf(
                ssm_a[j] *
                qwen4_rocm_softplus(a[p] + dt_bias[j]));

            b[p] = qwen4_rocm_sigmoid(b[p]);
        }
    }
}

__global__ static void qwen4_rocm_gdn_scan_kernel(
        float *out,
        float *state,
        const float *qkv,
        const float *a,
        const float *b,
        uint32_t T,
        uint32_t Hk,
        uint32_t Hv,
        uint32_t D,
        float *snap,
        uint32_t snap_t,
        float *snap2,
        uint32_t snap2_t) {
    const uint32_t h = blockIdx.x;
    const uint32_t dv = threadIdx.x;

    if (h >= Hv || dv >= D) return;

    const uint32_t kh = h % Hk;
    const uint32_t C = (2u * Hk + Hv) * D;

    float *S =
        state +
        ((uint64_t)h * D + dv) * D;

    for (uint32_t t = 0; t < T; ++t) {
        const float *q =
            qkv + (uint64_t)t * C +
            (uint64_t)kh * D;

        const float *k =
            qkv + (uint64_t)t * C +
            (uint64_t)Hk * D +
            (uint64_t)kh * D;

        const float *v =
            qkv + (uint64_t)t * C +
            (uint64_t)2u * Hk * D +
            (uint64_t)h * D;

        const float decay =
            a[(uint64_t)t * Hv + h];

        const float beta =
            b[(uint64_t)t * Hv + h];

        float u = 0.0f;

        for (uint32_t dk = 0; dk < D; ++dk) {
            S[dk] *= decay;
            u += S[dk] * k[dk];
        }

        const float delta =
            (v[dv] - u) * beta;

        float acc = 0.0f;

        for (uint32_t dk = 0; dk < D; ++dk) {
            S[dk] += k[dk] * delta;
            acc += S[dk] * q[dk];
        }

        out[((uint64_t)t * Hv + h) * D + dv] = acc;

        if (snap && t == snap_t) {
            float *dst =
                snap +
                ((uint64_t)h * D + dv) * D;

            for (uint32_t dk = 0; dk < D; ++dk) {
                dst[dk] = S[dk];
            }
        }

        if (snap2 && t == snap2_t) {
            float *dst =
                snap2 +
                ((uint64_t)h * D + dv) * D;

            for (uint32_t dk = 0; dk < D; ++dk) {
                dst[dk] = S[dk];
            }
        }
    }
}

__global__ static void qwen4_rocm_gdn_out_kernel(
        float *out,
        const float *z,
        const float *w,
        uint32_t H,
        uint32_t D,
        float eps) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;

    const uint64_t base =
        ((uint64_t)t * H + h) * D;

    __shared__ float red[128];

    float v = 0.0f;

    if (tid < D) {
        v = out[base + tid];
    }

    red[tid] = v * v;
    __syncthreads();

    for (uint32_t stride = 64u;
         stride != 0u;
         stride >>= 1u) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }

    if (tid < D) {
        const float scale =
            rsqrtf(red[0] / (float)D + eps);

        out[base + tid] =
            v *
            scale *
            w[tid] *
            qwen4_rocm_sigmoid(z[base + tid]);
    }
}

extern "C" int ds4_gpu_qwen4_conv_stream_tensor(
        ds4_gpu_tensor *x,
        ds4_gpu_tensor *state,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_tokens,
        uint32_t n_channels,
        uint32_t conv_kernel,
        bool apply_silu) {
    if (!x || !state || !model_map ||
        !n_tokens || !n_channels ||
        conv_kernel < 2u || conv_kernel > 4u) {
        return 0;
    }

    const uint64_t x_bytes =
        (uint64_t)n_tokens *
        n_channels * sizeof(float);

    const uint64_t state_bytes =
        (uint64_t)(conv_kernel - 1u) *
        n_channels * sizeof(float);

    const uint64_t weight_bytes =
        (uint64_t)n_channels *
        conv_kernel * sizeof(float);

    if (x->bytes < x_bytes ||
        state->bytes < state_bytes) {
        return 0;
    }

    const float *w =
        (const float *)cuda_model_range_ptr(
            model_map,
            weight_offset,
            weight_bytes,
            "Qwen GDN conv");

    if (!w) return 0;

    qwen4_rocm_conv_kernel<<<
        (n_channels + 255u) / 256u,
        256
    >>>(
        (float *)x->ptr,
        (float *)state->ptr,
        w,
        n_tokens,
        n_channels,
        conv_kernel,
        apply_silu,
        NULL,
        UINT_MAX,
        NULL,
        UINT_MAX);

    return cuda_ok(
        cudaGetLastError(),
        "Qwen ROCm GDN conv");
}

extern "C" int ds4_gpu_qwen4_gdn_prep_tensor(
        ds4_gpu_tensor *qkv,
        ds4_gpu_tensor *a,
        ds4_gpu_tensor *b,
        const void *model_map,
        uint64_t model_size,
        uint64_t ssm_a_offset,
        uint64_t dt_bias_offset,
        uint32_t n_tokens,
        uint32_t n_k_head,
        uint32_t n_v_head,
        uint32_t head_dim) {
    if (!qkv || !a || !b || !model_map ||
        !n_tokens || !n_k_head || !n_v_head ||
        head_dim < 32u ||
        head_dim > 128u ||
        (head_dim % 32u) != 0u) {
        return 0;
    }

    const uint64_t qkv_elems =
        (uint64_t)n_tokens *
        (2u * n_k_head + n_v_head) *
        head_dim;

    const uint64_t ab_elems =
        (uint64_t)n_tokens * n_v_head;

    if (qkv->bytes < qkv_elems * sizeof(float) ||
        a->bytes < ab_elems * sizeof(float) ||
        b->bytes < ab_elems * sizeof(float)) {
        return 0;
    }

    const float *ssm_a =
        (const float *)cuda_model_range_ptr(
            model_map,
            ssm_a_offset,
            (uint64_t)n_v_head * sizeof(float),
            "Qwen GDN ssm_a");

    const float *dt_bias =
        (const float *)cuda_model_range_ptr(
            model_map,
            dt_bias_offset,
            (uint64_t)n_v_head * sizeof(float),
            "Qwen GDN dt_bias");

    if (!ssm_a || !dt_bias) return 0;

    qwen4_rocm_gdn_prep_kernel<<<
        dim3(n_k_head, n_tokens),
        128
    >>>(
        (float *)qkv->ptr,
        (float *)a->ptr,
        (float *)b->ptr,
        ssm_a,
        dt_bias,
        n_k_head,
        n_v_head,
        head_dim);

    return cuda_ok(
        cudaGetLastError(),
        "Qwen ROCm GDN prep");
}

extern "C" void ds4_gpu_qwen4_set_verify_rows_exact(bool on) {
    (void)on;
}

extern "C" int ds4_gpu_qwen4_gdn_scan_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *state,
        const ds4_gpu_tensor *qkv,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t n_tokens,
        uint32_t n_k_head,
        uint32_t n_v_head,
        uint32_t head_dim,
        ds4_gpu_tensor *snap_state,
        uint32_t snap_tok,
        ds4_gpu_tensor *snap2_state,
        uint32_t snap2_tok) {
    if (!out || !state || !qkv || !a || !b ||
        !n_tokens || !n_k_head || !n_v_head ||
        (n_v_head % n_k_head) != 0u ||
        head_dim < 32u ||
        head_dim > 128u ||
        (head_dim % 32u) != 0u) {
        return 0;
    }

    const uint64_t state_elems =
        (uint64_t)n_v_head *
        head_dim * head_dim;

    const uint64_t out_elems =
        (uint64_t)n_tokens *
        n_v_head * head_dim;

    const uint64_t qkv_elems =
        (uint64_t)n_tokens *
        (2u * n_k_head + n_v_head) *
        head_dim;

    const uint64_t ab_elems =
        (uint64_t)n_tokens * n_v_head;

    if (state->bytes < state_elems * sizeof(float) ||
        out->bytes < out_elems * sizeof(float) ||
        qkv->bytes < qkv_elems * sizeof(float) ||
        a->bytes < ab_elems * sizeof(float) ||
        b->bytes < ab_elems * sizeof(float) ||
        (snap_state &&
         snap_state->bytes <
             state_elems * sizeof(float)) ||
        (snap2_state &&
         snap2_state->bytes <
             state_elems * sizeof(float))) {
        return 0;
    }

    qwen4_rocm_gdn_scan_kernel<<<
        n_v_head,
        128
    >>>(
        (float *)out->ptr,
        (float *)state->ptr,
        (const float *)qkv->ptr,
        (const float *)a->ptr,
        (const float *)b->ptr,
        n_tokens,
        n_k_head,
        n_v_head,
        head_dim,
        snap_state
            ? (float *)snap_state->ptr
            : NULL,
        snap_tok,
        snap2_state
            ? (float *)snap2_state->ptr
            : NULL,
        snap2_tok);

    return cuda_ok(
        cudaGetLastError(),
        "Qwen ROCm GDN scan");
}

extern "C" int ds4_gpu_qwen4_gdn_out_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *z,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        float eps) {
    if (!out || !z || !model_map ||
        !n_tokens || !n_head ||
        head_dim < 32u ||
        head_dim > 128u ||
        (head_dim % 32u) != 0u) {
        return 0;
    }

    const uint64_t elems =
        (uint64_t)n_tokens *
        n_head * head_dim;

    if (out->bytes < elems * sizeof(float) ||
        z->bytes < elems * sizeof(float)) {
        return 0;
    }

    const float *w =
        (const float *)cuda_model_range_ptr(
            model_map,
            weight_offset,
            (uint64_t)head_dim * sizeof(float),
            "Qwen GDN output norm");

    if (!w) return 0;

    qwen4_rocm_gdn_out_kernel<<<
        dim3(n_head, n_tokens),
        128
    >>>(
        (float *)out->ptr,
        (const float *)z->ptr,
        w,
        n_head,
        head_dim,
        eps);

    return cuda_ok(
        cudaGetLastError(),
        "Qwen ROCm GDN out");
}

extern "C" int ds4_gpu_qwen4_gdn_front_tensor(
        ds4_gpu_tensor *qkv,
        ds4_gpu_tensor *state,
        const ds4_gpu_tensor *mixed,
        ds4_gpu_tensor *ga,
        ds4_gpu_tensor *gb,
        const void *model_map,
        uint64_t model_size,
        uint64_t conv_offset,
        uint64_t alpha_offset,
        uint64_t beta_offset,
        uint64_t ssm_a_offset,
        uint64_t dt_bias_offset,
        uint32_t weight_type,
        uint32_t n_tokens,
        uint32_t n_k_head,
        uint32_t n_v_head,
        uint32_t head_dim,
        uint32_t conv_kernel,
        uint32_t in_dim,
        ds4_gpu_tensor *snap_state,
        uint32_t snap_tok,
        ds4_gpu_tensor *snap2_state,
        uint32_t snap2_tok) {
    const uint64_t channels =
        ((uint64_t)2u * n_k_head + n_v_head) *
        head_dim;

    if (!qkv || !state || !mixed ||
        !ga || !gb || !model_map ||
        !n_tokens || !n_k_head ||
        !n_v_head || !head_dim ||
        !in_dim ||
        channels > UINT_MAX ||
        conv_kernel < 2u ||
        conv_kernel > 4u) {
        return 0;
    }

    const uint64_t history_bytes =
        (uint64_t)(conv_kernel - 1u) *
        channels * sizeof(float);

    if (qkv->bytes <
            (uint64_t)n_tokens *
            channels * sizeof(float) ||
        state->bytes < history_bytes ||
        (snap_state &&
         (snap_tok >= n_tokens ||
          snap_state->bytes < history_bytes)) ||
        (snap2_state &&
         (snap2_tok >= n_tokens ||
          snap2_state->bytes < history_bytes))) {
        return 0;
    }

    if (!ds4_gpu_qwen4_dense_mm_tensor(
            ga,
            mixed,
            model_map,
            model_size,
            alpha_offset,
            weight_type,
            n_tokens,
            in_dim,
            n_v_head)) {
        return 0;
    }

    if (!ds4_gpu_qwen4_dense_mm_tensor(
            gb,
            mixed,
            model_map,
            model_size,
            beta_offset,
            weight_type,
            n_tokens,
            in_dim,
            n_v_head)) {
        return 0;
    }

    const uint64_t conv_bytes =
        channels *
        conv_kernel *
        sizeof(float);

    const float *w =
        (const float *)cuda_model_range_ptr(
            model_map,
            conv_offset,
            conv_bytes,
            "Qwen GDN fused conv");

    if (!w) return 0;

    qwen4_rocm_conv_kernel<<<
        ((uint32_t)channels + 255u) / 256u,
        256
    >>>(
        (float *)qkv->ptr,
        (float *)state->ptr,
        w,
        n_tokens,
        (uint32_t)channels,
        conv_kernel,
        true,
        snap_state
            ? (float *)snap_state->ptr
            : NULL,
        snap_tok,
        snap2_state
            ? (float *)snap2_state->ptr
            : NULL,
        snap2_tok);

    if (!cuda_ok(
            cudaGetLastError(),
            "Qwen ROCm GDN fused conv")) {
        return 0;
    }

    return ds4_gpu_qwen4_gdn_prep_tensor(
        qkv,
        ga,
        gb,
        model_map,
        model_size,
        ssm_a_offset,
        dt_bias_offset,
        n_tokens,
        n_k_head,
        n_v_head,
        head_dim);
}


/* Qwen3.8 indexer - ROCm correctness path. */

__global__ static void qwen4_rocm_idx_score_kernel(
        float *out,
        const float *q,
        const __half *key,
        uint32_t N,
        uint32_t H,
        uint32_t D,
        uint32_t pos0,
        uint32_t ratio) {
    const uint32_t b =
        blockIdx.x * 4u + threadIdx.x / 32u;

    const uint32_t t = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;

    if (b >= N) return;

    float score = 0.0f;

    if (b >= (pos0 + t + 1u) / ratio) {
        score = -3.0e38f;
    } else {
        for (uint32_t h = 0; h < H; ++h) {
            float acc = 0.0f;

            for (uint32_t i = lane; i < D; i += 32u) {
                acc +=
                    q[((uint64_t)t * H + h) * D + i] *
                    __half2float(
                        key[(uint64_t)b * D + i]);
            }

            acc = warp_sum_f32(acc);

            if (lane == 0u) {
                score += fmaxf(acc, 0.0f);
            }
        }
    }

    if (lane == 0u) {
        out[(uint64_t)t * N + b] = score;
    }
}

__global__ static void qwen4_rocm_idx_tile_max_kernel(
        uint32_t *out,
        const float *scores,
        uint32_t N,
        uint32_t n_tiles) {
    const uint32_t tile =
        blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t t = blockIdx.y;

    if (tile >= n_tiles) return;

    uint32_t best = 0u;

    const uint32_t begin = tile * 8u;
    const uint32_t end =
        min(N, begin + 8u);

    for (uint32_t i = begin; i < end; ++i) {
        const float v =
            fmaxf(scores[(uint64_t)t * N + i], 0.0f);

        best = max(best, __float_as_uint(v));
    }

    out[(uint64_t)t * n_tiles + tile] = best;
}

__global__ static void qwen4_rocm_idx_expand_kernel(
        int32_t *out,
        uint32_t *count,
        const int32_t *blocks,
        uint32_t K,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t stride) {
    const uint32_t t = blockIdx.x;
    const uint32_t pos = pos0 + t;

    const uint32_t tail =
        ((pos + 1u) / ratio) * ratio;

    for (uint32_t i = threadIdx.x;
         i < K * ratio;
         i += blockDim.x) {
        out[(uint64_t)t * stride + i] =
            blocks[(uint64_t)t * K + i / ratio] *
                (int32_t)ratio +
            (int32_t)(i % ratio);
    }

    for (uint32_t i = tail + threadIdx.x;
         i <= pos;
         i += blockDim.x) {
        out[
            (uint64_t)t * stride +
            K * ratio +
            i - tail
        ] = (int32_t)i;
    }

    if (threadIdx.x == 0u) {
        count[t] =
            K * ratio +
            pos + 1u - tail;
    }
}

extern "C" int ds4_gpu_qwen4_idx_score_tensor(
        ds4_gpu_tensor *score,
        ds4_gpu_tensor *tile_max,
        const ds4_gpu_tensor *iq,
        const ds4_gpu_tensor *block_key,
        uint32_t n_tokens,
        uint32_t n_blocks,
        uint32_t n_idx_head,
        uint32_t idx_dim,
        uint32_t pos0,
        uint32_t ratio) {
    if (!score || !iq || !block_key ||
        !n_tokens || !n_blocks ||
        !n_idx_head || !idx_dim || !ratio) {
        return 0;
    }

    const uint64_t score_bytes =
        (uint64_t)n_tokens *
        n_blocks *
        sizeof(float);

    const uint64_t q_bytes =
        (uint64_t)n_tokens *
        n_idx_head *
        idx_dim *
        sizeof(float);

    const uint64_t key_bytes =
        (uint64_t)n_blocks *
        idx_dim *
        sizeof(__half);

    if (score->bytes < score_bytes ||
        iq->bytes < q_bytes ||
        block_key->bytes < key_bytes) {
        return 0;
    }

    qwen4_rocm_idx_score_kernel<<<
        dim3((n_blocks + 3u) / 4u, n_tokens),
        128
    >>>(
        (float *)score->ptr,
        (const float *)iq->ptr,
        (const __half *)block_key->ptr,
        n_blocks,
        n_idx_head,
        idx_dim,
        pos0,
        ratio);

    if (!cuda_ok(
            cudaGetLastError(),
            "Qwen ROCm index score")) {
        return 0;
    }

    if (tile_max) {
        const uint32_t n_tiles =
            (n_blocks + 7u) / 8u;

        if (tile_max->bytes <
            (uint64_t)n_tokens *
            n_tiles *
            sizeof(uint32_t)) {
            return 0;
        }

        qwen4_rocm_idx_tile_max_kernel<<<
            dim3((n_tiles + 255u) / 256u, n_tokens),
            256
        >>>(
            (uint32_t *)tile_max->ptr,
            (const float *)score->ptr,
            n_blocks,
            n_tiles);

        if (!cuda_ok(
                cudaGetLastError(),
                "Qwen ROCm index tile max")) {
            return 0;
        }
    }

    return 1;
}

extern "C" int ds4_gpu_qwen4_idx_select_tensor(
        ds4_gpu_tensor *sel,
        const ds4_gpu_tensor *score,
        const ds4_gpu_tensor *tile_max,
        uint32_t n_blocks,
        uint32_t n_tokens,
        uint32_t top_k) {
    (void)tile_max;

    return ds4_gpu_indexer_topk_tensor(
        sel,
        score,
        n_blocks,
        n_tokens,
        top_k);
}

extern "C" int ds4_gpu_qwen4_idx_expand_tensor(
        ds4_gpu_tensor *sel_tokens,
        ds4_gpu_tensor *n_sel,
        const ds4_gpu_tensor *sel_blocks,
        uint32_t n_tokens,
        uint32_t n_sel_blocks,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t sel_stride) {
    if (!sel_tokens || !n_sel || !sel_blocks ||
        !n_tokens || !n_sel_blocks ||
        !ratio ||
        (uint64_t)n_sel_blocks * ratio +
            ratio - 1u > sel_stride) {
        return 0;
    }

    if (sel_tokens->bytes <
            (uint64_t)n_tokens *
            sel_stride *
            sizeof(int32_t) ||
        n_sel->bytes <
            (uint64_t)n_tokens *
            sizeof(uint32_t) ||
        sel_blocks->bytes <
            (uint64_t)n_tokens *
            n_sel_blocks *
            sizeof(int32_t)) {
        return 0;
    }

    qwen4_rocm_idx_expand_kernel<<<
        n_tokens,
        256
    >>>(
        (int32_t *)sel_tokens->ptr,
        (uint32_t *)n_sel->ptr,
        (const int32_t *)sel_blocks->ptr,
        n_sel_blocks,
        ratio,
        pos0,
        sel_stride);

    return cuda_ok(
        cudaGetLastError(),
        "Qwen ROCm index expand");
}


/* Qwen3.8 block-key - ROCm correctness path. */

typedef struct {
    float freq[32];
    float scale;
    uint32_t custom;
} qwen4_rocm_rope_args;

static float qwen4_rocm_rope_freq[32] = {};
static float qwen4_rocm_rope_scale = 1.0f;
static bool qwen4_rocm_rope_custom = false;

extern "C" void ds4_gpu_qwen4_set_rope(
        const float *freq,
        uint32_t n_pairs,
        float mscale) {
    qwen4_rocm_rope_custom = freq != NULL;
    qwen4_rocm_rope_scale = freq ? mscale : 1.0f;

    for (uint32_t i = 0; i < 32u; ++i) {
        qwen4_rocm_rope_freq[i] =
            freq && i < n_pairs ? freq[i] : 0.0f;
    }
}

__global__ static void qwen4_rocm_idx_block_key_kernel(
        __half *out,
        const float *ik,
        const uint32_t *pos3,
        const float *gamma,
        uint32_t block0,
        uint32_t ratio,
        uint32_t D,
        uint32_t n_rot,
        float rope_base,
        float eps,
        qwen4_rocm_rope_args rp) {
    const uint32_t b = block0 + blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t npt = D / 32u;

    __shared__ float row[128];

    float v[4] = {0, 0, 0, 0};
    float ss = 0.0f;

    for (uint32_t i = 0; i < npt; ++i) {
        const uint32_t d = lane * npt + i;

        float acc = 0.0f;

        for (uint32_t t = 0; t < ratio; ++t) {
            acc += ik[
                ((uint64_t)b * ratio + t) * D + d
            ];
        }

        v[i] = acc / (float)ratio;
        ss += v[i] * v[i];
    }

    /* warp_sum_f32 leaves the complete reduction in lane 0 only. Every
     * lane normalizes its own four dimensions, so broadcast the sum. */
    ss = qwen4_rocm_wave32_sum_xor(ss);

    const float inv =
        rsqrtf(ss / (float)D + eps);

    for (uint32_t i = 0; i < npt; ++i) {
        const uint32_t d = lane * npt + i;

        row[d] =
            v[i] *
            inv *
            gamma[d];
    }

    __syncthreads();

    if (lane < n_rot / 2u) {
        const uint32_t *pos =
            pos3 + (uint64_t)b * ratio * 4u;

        const float freq =
            rp.custom
                ? rp.freq[lane]
                : powf(
                    rope_base,
                    -2.0f * (float)lane /
                    (float)n_rot);

        const float theta =
            (float)pos[lane % 3u] * freq;

        const float c =
            cosf(theta) * rp.scale;

        const float sn =
            sinf(theta) * rp.scale;

        const float a = row[lane];
        const float bb =
            row[lane + n_rot / 2u];

        row[lane] =
            a * c - bb * sn;

        row[lane + n_rot / 2u] =
            a * sn + bb * c;
    }

    __syncthreads();

    for (uint32_t d = lane;
         d < D;
         d += 32u) {
        out[(uint64_t)b * D + d] =
            __float2half_rn(row[d]);
    }
}

extern "C" int ds4_gpu_qwen4_idx_block_key_tensor(
        ds4_gpu_tensor *block_key,
        const ds4_gpu_tensor *ik_cache,
        const ds4_gpu_tensor *pos3,
        const void *model_map,
        uint64_t model_size,
        uint64_t g_ik_offset,
        uint32_t block0,
        uint32_t n_blocks,
        uint32_t ratio,
        uint32_t idx_dim,
        uint32_t n_rot,
        float rope_base,
        float eps) {
    if (!block_key || !ik_cache ||
        !pos3 || !model_map ||
        !n_blocks || !ratio ||
        idx_dim < 32u ||
        idx_dim > 128u ||
        (idx_dim % 32u) != 0u ||
        n_rot > idx_dim ||
        n_rot > 64u ||
        (n_rot & 1u)) {
        return 0;
    }

    const uint64_t rows =
        (uint64_t)block0 + n_blocks;

    const uint64_t out_bytes =
        rows *
        idx_dim *
        sizeof(__half);

    const uint64_t ik_bytes =
        rows *
        ratio *
        idx_dim *
        sizeof(float);

    const uint64_t pos_bytes =
        rows *
        ratio *
        4u *
        sizeof(uint32_t);

    if (block_key->bytes < out_bytes ||
        ik_cache->bytes < ik_bytes ||
        pos3->bytes < pos_bytes) {
        return 0;
    }

    const float *gamma =
        (const float *)cuda_model_range_ptr(
            model_map,
            g_ik_offset,
            (uint64_t)idx_dim * sizeof(float),
            "Qwen index block-key gamma");

    if (!gamma) return 0;

    qwen4_rocm_rope_args rp = {};
    rp.scale = qwen4_rocm_rope_scale;
    rp.custom =
        qwen4_rocm_rope_custom ? 1u : 0u;

    for (uint32_t i = 0; i < 32u; ++i) {
        rp.freq[i] =
            qwen4_rocm_rope_freq[i];
    }

    qwen4_rocm_idx_block_key_kernel<<<
        n_blocks,
        32
    >>>(
        (__half *)block_key->ptr,
        (const float *)ik_cache->ptr,
        (const uint32_t *)pos3->ptr,
        gamma,
        block0,
        ratio,
        idx_dim,
        n_rot,
        rope_base,
        eps,
        rp);

    return cuda_ok(
        cudaGetLastError(),
        "Qwen ROCm index block-key");
}
