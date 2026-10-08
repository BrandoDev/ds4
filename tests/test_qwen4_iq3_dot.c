/* SIMD IQ3 against independent double accumulation, with activation buffers
 * ending at an inaccessible page so padded weight tails cannot be read. */
#define DS4_TEST_HOOKS
#define DS4_NO_GPU
#include "../ds4.c"
#include <assert.h>

static uint32_t state = 1234567;
static uint32_t random_word(void) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

static double reference(const uint8_t *row, const float *x, int n, double *magnitude) {
    double sum = 0; *magnitude = 0;
    for (int k = 0; k < n; k++) {
        const uint8_t *p = row + (k / 256) * 98;
        const int group = (k % 256) / 32, pair = (k % 32) / 8, lane = k % 8;
        uint16_t dh; uint32_t aux;
        memcpy(&dh, p, 2); memcpy(&aux, p + 66 + group * 4, 4);
        const uint32_t grid = iq3xxs_grid[p[2 + group * 8 + pair * 2 + lane / 4]];
        int v = (grid >> ((lane % 4) * 8)) & 255u;
        if (ksigns_iq2xs[(aux >> (pair * 7)) & 127u] & (1u << lane)) v = -v;
        const float w = f16_to_f32(dh) * (0.5f + (float)(aux >> 28)) * 0.5f * v;
        const double product = (double)w * x[k];
        sum += product; *magnitude += fabs(product);
    }
    return sum;
}

int main(void) {
#if defined(__AVX2__)
    const int widths[][2] = {{256,256},{768,640},{2560,2560},{768,641}};
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    float worst = 0;
    for (unsigned shape = 0; shape < 4; shape++) {
        const int physical = widths[shape][0], logical = widths[shape][1];
        const size_t span = ((logical * sizeof(float) + page - 1) / page) * page;
        char *map = mmap(NULL, span + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(map != MAP_FAILED && mprotect(map + span, page, PROT_NONE) == 0);
        float *x = (float *)(map + span - logical * sizeof(float));
        uint8_t *row = xmalloc((physical / 256) * 98);
        for (int sample = 0; sample < 256; sample++) {
            for (int j = 0; j < (physical / 256) * 98; j++) row[j] = (uint8_t)random_word();
            for (int b = 0; b < physical / 256; b++) {
                const uint16_t d = 0x1800u + (random_word() & 1023u);
                memcpy(row + b * 98, &d, 2);
            }
            for (int k = 0; k < logical; k++) x[k] = ((int)(random_word() & 65535u) - 32768) / 32768.0f;
            block_q8_K q[10];
            qwen4_quantize_q8(logical, x, q);
            float dequant[2560];
            for (int k = 0; k < ((logical + 255) / 256) * 256; k++) {
                assert(q[k / 256].qs[k % 256] >= -127);
                dequant[k] = q[k / 256].d * q[k / 256].qs[k % 256];
                if (k >= logical) assert(dequant[k] == 0);
            }
            const block_q8_K *quantized[4] = {q,q,q,q};
            float quant_outputs[4];
            double quant_magnitude;
            const double quant_expected = reference(row, dequant, logical, &quant_magnitude);
            for (uint32_t count = 1; count <= 4; count++) {
                qwen4_iq3_q8_many(logical, row, quantized, quant_outputs, count);
                for (uint32_t t = 0; t < count; t++)
                    assert(fabs(quant_outputs[t] - quant_expected) <= 2e-6 * quant_magnitude + 1e-6);
            }
            /* Independent grouped inputs catch slot/accumulator mixups that
             * identical inputs cannot reveal. Include byte extrema. */
            block_q8_K distinct[4][10];
            const block_q8_K *distinct_ptrs[4];
            for (int t = 0; t < 4; t++) {
                distinct_ptrs[t] = distinct[t];
                memcpy(distinct[t], q, ((logical + 255) / 256) * sizeof(block_q8_K));
                for (int k = 0; k < logical; k++)
                    distinct[t][k / 256].qs[k % 256] = (int8_t)((random_word() % 255) - 127);
                for (int b = 0; b < (logical + 255) / 256; b++) distinct[t][b].d *= t + 0.5f;
            }
            for (uint32_t count = 1; count <= 4; count++) {
                qwen4_iq3_q8_many(logical, row, distinct_ptrs, quant_outputs, count);
                for (uint32_t t = 0; t < count; t++) {
                    for (int k = 0; k < logical; k++) dequant[k] = distinct[t][k / 256].d * distinct[t][k / 256].qs[k % 256];
                    const double expected_distinct = reference(row, dequant, logical, &quant_magnitude);
                    assert(fabs(quant_outputs[t] - expected_distinct) <= 2e-6 * quant_magnitude + 1e-6);
                    float single;
                    qwen4_iq3_q8_many(logical, row, distinct_ptrs + t, &single, 1);
                    assert(single == quant_outputs[t]);
                }
            }
            double magnitude;
            const double expected = reference(row, x, logical, &magnitude);
            const float actual = qwen4_iq3_dot_avx2(physical, logical, row, x);
            const float scalar = ds4_vec_dot_iq3_xxs_f32(physical, logical, row, x);
            assert(isfinite(actual) && fabs(actual - expected) <= 2e-6 * magnitude + 1e-6);
            assert(fabs(scalar - expected) <= 2e-6 * magnitude + 1e-6);
            if (fabsf(actual - scalar) > worst) worst = fabsf(actual - scalar);
            if (!(logical % 8)) {
                float *other[3];
                const float *inputs[4] = {x, NULL, NULL, NULL};
                for (int t = 0; t < 3; t++) {
                    other[t] = xmalloc(logical * sizeof(float));
                    for (int k = 0; k < logical; k++) other[t][k] = x[k] * (t + 1.25f) + 0.1f;
                    inputs[t + 1] = other[t];
                }
                for (uint32_t count = 1; count <= 4; count++) {
                    float outputs[4], gates[4], ups[4];
                    const size_t rb = (physical / 256) * 98;
                    uint8_t *up = xmalloc(rb);
                    memcpy(up, row, rb);
                    for (size_t i = 0; i < rb; i++) if (i % 98 >= 2) up[i] ^= (uint8_t)random_word();
                    qwen4_iq3_dot_many(logical, row, inputs, outputs, count);
                    qwen4_iq3_gate_up_many(logical, row, up, inputs, gates, ups, count);
                    for (uint32_t t = 0; t < count; t++) {
                        const double expected_many = reference(row, inputs[t], logical, &magnitude);
                        assert(fabs(outputs[t] - expected_many) <= 2e-6 * magnitude + 1e-6);
                        assert(gates[t] == outputs[t]);
                        const double expected_up = reference(up, inputs[t], logical, &magnitude);
                        assert(fabs(ups[t] - expected_up) <= 2e-6 * magnitude + 1e-6);
                    }
                    free(up);
                }
                for (int t = 0; t < 3; t++) free(other[t]);
            }
        }
        if (logical == 2560) {
            volatile float sink = 0;
            const double t0 = now_sec();
            for (int i = 0; i < 10000; i++) sink += ds4_vec_dot_iq3_xxs_f32(physical, logical, row, x);
            const double t1 = now_sec();
            for (int i = 0; i < 10000; i++) sink += qwen4_iq3_dot_avx2(physical, logical, row, x);
            const double t2 = now_sec();
            const float *inputs[1] = {x};
            float output[1];
            for (int i = 0; i < 10000; i++) {
                qwen4_iq3_dot_many(logical, row, inputs, output, 1);
                sink += output[0];
            }
            const double t3 = now_sec();
            block_q8_K q[10]; qwen4_quantize_q8(logical, x, q);
            const block_q8_K *quantized[1] = {q};
            for (int i = 0; i < 10000; i++) {
                qwen4_iq3_q8_many(logical, row, quantized, output, 1);
                sink += output[0];
            }
            const double t4 = now_sec();
            printf("cached row: Q8 %.3f ms, float/Q8 %.2fx\n", (t4-t3)*1000, (t3-t2)/(t4-t3));
            printf("cached row: scalar %.3f ms SIMD %.3f ms speedup %.2fx (sink=%g)\n",
                    (t1-t0)*1000, (t2-t1)*1000, (t1-t0)/(t2-t1), (double)sink);
            printf("cached row: hybrid %.3f ms, AVX2/hybrid %.2fx\n", (t3-t2)*1000, (t2-t1)/(t3-t2));
        }
        free(row); munmap(map, span + page);
    }
    printf("IQ3 SIMD: 1024 cases, worst scalar difference %g; padded activation guard passed\n", (double)worst);
#else
    puts("IQ3 SIMD: AVX2 unavailable; scalar fallback retained");
#endif
    return 0;
}
