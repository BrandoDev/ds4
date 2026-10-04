/* Run on a ROCm host: make test-qwen4-rocm-iq3 ROCM_ARCH=gfx1201 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4_gpu.h"

static const uint32_t grid[256] = {
#include "ds4_iq3xxs_grid.inc"
};

static uint32_t rng = 0x9e3779b9u;
static uint32_t next_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static float reference(const uint8_t *row, const float *x, uint32_t logical, uint32_t physical) {
    double sum = 0.0;
    for (uint32_t b = 0; b < physical / 256u; b++) {
        const uint8_t *p = row + b * 98u;
        /* Test blocks use the exactly representable FP16 scale 2^-10. */
        for (uint32_t group = 0; group < 8u; group++) {
            uint32_t aux;
            memcpy(&aux, p + 66u + group * 4u, 4);
            const double scale = (0.5 + (aux >> 28u)) * 0.5 / 1024.0;
            for (uint32_t pair = 0; pair < 4u; pair++) {
                const uint32_t s = (aux >> (pair * 7u)) & 127u;
                const uint32_t signs = s | ((__builtin_parity(s) & 1u) << 7u);
                const uint32_t g0 = grid[p[2u + group * 8u + pair * 2u]];
                const uint32_t g1 = grid[p[3u + group * 8u + pair * 2u]];
                for (uint32_t j = 0; j < 8u; j++) {
                    const uint32_t k = b * 256u + group * 32u + pair * 8u + j;
                    if (k >= logical) break;
                    const uint32_t g = j < 4u ? g0 : g1;
                    const int v = (int)((g >> ((j & 3u) * 8u)) & 255u);
                    sum += scale * (signs & (1u << j) ? -v : v) * x[k];
                }
            }
        }
    }
    return (float)sum;
}

static void fill_iq3_rows(uint8_t *weights, uint32_t rows, uint32_t physical) {
    const size_t rb = physical / 256u * 98u;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t b = 0; b < physical / 256u; b++) {
            uint8_t *p = weights + r * rb + b * 98u;
            p[0] = 0; p[1] = 0x14;
            for (uint32_t j = 2; j < 66; j++) p[j] = (uint8_t)next_u32();
            for (uint32_t j = 66; j < 98; j += 4) {
                uint32_t aux = next_u32();
                memcpy(p + j, &aux, 4);
            }
        }
    }
}

static int run_case(uint32_t logical, uint32_t physical) {
    enum { ROWS = 17 };
    const size_t rb = physical / 256u * 98u;
    const size_t size = ROWS * rb;
    uint8_t *weights = malloc(size);
    float *x = malloc((logical + 16u) * sizeof(float));
    if (!weights || !x) return 0;
    for (uint32_t i = 0; i < logical; i++) x[i] = (float)((int)(next_u32() % 2001u) - 1000) / 1000.0f;
    for (uint32_t i = logical; i < logical + 16u; i++) x[i] = 1000000.0f;
    fill_iq3_rows(weights, ROWS, physical);
    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc((uint64_t)logical * 4u);
    ds4_gpu_tensor *dy = ds4_gpu_tensor_alloc(ROWS * 4u);
    float y[ROWS];
    if (!dx || !dy || !ds4_gpu_tensor_write(dx, 0, x, (uint64_t)logical * 4u) ||
        !ds4_gpu_qwen4_iq3_rows_tensor(dy, dx, weights, size, 0, ROWS, logical, physical) ||
        !ds4_gpu_tensor_read(dy, 0, y, sizeof(y))) {
        fprintf(stderr, "IQ3 ROCm launch/read failed (K=%u/%u)\n", logical, physical);
        return 0;
    }
    double max_abs = 0.0, mean_abs = 0.0, max_rel = 0.0;
    for (uint32_t r = 0; r < ROWS; r++) {
        const float want = reference(weights + r * rb, x, logical, physical);
        if (!isfinite(y[r]) || !isfinite(want)) return 0;
        const double abs_err = fabs((double)y[r] - want);
        const double rel_err = abs_err / fmax(fabs(want), 1e-3);
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
        mean_abs += abs_err / ROWS;
    }
    printf("IQ3 K=%u/%u rows=%u max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n",
           logical, physical, ROWS, max_abs, mean_abs, max_rel);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(dy);
    free(weights); free(x);
    return max_abs <= 0.015 && max_rel <= 2e-3;
}

static int run_expert_forward(void) {
    enum { NE = 7, NS = 2, K = 2560, F = 640, D = 16 };
    const size_t gate_rb = K / 256u * 98u, down_rb = 3u * 98u;
    const size_t gate_bytes = NE * F * gate_rb, down_bytes = NE * D * down_rb;
    const size_t go = 0, uo = gate_bytes, dno = gate_bytes * 2u;
    const size_t model_bytes = dno + down_bytes;
    uint8_t *model = malloc(model_bytes);
    float *x = malloc(K * sizeof(float));
    float *mid_ref = malloc(NS * F * sizeof(float));
    if (!model || !x || !mid_ref) return 0;
    fill_iq3_rows(model + go, NE * F, K);
    fill_iq3_rows(model + uo, NE * F, K);
    fill_iq3_rows(model + dno, NE * D, 768u);
    for (uint32_t i = 0; i < K; i++) x[i] = (float)((int)(next_u32() % 2001u) - 1000) / 10000.0f;
    const int selected[NS] = { 5, 1 };
    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(K * 4u);
    ds4_gpu_tensor *ds = ds4_gpu_tensor_alloc(NS * 4u);
    ds4_gpu_tensor *dm = ds4_gpu_tensor_alloc(NS * F * 4u);
    ds4_gpu_tensor *dy = ds4_gpu_tensor_alloc(NS * D * 4u);
    float got[NS * D];
    if (!dx || !ds || !dm || !dy ||
        !ds4_gpu_tensor_write(dx, 0, x, K * 4u) ||
        !ds4_gpu_tensor_write(ds, 0, selected, sizeof(selected)) ||
        !ds4_gpu_qwen4_moe_mid_tensor(dm, dx, ds, model, model_bytes, go, uo,
                                      18u, NE, 1u, NS, K, F, 0, 0, UINT32_MAX) ||
        !ds4_gpu_qwen4_moe_down_tensor(dy, dm, ds, model, model_bytes, dno,
                                       18u, NE, 1u, NS, F, D, 0, UINT32_MAX) ||
        !ds4_gpu_tensor_read(dy, 0, got, sizeof(got))) {
        fprintf(stderr, "IQ3 expert forward failed\n");
        return 0;
    }
    for (uint32_t s = 0; s < NS; s++) {
        const uint32_t e = selected[s];
        for (uint32_t r = 0; r < F; r++) {
            const float g = reference(model + go + ((size_t)e * F + r) * gate_rb, x, K, K);
            const float u = reference(model + uo + ((size_t)e * F + r) * gate_rb, x, K, K);
            mid_ref[(size_t)s * F + r] = g / (1.0f + expf(-g)) * u;
        }
    }
    double max_abs = 0.0, mean_abs = 0.0, max_rel = 0.0;
    for (uint32_t s = 0; s < NS; s++) for (uint32_t r = 0; r < D; r++) {
        const float want = reference(model + dno + ((size_t)selected[s] * D + r) * down_rb,
                                     mid_ref + (size_t)s * F, F, 768u);
        if (!isfinite(got[s * D + r]) || !isfinite(want)) return 0;
        const double err = fabs((double)got[s * D + r] - want);
        const double rel = err / fmax(fabs(want), 1e-3);
        if (err > max_abs) max_abs = err;
        if (rel > max_rel) max_rel = rel;
        mean_abs += err / (NS * D);
    }
    printf("IQ3 expert forward slots=%u max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n",
           NS, max_abs, mean_abs, max_rel);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(ds);
    ds4_gpu_tensor_free(dm); ds4_gpu_tensor_free(dy);
    free(model); free(x); free(mid_ref);
    return max_abs <= 0.1 && max_rel <= 1e-2;
}

static void fill_q2_rows(uint8_t *weights, uint32_t rows, uint32_t physical) {
    const size_t rb = physical / 256u * 84u;
    for (uint32_t r = 0; r < rows; r++) for (uint32_t b = 0; b < physical / 256u; b++) {
        uint8_t *p = weights + r * rb + b * 84u;
        for (uint32_t i = 0; i < 16u; i++) p[i] = (uint8_t)(1u + next_u32() % 4u);
        for (uint32_t i = 16u; i < 80u; i++) p[i] = (uint8_t)next_u32();
        p[80] = 0; p[81] = 0x14; p[82] = 0; p[83] = 0;
    }
}

static float reference_q2(const uint8_t *row, const float *x, uint32_t logical) {
    double total = 0.0;
    for (uint32_t i = 0; i < logical; i++) {
        const uint8_t *p = row + (i / 256u) * 84u;
        const uint32_t j = i & 255u;
        const uint32_t q = (p[16u + (j / 128u) * 32u + j % 32u] >>
                            (2u * ((j % 128u) / 32u))) & 3u;
        total += (double)(p[j / 16u] & 15u) * q * x[i] / 1024.0;
    }
    return (float)total;
}

static int run_q2_baseline(void) {
    enum { NE = 11, NS = 2, K = 2560, F = 640, D = 16 };
    const size_t grb = K / 256u * 84u, drb = 3u * 84u;
    const size_t gb = NE * F * grb, db = NE * D * drb;
    uint8_t *model = malloc(gb * 2u + db);
    float *x = malloc(K * sizeof(float));
    float *mid = malloc(NS * F * sizeof(float));
    if (!model || !x || !mid) return 0;
    fill_q2_rows(model, NE * F, K);
    fill_q2_rows(model + gb, NE * F, K);
    fill_q2_rows(model + gb * 2u, NE * D, 768u);
    for (uint32_t i = 0; i < K; i++) x[i] = (float)((int)(next_u32() % 2001u) - 1000) / 1000.0f;
    const int selected[NS] = { 9, 3 };
    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc(K * 4u);
    ds4_gpu_tensor *ds = ds4_gpu_tensor_alloc(NS * 4u);
    ds4_gpu_tensor *dm = ds4_gpu_tensor_alloc(NS * F * 4u);
    ds4_gpu_tensor *dy = ds4_gpu_tensor_alloc(NS * D * 4u);
    float got[NS * D];
    if (!dx || !ds || !dm || !dy ||
        !ds4_gpu_tensor_write(dx, 0, x, K * 4u) ||
        !ds4_gpu_tensor_write(ds, 0, selected, sizeof(selected)) ||
        !ds4_gpu_qwen4_moe_mid_tensor(dm, dx, ds, model, gb * 2u + db, 0, gb,
                                      10u, NE, 1u, NS, K, F, 0, 0, UINT32_MAX) ||
        !ds4_gpu_qwen4_moe_down_tensor(dy, dm, ds, model, gb * 2u + db, gb * 2u,
                                       10u, NE, 1u, NS, F, D, 0, UINT32_MAX) ||
        !ds4_gpu_tensor_read(dy, 0, got, sizeof(got))) {
        fprintf(stderr, "Q2 baseline expert forward failed\n");
        return 0;
    }
    for (uint32_t s = 0; s < NS; s++) for (uint32_t r = 0; r < F; r++) {
        const uint32_t e = selected[s];
        const float g = reference_q2(model + ((size_t)e * F + r) * grb, x, K);
        const float u = reference_q2(model + gb + ((size_t)e * F + r) * grb, x, K);
        mid[(size_t)s * F + r] = g / (1.0f + expf(-g)) * u;
    }
    double max_abs = 0.0, mean_abs = 0.0, max_rel = 0.0;
    for (uint32_t s = 0; s < NS; s++) for (uint32_t r = 0; r < D; r++) {
        const float want = reference_q2(model + gb * 2u + ((size_t)selected[s] * D + r) * drb,
                                        mid + (size_t)s * F, F);
        if (!isfinite(got[s * D + r]) || !isfinite(want)) return 0;
        const double err = fabs((double)got[s * D + r] - want);
        const double rel = err / fmax(fabs(want), 1e-3);
        if (err > max_abs) max_abs = err;
        if (rel > max_rel) max_rel = rel;
        mean_abs += err / (NS * D);
    }
    printf("Q2 baseline expert forward slots=%u max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n",
           NS, max_abs, mean_abs, max_rel);
    ds4_gpu_tensor_free(dx); ds4_gpu_tensor_free(ds);
    ds4_gpu_tensor_free(dm); ds4_gpu_tensor_free(dy);
    free(model); free(x); free(mid);
    return max_abs <= 0.1 && max_rel <= 1e-2;
}

int main(void) {
    if (!ds4_gpu_init()) { fprintf(stderr, "ROCm GPU unavailable\n"); return 2; }
    ds4_gpu_qwen4_use_device_weight_copies();
    const int ok = run_q2_baseline() && run_case(2560u, 2560u) &&
                   run_case(640u, 768u) && run_expert_forward();
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
