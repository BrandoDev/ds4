/* GGML IQ3_XXS routed row dot. The logical width may be smaller than the
 * encoded width (Qwen down: 640 values in three 256-value blocks). */
__device__ __constant__ static const uint32_t qwen4_rocm_iq3_grid[256] = {
#include "../ds4_iq3xxs_grid.inc"
};

__device__ static float qwen4_rocm_iq3_dot(
        const uint8_t *w, const float *x, uint32_t logical_k, uint32_t physical_k) {
    float total = 0.0f;
    for (uint32_t b = 0; b < physical_k / 256u; b++) {
        const uint8_t *p = w + (uint64_t)b * 98u;
        const float d = __half2float(*reinterpret_cast<const __half *>(p));
        for (uint32_t group = 0; group < 8u; group++) {
            const uint32_t aoff = 66u + group * 4u;
            const uint32_t aux = (uint32_t)p[aoff] |
                                 ((uint32_t)p[aoff + 1u] << 8u) |
                                 ((uint32_t)p[aoff + 2u] << 16u) |
                                 ((uint32_t)p[aoff + 3u] << 24u);
            const float scale = d * (0.5f + (float)(aux >> 28u)) * 0.5f;
            for (uint32_t pair = 0; pair < 4u; pair++) {
                const uint8_t signs = cuda_ksigns_iq2xs[(aux >> (pair * 7u)) & 127u];
                const uint32_t g0 = qwen4_rocm_iq3_grid[p[2u + group * 8u + pair * 2u]];
                const uint32_t g1 = qwen4_rocm_iq3_grid[p[3u + group * 8u + pair * 2u]];
                const uint32_t base = b * 256u + group * 32u + pair * 8u;
                for (uint32_t j = 0; j < 8u; j++) {
                    const uint32_t k = base + j;
                    if (k >= logical_k) break; /* never read the padded tail */
                    const uint32_t grid = j < 4u ? g0 : g1;
                    const float v = (float)((grid >> ((j & 3u) * 8u)) & 255u);
                    total += scale * ((signs & (1u << j)) ? -v : v) * x[k];
                }
            }
        }
    }
    return total;
}

__global__ static void qwen4_rocm_iq3_rows_kernel(
        float *out, const uint8_t *weights, const float *x,
        uint32_t rows, uint32_t logical_k, uint32_t physical_k) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint8_t *w = weights + (uint64_t)row * (physical_k / 256u) * 98u;
    out[row] = qwen4_rocm_iq3_dot(w, x, logical_k, physical_k);
}

extern "C" int ds4_gpu_qwen4_iq3_rows_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t rows, uint32_t logical_k, uint32_t physical_k) {
    if (!out || !x || !model_map || !rows || !logical_k ||
        !physical_k || (physical_k % 256u) || logical_k > physical_k ||
        out->bytes < (uint64_t)rows * sizeof(float) ||
        x->bytes < (uint64_t)logical_k * sizeof(float)) return 0;
    const uint64_t row_bytes = (uint64_t)(physical_k / 256u) * 98u;
    if (weight_offset > model_size ||
        (uint64_t)rows * row_bytes > model_size - weight_offset) return 0;
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(
        model_map, weight_offset, (uint64_t)rows * row_bytes, "Qwen IQ3_XXS rows");
    if (!w) return 0;
    qwen4_rocm_iq3_rows_kernel<<<(rows + 127u) / 128u, 128>>>(
        (float *)out->ptr, w, (const float *)x->ptr, rows, logical_k, physical_k);
    return cuda_ok(cudaGetLastError(), "Qwen IQ3_XXS rows");
}
