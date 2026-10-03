import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen4_iq2 import IQ3Quantizer, tensor_bytes


class IQ3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('DS4_TEST_GGML_LIBRARY')
        if not path or not Path(path).is_file():
            raise unittest.SkipTest('set DS4_TEST_GGML_LIBRARY to libggml-base.so')
        cls.quant = IQ3Quantizer(Path(path))
        cls.quant.lib.dequantize_row_iq3_xxs.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]

    def dequant(self, payload, rows, width):
        out = np.empty((rows, width), dtype=np.float32)
        raw = ctypes.create_string_buffer(payload)
        row_bytes = width // 256 * 98
        for row in range(rows):
            self.quant.lib.dequantize_row_iq3_xxs(
                ctypes.byref(raw, row * row_bytes),
                out[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), width)
        return out

    def test_gate_up_standard_iq3_with_importance(self):
        rng = np.random.default_rng(39)
        source = rng.normal(size=(2, 2560)).astype(np.float32)
        weights = rng.uniform(0.1, 2.0, size=2560).astype(np.float32)
        payload = self.quant.encode(source, 'IQ3_XXS', weights)
        self.assertEqual(len(payload), tensor_bytes(18, [2560, 2]))
        self.assertEqual(payload, self.quant.encode(source, 'IQ3_XXS', weights))
        changed = weights.copy()
        changed[::2] *= 100
        changed[1::2] *= 0.01
        self.assertNotEqual(payload, self.quant.encode(source, 'IQ3_XXS', changed))

    def test_padded_down_matches_exact_dequantized_dot(self):
        rng = np.random.default_rng(40)
        source = rng.normal(size=(3, 640)).astype(np.float32)
        importance = rng.uniform(0.1, 2.0, size=640).astype(np.float32)
        padded = np.pad(source, ((0, 0), (0, 128)))
        weights = np.pad(importance, (0, 128))
        np.testing.assert_array_equal(padded[:, :640], source)
        np.testing.assert_array_equal(weights[:640], importance)
        self.assertTrue(np.all(padded[:, 640:] == 0))
        self.assertTrue(np.all(weights[640:] == 0))
        payload = self.quant.encode(padded, 'IQ3_XXS', weights)
        self.assertEqual(len(payload), 3 * 3 * 98)
        self.assertEqual(payload, self.quant.encode(padded, 'IQ3_XXS', weights))
        decoded = self.dequant(payload, 3, 768)
        x = rng.normal(size=640).astype(np.float32)
        x_padded = np.pad(x, (0, 128))
        exact = decoded[:, :640] @ x
        actual = decoded @ x_padded
        np.testing.assert_allclose(actual, exact, rtol=1e-6, atol=1e-5)
        self.assertAlmostEqual(len(payload) * 8 / (3 * 640), 3.675)

    def test_cpu_reference_does_not_read_activation_tail(self):
        cc = shutil.which('cc')
        if not cc:
            self.skipTest('C compiler unavailable')
        root = Path(__file__).resolve().parents[2]
        source = '''
#define DS4_NO_GPU 1
#include "ds4.c"
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 3;
    uint8_t row[294];
    float x[641];
    if (fread(row, 1, sizeof(row), f) != sizeof(row) ||
        fread(x, sizeof(float), 641, f) != 641) return 4;
    fclose(f);
    printf("%.9g\\n", ds4_vec_dot_iq3_xxs_f32(768, 640, row, x));
    return 0;
}
'''
        rng = np.random.default_rng(73)
        errors = []
        relatives = []
        with tempfile.TemporaryDirectory() as temporary:
            probe = Path(temporary) / 'probe.c'
            binary = Path(temporary) / 'probe'
            payload_path = Path(temporary) / 'row.bin'
            probe.write_text(source)
            subprocess.run([cc, '-O1', '-std=c99', '-D_GNU_SOURCE',
                            '-ffunction-sections', '-fdata-sections',
                            '-Wl,--gc-sections', '-I', str(root),
                            str(probe), '-lm', '-pthread', '-o', str(binary)],
                           check=True, capture_output=True)
            for _ in range(8):
                matrix = np.pad(rng.normal(size=(1, 640)).astype(np.float32),
                                ((0, 0), (0, 128)))
                weights = np.pad(rng.uniform(0.1, 2, 640).astype(np.float32),
                                 (0, 128))
                payload = self.quant.encode(matrix, 'IQ3_XXS', weights)
                x = rng.normal(size=641).astype(np.float32)
                x[640] = 1e9  # The logical activation ends before this sentinel.
                payload_path.write_bytes(payload + x.tobytes())
                decoded = self.dequant(payload, 1, 768)[0]
                exact = float(np.dot(decoded[:640].astype(np.float64),
                                     x[:640].astype(np.float64)))
                actual = float(subprocess.check_output([str(binary),
                                                        str(payload_path)]))
                errors.append(abs(actual - exact))
                relatives.append(abs(actual - exact) / max(abs(exact), 1e-8))
        print(f'IQ3 CPU padded dot: max abs {max(errors):.3g}, '
              f'mean abs {np.mean(errors):.3g}, max rel {max(relatives):.3g}')
        self.assertLess(max(errors), 1e-3)


if __name__ == '__main__':
    unittest.main()
