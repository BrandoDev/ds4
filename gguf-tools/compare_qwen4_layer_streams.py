#!/usr/bin/env python3
"""Compare CPU first-token hidden streams with Qwen ROCm per-layer dumps.

Use DS4_QWEN4_GPU_CHUNK=1 so every GPU dump contains one token row.
"""

import argparse
import math
from pathlib import Path
import re
import struct


def read_f32(path):
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"{path}: byte length is not divisible by four")
    return [v[0] for v in struct.iter_unpack("<f", data)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path, help="DS4_QWEN4_FT_HIDDEN output")
    parser.add_argument("gpu_prefix", type=Path, help="DS4_ROCM_GRAPH_DUMP_PREFIX value")
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()

    prefix = args.gpu_prefix
    layer0 = sorted(prefix.parent.glob(prefix.name + "_qwen_layer_stream-0_pos*.bin"))
    pattern = re.compile(re.escape(prefix.name) + r"_qwen_layer_stream-0_pos(\d+)\.bin$")
    positions = sorted(int(m.group(1)) for p in layer0 if (m := pattern.fullmatch(p.name)))
    if not positions or positions != list(range(len(positions))):
        raise ValueError("GPU layer-0 dumps must cover consecutive positions from zero; use GPU_CHUNK=1")
    width = len(read_f32(layer0[0]))
    cpu = read_f32(args.cpu)
    if not width or len(cpu) % (len(positions) * width):
        raise ValueError("CPU hidden dump shape does not match GPU token dumps")
    layers = len(cpu) // (len(positions) * width)
    first_bad = None
    for layer in range(layers):
        max_abs = total_abs = max_rel = 0.0
        bad = 0
        for pos in positions:
            path = Path(f"{prefix}_qwen_layer_stream-{layer}_pos{pos}.bin")
            gpu = read_f32(path)
            if len(gpu) != width:
                raise ValueError(f"{path}: expected {width} values, got {len(gpu)}")
            off = (pos * layers + layer) * width
            for a, b in zip(cpu[off:off + width], gpu):
                if not math.isfinite(a) or not math.isfinite(b):
                    raise ValueError(f"non-finite value at layer {layer}, position {pos}")
                err = abs(a - b)
                max_abs = max(max_abs, err)
                total_abs += err
                max_rel = max(max_rel, err / max(abs(a), 1e-6))
                bad += err > args.atol + args.rtol * abs(a)
        print(f"layer {layer:02d}: max_abs={max_abs:.6g} mean_abs={total_abs / (len(positions) * width):.6g} "
              f"max_rel={max_rel:.6g} outside_tol={bad}")
        if bad and first_bad is None:
            first_bad = layer
    print(f"first divergent layer: {first_bad if first_bad is not None else 'none'}")
    return int(first_bad is not None)


if __name__ == "__main__":
    raise SystemExit(main())
