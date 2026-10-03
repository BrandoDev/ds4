#!/usr/bin/env python3
"""Build calibrated Qwen routed experts: IQ2/Q2 or experimental IQ3_XXS.

The two stages allow quantization beside the original BF16 checkpoint and
assembly beside an existing qwen4exp model. All other tensors are byte-copied.

For IQ3, assemble gate-up from the existing combined padded-Q2 model, then
assemble down from that IQ3-gate/up result. This preserves the protected Q2
recipe for every non-routed tensor, including MTP and the disk-only PLE table.
Pass the path to a standard GGML libggml-base shared library to --library for
IQ3 quantization; GGML's normal imatrix-aware IQ3_XXS algorithm is used.
"""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from qwen4_pack import (SourceDB, QwenGGUFImatrix, GGMLQuantizer,
                        Q2_IMATRIX_SHA256, encode_weighted_experts)
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str, tnbytes


class IQ3Quantizer:
    """Use GGML's standard, imatrix-aware IQ3_XXS encoder verbatim."""

    def __init__(self, library):
        import numpy as np
        self.np = np
        self.lib = ctypes.CDLL(str(library.resolve()))
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_init.restype = None
        self.lib.ggml_quantize_init(18)

    def encode(self, values, qtype, imatrix):
        if qtype != 'IQ3_XXS' or imatrix is None:
            raise ValueError('IQ3_XXS requires its importance matrix')
        np = self.np
        from qwen4_pack import bf16_to_f32
        source = bf16_to_f32(values) if values.dtype == np.dtype('<u2') else values
        source = np.ascontiguousarray(source, dtype='<f4')
        weights = np.ascontiguousarray(imatrix, dtype='<f4')
        if (source.ndim < 2 or source.shape[-1] % 256 or
                weights.shape != (source.shape[-1],)):
            raise ValueError('invalid IQ3_XXS source or importance geometry')
        if not np.all(np.isfinite(source)):
            raise ValueError('nonfinite IQ3_XXS source values')
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError('nonfinite or negative IQ3_XXS importance values')
        rows, width = source.size // source.shape[-1], source.shape[-1]
        out = np.empty(rows * (width // 256) * 98, dtype=np.uint8)
        written = self.lib.ggml_quantize_chunk(18,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.c_void_p), 0, rows, width,
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if written != out.nbytes:
            raise ValueError(f'GGML IQ3_XXS wrote {written} of {out.nbytes} bytes')
        return out.tobytes()


def tensor_bytes(kind, dims):
    n = math.prod(dims)
    if kind in (10, 16, 18):
        if dims[0] % 256:
            raise ValueError('K-quant weight rows must be block aligned')
        return n // 256 * {10: 84, 16: 66, 18: 98}[kind]
    return tnbytes(kind, n)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def quantize(args):
    args.experts_dir.mkdir(parents=True, exist_ok=True)
    db = SourceDB(args.source)
    imatrix = QwenGGUFImatrix(args.imatrix, Q2_IMATRIX_SHA256)
    iq3 = args.recipe == 'iq3'
    quant = IQ3Quantizer(args.library) if iq3 else GGMLQuantizer(args.library)
    if not iq3:
        quant.lib.ds4q_quantize_init(16)  # Initialize shared lookup tables before workers.
    manifest = args.experts_dir / 'experts.json'
    down = args.projection == 'down'
    qtype = 'IQ3_XXS' if iq3 else ('Q2_K' if down else 'IQ2_XXS')
    identity = {'source': str(args.source.resolve()), 'imatrix': imatrix.provenance(),
                'library_sha256': digest(args.library), 'format': qtype}
    if down:
        identity['padding'] = {'logical_input': 640, 'physical_input': 768}
    report = json.loads(manifest.read_text()) if manifest.exists() else {**identity, 'tensors': {}}
    for key, value in identity.items():
        if report[key] != value:
            raise SystemExit(f'Resume provenance mismatch: {key}')
    for layer in range(48):
        source = f'model.language_model.layers.{layer}.mlp.experts.' + ('down_proj' if down else 'gate_up_proj')
        info = db.tensors[source]
        shape = (512, 2560, 640) if down else (512, 1280, 2560)
        if info['dtype'] != 'BF16' or info['shape'] != shape:
            raise SystemExit(f'Unexpected source layout: {source}: {info}')
        names = [f'blk.{layer}.ffn_{part}_exps.weight' for part in (('down',) if down else ('gate', 'up'))]
        if all(n in report['tensors'] and (args.experts_dir / n).is_file() and
               digest(args.experts_dir / n) == report['tensors'][n]['sha256'] for n in names):
            print(f'Layer {layer}: verified existing tensors', flush=True)
            continue
        values = db.read(source)
        source_hash = hashlib.sha256(values).hexdigest()
        for part, name in enumerate(names):
            start = time.monotonic()
            raw = encode_weighted_experts(values if down else values[:, part * 640:(part + 1) * 640, :],
                                         qtype, imatrix, name, quant, args.threads,
                                         pad_last_to=768 if down else 0)
            block_bytes = 98 if iq3 else (84 if down else 66)
            blocks = 3 if down else 10
            rows = 2560 if down else 640
            if len(raw) != 512 * rows * blocks * block_bytes:
                raise SystemExit(f'Invalid output size for {name}')
            path = args.experts_dir / name
            tmp = path.with_suffix('.incomplete')
            tmp.write_bytes(raw)
            tmp.replace(path)
            report['tensors'][name] = {'sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw), 'source_tensor': source, 'source_sha256': source_hash,
                'seconds': time.monotonic() - start}
            save(manifest, report)
            print(f'{name}: {len(raw)} bytes, {time.monotonic() - start:.1f}s', flush=True)
        del values
    imatrix.close()


def assemble(args):
    if args.out.exists() or Path(str(args.out) + '.incomplete').exists():
        raise SystemExit('Output already exists; choose a new path')
    manifest = json.loads((args.experts_dir / 'experts.json').read_text())
    down = args.projection == 'down'
    iq3 = args.recipe == 'iq3'
    expected = {f'blk.{i}.ffn_{p}_exps.weight' for i in range(48) for p in (('down',) if down else ('gate', 'up'))}
    if set(manifest['tensors']) != expected:
        raise SystemExit(f'Expected all {len(expected)} calibrated trunk tensors')
    if manifest['format'] != ('IQ3_XXS' if iq3 else ('Q2_K' if down else 'IQ2_XXS')):
        raise SystemExit('Quantization format does not match projection')
    if down and manifest.get('padding') != {'logical_input': 640, 'physical_input': 768}:
        raise SystemExit('Invalid down projection padding provenance')
    base = Reader(str(args.template))
    if base.kv.get('general.architecture') != 'qwen4exp' or base.kv.get('qwen4exp.block_count') != 49:
        raise SystemExit('Template must be a combined 49-layer qwen4exp GGUF')
    ngram = base.tensors.get('per_layer_token_embd.weight')
    if ngram is not None and ngram[:2] != (30, [160, 320001536]):
        raise SystemExit('Unexpected embedded BF16 PLE table layout')
    if ngram is not None and next(reversed(base.tensors)) != 'per_layer_token_embd.weight':
        raise SystemExit('Embedded BF16 PLE table must follow resident weights')
    if down:
        for layer in range(48):
            for part in ('gate', 'up'):
                kind, dims, _ = base.tensors[f'blk.{layer}.ffn_{part}_exps.weight']
                if kind != (18 if iq3 else 16) or dims != [2560, 640, 512]:
                    raise SystemExit('Down experiment requires matching routed gate/up tensors')
    alignment = 4096 if ngram else base.kv.get('general.alignment', 32)
    align = lambda n: (n + alignment - 1) // alignment * alignment
    metadata = {k: (base.kv_types[k], v) for k, v in base.kv.items()}
    if ngram:
        metadata['general.alignment'] = (4, alignment)
    metadata['general.name'] = (8, 'Qwen3.8 Flash Next ' +
        ('IQ3_XXS' if iq3 else 'IQ2_XXS') + ' imatrix trunk, ' +
        ('padded ' + ('IQ3_XXS' if iq3 else 'Q2_K') + ' down' if down else 'existing down') + ', MTP')
    metadata['general.file_type'] = (4, 17 if iq3 else 19)
    metadata['ds4.' + ('iq3' if iq3 else 'iq2') + '.imatrix_sha256'] = (8, manifest['imatrix']['sha256'])
    if down:
        metadata['ds4.qwen4.down.logical_input'] = (4, 640)
        metadata['ds4.qwen4.down.physical_input'] = (4, 768)
    plan = []
    offset = 0
    for name, (kind, dims, old_offset) in base.tensors.items():
        if name == 'per_layer_token_embd.weight':
            offset = (offset + 4095) & ~4095
        replace = name in expected
        # IQ3 replaces the routed tensors in the completed Q2 recipe:
        # IQ2_XXS gate/up and physically padded Q2_K down. The Q2 assembly
        # stage instead starts from Q4_K gate/up and MXFP4 down.
        expected_kind = (10 if iq3 else 39) if down else (16 if iq3 else 12)
        expected_dims = [768, 2560, 512] if down and iq3 else (
            [640, 2560, 512] if down else [2560, 640, 512])
        if replace and (kind != expected_kind or dims != expected_dims):
            raise SystemExit(f'Unexpected template expert layout: {name}')
        if replace:
            kind = 18 if iq3 else (10 if down else 16)
            if down:
                dims = [768, 2560, 512]
        size = tensor_bytes(kind, dims)
        if replace:
            path = args.experts_dir / name
            if path.stat().st_size != size or digest(path) != manifest['tensors'][name]['sha256']:
                raise SystemExit(f'Invalid calibrated payload: {name}')
        plan.append((name, kind, dims, offset, size, old_offset))
        offset = align(offset + size)
    prefix = b'GGUF' + struct.pack('<IQQ', 3, len(plan), len(metadata))
    prefix += b''.join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())
    for name, kind, dims, pos, size, old in plan:
        prefix += w_str(name) + struct.pack('<I', len(dims))
        prefix += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, pos)
    data_start = align(len(prefix))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.out) + '.incomplete')
    records = {}
    with tmp.open('wb') as out:
        out.write(prefix)
        out.write(bytes(data_start - len(prefix)))
        for name, kind, dims, pos, size, old in plan:
            out.seek(data_start + pos)
            src = (args.experts_dir / name).open('rb') if name in expected else base.f
            if src is base.f:
                src.seek(base.data_start + old)
            h = hashlib.sha256()
            left = size
            while left:
                chunk = src.read(min(left, 8 << 20))
                if not chunk:
                    raise SystemExit(f'Short tensor read: {name}')
                out.write(chunk)
                h.update(chunk)
                left -= len(chunk)
            if src is not base.f:
                src.close()
            records[name] = {'type': kind, 'bytes': size, 'sha256': h.hexdigest(),
                             'replaced': name in expected}
        out.truncate(data_start + offset)
    # Read back every payload before publishing the completed artifact.
    check = Reader(str(tmp))
    for name, rec in records.items():
        check.f.seek(check.data_start + check.tensors[name][2])
        h = hashlib.sha256()
        left = rec['bytes']
        while left:
            chunk = check.f.read(min(left, 8 << 20))
            if not chunk:
                raise SystemExit(f'Short verification read: {name}')
            h.update(chunk)
            left -= len(chunk)
        if h.hexdigest() != rec['sha256']:
            raise SystemExit(f'Written tensor mismatch: {name}')
    check.f.close()
    base.f.close()
    tmp.replace(args.out)
    by_type = {}
    routed = {'gate': 0, 'up': 0, 'down': 0}
    trunk_routed = {'gate': 0, 'up': 0, 'down': 0}
    resident = 0
    disk_only_ngrams = 0
    for name, rec in records.items():
        by_type[str(rec['type'])] = by_type.get(str(rec['type']), 0) + 1
        if name == 'per_layer_token_embd.weight':
            disk_only_ngrams += rec['bytes']
        else:
            resident += rec['bytes']
        for part in routed:
            if name.endswith(f'.ffn_{part}_exps.weight'):
                routed[part] += rec['bytes']
                if name.startswith('blk.') and int(name.split('.')[1]) < 48:
                    trunk_routed[part] += rec['bytes']
    routed_total = sum(routed.values())
    storage = {'routed_bytes': routed, 'routed_total_bytes': routed_total,
               'trunk_routed_bytes': trunk_routed,
               'non_routed_resident_bytes': resident - routed_total,
               'main_mtp_resident_tensor_bytes': resident,
               'disk_only_ngrams_bytes': disk_only_ngrams,
               'tensor_count_by_ggml_type': by_type}
    if iq3 and down:
        logical_weights = 48 * 512 * 2560 * 640
        storage['trunk_down_effective_bpw'] = trunk_routed['down'] * 8 / logical_weights
    save(Path(str(args.out) + '.json'), {'template': str(args.template.resolve()),
        'quantization': manifest, 'tensors': records, 'bytes': args.out.stat().st_size,
        'sha256': digest(args.out), 'storage': storage})
    print(f'Verified {args.out}: {args.out.stat().st_size} bytes', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='stage', required=True)
    q = sub.add_parser('quantize')
    q.add_argument('--source', type=Path, required=True)
    q.add_argument('--imatrix', type=Path, required=True)
    q.add_argument('--library', type=Path, required=True,
                   help='DS4 libds4quants for q2; standard libggml-base for iq3')
    q.add_argument('--threads', type=int, default=8)
    a = sub.add_parser('assemble')
    a.add_argument('--template', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    for p in (q, a):
        p.add_argument('--experts-dir', type=Path, required=True)
        p.add_argument('--projection', choices=('gate-up', 'down'), default='gate-up')
        p.add_argument('--recipe', choices=('q2', 'iq3'), default='q2')
    args = ap.parse_args()
    (quantize if args.stage == 'quantize' else assemble)(args)


if __name__ == '__main__':
    main()
