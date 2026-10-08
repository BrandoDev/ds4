#!/usr/bin/env python3
"""Sequential full-model float/Q8 diagnostic: speed, greedy output, NLL and logits.

Q8 is experimental. This small regression pilot does not certify unchanged
accuracy. The original float activation path remains the production default.
"""
import hashlib, json, os, re, subprocess, time
from pathlib import Path
import numpy as np
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('gguf/benchmarks/hybrid-cpu-q8'))
parser.add_argument('--model', type=Path, default=Path('gguf/Qwen3.8-Flash-Next-IQ3.gguf'))
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[2]
BASE = args.output.resolve()
BASE.mkdir(parents=True, exist_ok=True)
MODEL = args.model.resolve()
ENV = {k: v for k, v in os.environ.items() if not k.startswith('DS4_')}
REPORT = {'scope': 'Experimental CPU IQ3 x Q8_K, original weights; float stays default pending quality assessment', 'runs': []}
common = ['--rocm', '-m', str(MODEL), '--gpu-memory-budget', '14GB', '--gpu-expert-layers', '10', '--ctx', '4096', '--prefill-chunk', '128', '--threads', '16', '--nothink', '--temp', '0']

def save():
    (BASE / 'summary.json').write_text(json.dumps(REPORT, indent=2, ensure_ascii=False) + '\n')

def run(name, args, q8=False):
    item = {'name': name, 'command': args, 'environment': {k: v for k, v in ENV.items() if k.startswith('DS4_')} | ({'DS4_QWEN4_CPU_Q8': '1'} if q8 else {}), 'peak_vram_bytes': 0, 'peak_rss_kib': 0, 'major_faults': 0}
    REPORT['runs'].append(item)
    save()
    start = time.monotonic()
    with (BASE / (name + '.txt')).open('w') as out, (BASE / (name + '.log')).open('w') as err:
        proc = subprocess.Popen(args, cwd=ROOT, env=ENV | item['environment'], stdout=out, stderr=err, stdin=subprocess.DEVNULL)
        try:
            while proc.poll() is None:
                if time.monotonic() - start > 300:
                    raise TimeoutError(name)
                for path in Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_used'):
                    item['peak_vram_bytes'] = max(item['peak_vram_bytes'], int(path.read_text()))
                try:
                    status = Path(f'/proc/{proc.pid}/status').read_text()
                    rss = re.search('^VmRSS:\\s+(\\d+)', status, re.M)
                    if rss:
                        item['peak_rss_kib'] = max(item['peak_rss_kib'], int(rss[1]))
                    stat = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()
                    item['major_faults'] = max(item['major_faults'], int(stat[9]))
                except FileNotFoundError:
                    pass
                time.sleep(0.1)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    item['seconds'] = time.monotonic() - start
    item['exit_code'] = proc.returncode
    item['response'] = (BASE / (name + '.txt')).read_text()
    log = (BASE / (name + '.log')).read_text()
    item['placement'] = [l for l in log.splitlines() if 'resident placement:' in l]
    match = re.search('prefill: ([\\d.]+) t/s, generation: ([\\d.]+) t/s', log)
    if match:
        item['prefill_tps'], item['decode_tps'] = map(float, match.groups())
    item['perplexity_lines'] = [l for l in log.splitlines() if 'perplex' in l or 'nll' in l]
    save()
    print(name, json.dumps(item, ensure_ascii=False), flush=True)
    assert proc.returncode == 0
    assert item['peak_vram_bytes'] < 14 * 2 ** 30
    return item
stat = MODEL.stat()
with MODEL.open('rb') as source:
    header_hash = hashlib.sha256(source.read(8 << 20)).hexdigest()
REPORT['model'] = {'path': str(MODEL), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'first_8MiB_sha256': header_hash}
REPORT['source_sha256'] = hashlib.sha256((ROOT / 'ds4.c').read_bytes()).hexdigest()
REPORT['binary_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ('ds4', 'tests/test_qwen4_iq3_dot', 'tests/test_qwen4_ngram_state')}
run('cpu-dot', ['./tests/test_qwen4_iq3_dot'])
for name, prompt in [('count', 'Count from 1 to 10, separated by commas. Reply with only the numbers.'), ('fact', 'Quando è morto Silvio Berlusconi? Rispondi solo con la data.'), ('prose', 'Spiega in due frasi perché il cielo appare azzurro.')]:
    a = run(name + '-float', ['./ds4'] + common + ['--mtp', '--mtp-timing', '-n', '32', '-p', prompt])
    b = run(name + '-q8', ['./ds4'] + common + ['--mtp', '--mtp-timing', '-n', '32', '-p', prompt], True)
    REPORT.setdefault('greedy_comparisons', []).append({'case': name, 'equal': a['response'] == b['response'], 'speedup': b['decode_tps'] / a['decode_tps']})
    save()
sample = json.loads((ROOT / 'gguf/benchmarks/arc-challenge/iq3-verified-fast-300/sample.json').read_text())
corpus = '\n\n'.join(('Question: ' + r['question']['stem'] + '\nAnswer: ' + next((c['text'] for c in r['question']['choices'] if c['label'] == r['answerKey'])) for r in sample[:20]))
(BASE / 'arc-corpus.txt').write_text(corpus)
REPORT['perplexity_protocol'] = {'source': 'Existing ARC-Challenge seed-42 sample, first 20 records question and gold answer text', 'corpus_sha256': hashlib.sha256(corpus.encode()).hexdigest(), 'prefix_tokens': 32, 'scored_tokens': 256, 'purpose': 'Paired teacher-forced NLL regression pilot, not ARC accuracy'}
save()
for q8 in (False, True):
    run('ppl-' + ('q8' if q8 else 'float'), ['./ds4'] + common + ['--perplexity-file', str(BASE / 'arc-corpus.txt'), '-n', '256'], q8)
ppls = []
for run_item in REPORT['runs']:
    match = re.search('avg_nll=([\\d.]+) ppl=([\\d.]+)', run_item['response'])
    if match:
        ppls.append({'case': run_item['name'], 'avg_nll': float(match[1]), 'ppl': float(match[2])})
REPORT['perplexity_comparison'] = ppls
REPORT['ppl_relative_change'] = ppls[1]['ppl'] / ppls[0]['ppl'] - 1
for mode in ('float', 'q8'):
    inputs = BASE / (mode + '-inputs.tsv')
    inputs.write_text(''.join((ids + '\t' + str(BASE / (mode + '-' + name + '.bin')) + '\n' for name, ids in [('all', '9419,1814,13'), ('teacher-a', '9419,1814,13|15,16'), ('teacher-b', '9419,1814,13|17,18')])))
    env_save = ENV.copy()
    ENV.update({'DS4_QWEN4_GPU': '1', 'DS4_QWEN4_FT_LIST': str(inputs)})
    run('logits-' + mode, ['./ds4'] + common + ['--first-token-test', '-p', 'Hybrid parity'], mode == 'q8')
    ENV.clear()
    ENV.update(env_save)
REPORT['logit_comparisons'] = []
for name in ('all', 'teacher-a', 'teacher-b'):
    a = np.fromfile(BASE / ('float-' + name + '.bin'), dtype=np.float32).reshape(-1, 248320).astype(np.float64)
    b = np.fromfile(BASE / ('q8-' + name + '.bin'), dtype=np.float32).reshape(a.shape).astype(np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()

    def logsoftmax(x):
        x = x - x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    la, lb = (logsoftmax(a), logsoftmax(b))
    REPORT['logit_comparisons'].append({'case': name, 'shape': list(a.shape), 'max_abs_difference': float(np.abs(a - b).max()), 'mean_abs_difference': float(np.abs(a - b).mean()), 'top1_equal_each_row': (a.argmax(axis=1) == b.argmax(axis=1)).tolist(), 'kl_float_to_q8_each_row': (np.exp(la) * (la - lb)).sum(axis=1).tolist()})
    save()
run('q8-state-regression', ['stdbuf', '-oL', './tests/test_qwen4_ngram_state', str(MODEL), 'hybrid'], True)
REPORT['completed'] = True
REPORT['quality_unchanged_certified'] = False
REPORT['production_default'] = 'float'
save()
