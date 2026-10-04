#!/usr/bin/env python3
"""Score ARC-Challenge with DS4's existing Qwen logit diagnostics.

Protocol: zero-shot raw ``Question: ...\nAnswer:`` context, a leading space
before each complete choice, sum of continuation log probabilities, and
character-length acc_norm as in EleutherAI's ARC task. No chat template,
sampling, MTP or answer extraction. One model load serves the entire run.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path, expected):
    if expected is None:
        return None
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected.lower()):
        raise ValueError('expected model SHA-256 must have 64 hexadecimal characters')
    actual = sha256(path)
    if actual != expected.lower():
        raise ValueError(f'model SHA-256 mismatch: {actual}')
    return actual


def select_sample(dataset, count, seed):
    records = [json.loads(line) for line in dataset.read_text().splitlines() if line]
    if len(records) != 1172 or len({r['id'] for r in records}) != len(records):
        raise ValueError('expected the complete 1172-question ARC-Challenge test split')
    if not 1 <= count <= len(records):
        raise ValueError('sample size must be between 1 and 1172')
    random.Random(seed).shuffle(records)
    return records[:count]


def tokenize(binary, model, text):
    result = subprocess.run([str(binary), '-m', str(model), '--dump-tokens',
                             '--raw-prompt', '-p', text],
                            check=True, capture_output=True, text=True)
    tokens = json.loads(result.stdout.splitlines()[0])
    if not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('invalid tokenizer output')
    return tokens


def continuation_logprobs(path, input_tokens, target_tokens, context_length,
                          teacher_forced=False):
    rows = len(target_tokens) if teacher_forced else len(input_tokens)
    if not rows or path.stat().st_size % (rows * 4):
        raise ValueError(f'invalid logit dump shape: {path}')
    vocab = path.stat().st_size // (rows * 4)
    if vocab <= max(target_tokens):
        raise ValueError('target token outside vocabulary')
    mapped = np.memmap(path, dtype='<f4', mode='r', shape=(rows, vocab))
    # Output row i predicts token i+1. The context's last row predicts the
    # first choice token; the last input excludes the final scored token.
    start = 0 if teacher_forced else context_length - 1
    logits = np.array(mapped[start:start + len(target_tokens)], dtype=np.float64)
    del mapped
    if logits.shape[0] != len(target_tokens) or not np.isfinite(logits).all():
        raise ValueError('missing or nonfinite continuation logits')
    maximum = logits.max(axis=1)
    logsum = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
    return logits[np.arange(len(target_tokens)), target_tokens] - logsum


def grade(record, scores):
    choices = record['question']['choices']
    gold = [c['label'] for c in choices].index(record['answerKey'])
    if len(scores) != len(choices) or not all(math.isfinite(s) for s in scores):
        raise ValueError('incomplete or nonfinite choice scores')
    lengths = [len(c['text']) for c in choices]
    if not all(lengths):
        raise ValueError('empty ARC choice')
    prediction = max(range(len(scores)), key=lambda i: scores[i])
    normalized = [s / n for s, n in zip(scores, lengths)]
    prediction_norm = max(range(len(scores)), key=lambda i: normalized[i])
    return {'id': record['id'], 'gold': gold,
            'prediction': prediction, 'prediction_norm': prediction_norm,
            'correct': prediction == gold, 'correct_norm': prediction_norm == gold,
            'choice_loglikelihoods': scores, 'choice_character_lengths': lengths,
            'choice_normalized_loglikelihoods': normalized}


def wilson(successes, total):
    p, z = successes / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def summarize(results, expected, elapsed, parity):
    summary = {'completed': len(results), 'expected': expected,
               'complete': len(results) == expected, 'elapsed_seconds': elapsed,
               'prefill_vs_teacher_forced_check': parity}
    if results:
        for metric, field in [('acc', 'correct'), ('acc_norm', 'correct_norm')]:
            successes = sum(r[field] for r in results)
            summary[metric] = successes / len(results)
            summary[metric + '_correct'] = successes
            summary[metric + '_wilson_95'] = wilson(successes, len(results))
        summary['seconds_per_question'] = elapsed / len(results)
    return summary


def load_previous(directory, metadata, records):
    old = json.loads((directory / 'run.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    if not summary.get('success'):
        raise ValueError('resume-from must be a successful completed run')
    for key in ('task', 'num_fewshot', 'chat_template', 'target_delimiter',
                'acc_norm_length', 'scoring', 'dataset_sha256', 'sample_sha256',
                'seed', 'sample_size', 'backend', 'model', 'model_bytes',
                'model_mtime_ns', 'model_first_8MiB_sha256', 'binary_sha256'):
        if old[key] != metadata[key]:
            raise ValueError(f'resume provenance mismatch: {key}')
    if old.get('model_sha256') != metadata.get('model_sha256'):
        raise ValueError('resume provenance mismatch: model_sha256')
    previous = [json.loads(line) for line in (directory / 'scores.jsonl').read_text().splitlines()]
    if len(previous) != summary['completed'] or len(previous) >= metadata['max_cases']:
        raise ValueError('previous run must be a shorter completed prefix')
    for record, result in zip(records, previous):
        checked = grade(record, result['choice_loglikelihoods'])
        if any(result[key] != value for key, value in checked.items()):
            raise ValueError('previous scores do not match the sample and grading protocol')
    return previous, summary['elapsed_seconds']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True,
                        help='original ARC-Challenge-Test.jsonl from AllenAI')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=ROOT / 'ds4')
    parser.add_argument('--expected-model-sha256',
                        help='verify the complete GGUF before inference against its published SHA-256')
    parser.add_argument('--backend', choices=('rocm', 'cuda', 'metal'), default='rocm')
    parser.add_argument('--sample-size', type=int, default=300)
    parser.add_argument('--max-cases', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scoring', choices=('teacher-forced', 'all-rows'),
                        default='teacher-forced',
                        help='teacher forcing is the reference; all-rows is a checked shortcut')
    parser.add_argument('--resume-from', type=Path,
                        help='extend a completed pilot with identical sample and model')
    args = parser.parse_args()
    if not 1 <= args.max_cases <= args.sample_size:
        parser.error('max-cases must be between 1 and sample-size')
    for name in ('dataset', 'model', 'binary', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    dumps = args.output / 'logits'
    dumps.mkdir()
    records = select_sample(args.dataset, args.sample_size, args.seed)
    save_json(args.output / 'sample.json', records)
    stat = args.model.stat()
    model_hash = None
    if args.expected_model_sha256:
        print('Verifying the complete model SHA-256 before inference', flush=True)
        model_hash = verify_model(args.model, args.expected_model_sha256)
    with args.model.open('rb') as f:
        model_header_hash = hashlib.sha256(f.read(8 << 20)).hexdigest()
    metadata = {'task': 'arc_challenge', 'split': 'test', 'num_fewshot': 0,
                'chat_template': False, 'target_delimiter': ' ',
                'acc_norm_length': 'choice Unicode characters, excluding delimiter',
                'scoring': args.scoring + ', complete answer text',
                'dataset': str(args.dataset), 'dataset_sha256': sha256(args.dataset),
                'sample_sha256': sha256(args.output / 'sample.json'),
                'seed': args.seed, 'sample_size': args.sample_size,
                'max_cases': args.max_cases, 'backend': args.backend,
                'model': str(args.model), 'model_bytes': stat.st_size,
                'model_mtime_ns': stat.st_mtime_ns,
                'model_first_8MiB_sha256': model_header_hash,
                'model_sha256': model_hash,
                'binary': str(args.binary), 'binary_sha256': sha256(args.binary),
                'runner_sha256': sha256(Path(__file__)),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                    cwd=ROOT, text=True).strip(),
                'started_utc': datetime.now(timezone.utc).isoformat()}
    source = args.dataset.with_name('source.json')
    if source.is_file():
        metadata['dataset_source'] = json.loads(source.read_text())
    previous, previous_seconds = [], 0.0
    if args.resume_from:
        args.resume_from = args.resume_from.resolve()
        previous, previous_seconds = load_previous(args.resume_from, metadata, records)
        metadata['resume_from'] = str(args.resume_from)
        metadata['previous_scores_sha256'] = sha256(args.resume_from / 'scores.jsonl')
        metadata['previous_run_sha256'] = sha256(args.resume_from / 'run.json')
        metadata['previous_completed'] = len(previous)
    save_json(args.output / 'run.json', metadata)
    jobs = []
    for case_index, record in enumerate(records[:args.max_cases]):
        if case_index < len(previous):
            continue
        context = f"Question: {record['question']['stem']}\nAnswer:"
        prefix = tokenize(args.binary, args.model, context)
        for choice_index, choice in enumerate(record['question']['choices']):
            whole = tokenize(args.binary, args.model, context + ' ' + choice['text'])
            if whole[:len(prefix)] != prefix or len(whole) <= len(prefix):
                raise ValueError(f"token boundary mismatch: {record['id']}")
            if len(whole) > 4096:
                raise ValueError('sequence exceeds diagnostic token limit')
            job = {'case': case_index, 'choice': choice_index,
                   'input': whole[:-1], 'target': whole[len(prefix):],
                   'context_length': len(prefix),
                   'teacher_forced': args.scoring == 'teacher-forced',
                   'prefix': prefix,
                   'path': dumps / f'{case_index:04d}-{choice_index}.bin'}
            jobs.append(job)
            if (args.scoring == 'all-rows' and
                    case_index == len(previous) and choice_index == 0):
                # Validate the faster all-row scoring against single-token
                # teacher forcing before accepting benchmark results.
                jobs.append({**job, 'path': dumps / 'parity.bin',
                             'teacher_forced': True, 'parity': True})
    manifest = args.output / 'inputs.tsv'
    with manifest.open('w') as f:
        for job in jobs:
            ids = ','.join(map(str, job['input']))
            if job.get('teacher_forced'):
                ids = ','.join(map(str, job['prefix'])) + '|' + ','.join(map(str, job['target']))
            f.write(f"{ids}\t{job['path']}\n")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('DS4_QWEN4_', 'DS4_ROCM_GRAPH_DUMP')):
            del env[key]
    env.update(DS4_QWEN4_GPU='1', DS4_QWEN4_FT_LIST=str(manifest))
    command = [str(args.binary), '--' + args.backend, '-m', str(args.model),
               '--first-token-test', '--ctx', '8192', '-p', 'ARC benchmark']
    metadata['command'] = command
    save_json(args.output / 'run.json', metadata)
    print(f'Prepared {args.max_cases - len(previous)} new questions from fixed '
          f'{args.sample_size}-case sample ({len(previous)} already completed); '
          f'{len(jobs)} forwards, scoring={args.scoring}', flush=True)
    results, scores = list(previous), {}
    parity = {'required': False, 'passed': None,
              'reason': 'using reference teacher forcing without all-row shortcut'} \
        if args.scoring == 'teacher-forced' else None
    start = time.monotonic()
    elapsed = lambda: previous_seconds + time.monotonic() - start
    process = None
    completed_elapsed = None
    integrity_passed = args.expected_model_sha256 is None
    with (args.output / 'stderr.log').open('w') as stderr, \
            (args.output / 'stdout.log').open('w') as stdout, \
            (args.output / 'scores.jsonl').open('w') as output:
        try:
            for result in previous:
                output.write(json.dumps(result) + '\n')
            output.flush()
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=stderr, text=True, bufsize=1)
            index = 0
            first_logprobs = None
            for line in process.stdout:
                stdout.write(line)
                stdout.flush()
                if not line.startswith('ft_list '):
                    continue
                if index >= len(jobs):
                    raise ValueError('unexpected extra dump')
                job = jobs[index]
                if line.strip().split(' -> ', 1)[-1] != str(job['path']):
                    raise ValueError('unexpected logit dump order')
                token_scores = continuation_logprobs(job['path'], job['input'],
                                                     job['target'], job['context_length'],
                                                     job.get('teacher_forced', False))
                job['path'].unlink()
                if job.get('parity'):
                    error = float(np.max(np.abs(token_scores - first_logprobs)))
                    parity = {'required': True,
                              'max_token_logprob_difference': error, 'tolerance': 0.03,
                              'passed': error <= 0.03}
                    save_json(args.output / 'scoring-parity.json', {
                        **parity, 'id': records[job['case']]['id'], 'choice': job['choice'],
                        'all_row_logprobs': first_logprobs.tolist(),
                        'teacher_forced_logprobs': token_scores.tolist()})
                    if not parity['passed']:
                        raise ValueError(f'prefill/teacher-forcing parity failed: {error}')
                    print(f'PASS scoring parity: max token logprob difference {error:.6g}', flush=True)
                else:
                    if index == 0:
                        first_logprobs = token_scores
                    scores.setdefault(job['case'], []).append(float(token_scores.sum()))
                    record = records[job['case']]
                    if len(scores[job['case']]) == len(record['question']['choices']):
                        result = grade(record, scores[job['case']])
                        result['elapsed_seconds'] = elapsed()
                        results.append(result)
                        output.write(json.dumps(result) + '\n')
                        output.flush()
                        summary = summarize(results, args.max_cases, elapsed(), parity)
                        save_json(args.output / 'summary.json', summary)
                        print(f"{len(results)}/{args.max_cases} {record['id']}: "
                              f"acc={summary['acc']:.1%} acc_norm={summary['acc_norm']:.1%} "
                              f"elapsed={summary['elapsed_seconds']:.1f}s", flush=True)
                index += 1
            if (process.wait() or index != len(jobs) or not parity or
                    (parity.get('required', True) and not parity['passed'])):
                raise RuntimeError(f'incomplete run; see {args.output / "stderr.log"}')
            completed_elapsed = elapsed()
            if args.expected_model_sha256:
                print('Verifying the complete model SHA-256 after inference', flush=True)
                verification_start = time.monotonic()
                metadata['model_final_sha256'] = verify_model(args.model, args.expected_model_sha256)
                metadata['postflight_verification_seconds'] = time.monotonic() - verification_start
                save_json(args.output / 'run.json', metadata)
                integrity_passed = True
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            summary = summarize(results, args.max_cases,
                                completed_elapsed if completed_elapsed is not None else elapsed(), parity)
            summary['success'] = (len(results) == args.max_cases and parity is not None
                                  and (not parity.get('required', True) or parity['passed'])
                                  and process is not None
                                  and process.returncode == 0 and integrity_passed)
            summary['model_integrity_verified_before_and_after'] = \
                integrity_passed if args.expected_model_sha256 else None
            save_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
