#!/usr/bin/env python3
"""Paired float/Q8 perplexity with fixed corpora and independent text windows.

Prepare downloaded sources, check batch/sequential alignment and deterministic
repeats, then score identical token windows with one model process at a time.
Bootstrap intervals resample text windows within each source, never tokens.
This is a regression study, not the canonical full WikiText leaderboard score.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'wiki': 'https://huggingface.co/datasets/EleutherAI/wikitext_document_level',
    'shakespeare': 'https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt',
    'pinocchio': 'https://www.gutenberg.org/ebooks/52484',
    'arc': 'https://allenai.org/data/arc',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def tokenize(model, text, scratch):
    scratch.write_text(text)
    out = subprocess.run([str(ROOT/'ds4'), '-m', str(model), '--dump-tokens',
                          '--raw-prompt', '--prompt-file', str(scratch)],
                         check=True, capture_output=True, text=True)
    return json.loads(out.stdout.splitlines()[0])


def prepare(base, model):
    rng = random.Random(20261004)
    data = base/'data'
    scratch = base/'tokenizer-input.txt'
    documents = []

    def add(name, source, text, prefix=128, offset=None, provenance=None):
        tokens = tokenize(model, text, scratch)
        length = prefix + 512
        if len(tokens) < length:
            raise ValueError(f'{name}: insufficient tokens: {len(tokens)}')
        if offset is None:
            offset = rng.randrange(len(tokens)-length+1)
        ids = tokens[offset:offset+length]
        documents.append({'name': name, 'source': source, 'prefix_tokens': prefix,
                          'scored_tokens': 512, 'offset_tokens': offset,
                          'original_tokens': len(tokens), 'tokens': ids,
                          'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                          'tokens_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                          'provenance': provenance})
        print('prepared', name, len(tokens), 'offset', offset, flush=True)

    wiki = json.loads((data/'wikitext-documents.json').read_text())['rows']
    assert all(not r['truncated_cells'] for r in wiki)
    rng.shuffle(wiki)
    long_docs = [r for r in wiki if len(r['row']['page']) > 18000][:4]
    long_ids = {r['row_idx'] for r in long_docs}
    short_docs = [r for r in wiki if r['row_idx'] not in long_ids and len(r['row']['page']) > 6000][:8]
    assert len(short_docs) == 8 and len(long_docs) == 4
    for row in short_docs + long_docs:
        prefix = 1536 if row['row_idx'] in long_ids else 128
        add(f"wiki-{row['row_idx']}-p{prefix}", 'wiki', row['row']['page'], prefix,
            provenance={'row_idx':row['row_idx'], 'title':row['row']['page'].splitlines()[0]})
    for source, filename in [('shakespeare', 'shakespeare.txt'), ('pinocchio', 'pinocchio.txt')]:
        text = (data/filename).read_text(encoding='utf-8-sig')
        if source == 'pinocchio':
            text = text.split('*** START OF THE PROJECT GUTENBERG EBOOK', 1)[1].split('***', 1)[1]
            text = text.split('*** END OF THE PROJECT GUTENBERG EBOOK', 1)[0]
        # Disjoint text partitions avoid overlapping scored spans in one book.
        for i in range(4):
            fragment = text[len(text)*i//4:len(text)*(i+1)//4]
            add(f'{source}-{i}', source, fragment, provenance={'character_partition':i, 'partitions':4})
    sample = json.loads((ROOT/'gguf/benchmarks/arc-challenge/iq3-verified-fast-300/sample.json').read_text())
    for i in range(8):
        records = sample[len(sample)*i//8:len(sample)*(i+1)//8]
        text = '\n\n'.join('Question: '+r['question']['stem']+'\nAnswer: '+next(
            c['text'] for c in r['question']['choices'] if c['label']==r['answerKey']) for r in records)
        add(f'arc-{i}', 'arc', text, provenance={'record_ids':[r['id'] for r in records]})
    manifest = {'seed':20261004, 'documents':documents, 'source_urls':SOURCES,
                'source_sha256':{p.name:digest(p) for p in data.iterdir() if p.is_file()},
                'arc_sample_sha256':digest(ROOT/'gguf/benchmarks/arc-challenge/iq3-verified-fast-300/sample.json'),
                'protocol':{'raw_text':True, 'chat_template':False, 'mtp':False,
                            'reset_between_windows':True, 'score':'natural-log P(next token | fixed preceding tokens)',
                            'bootstrap_unit':'text window, stratified by source and context length; same book windows are not independent books',
                            'canonical_wikitext_score':False}}
    stat = model.stat()
    with model.open('rb') as source:
        header_hash = hashlib.sha256(source.read(8 << 20)).hexdigest()
    manifest['model'] = {'path': str(model), 'bytes': stat.st_size,
                         'mtime_ns': stat.st_mtime_ns, 'first_8MiB_sha256': header_hash}
    manifest['total_scored_tokens'] = sum(d['scored_tokens'] for d in documents)
    write_json(base/'manifest.json', manifest)
    scratch.unlink()


def load_scores(path, doc):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    selected = [r for r in rows if int(r['position']) >= doc['prefix_tokens']]
    assert len(selected) == doc['scored_tokens'], (path, len(selected))
    positions = [int(r['position']) for r in selected]
    assert positions == list(range(doc['prefix_tokens'],len(doc['tokens'])))
    assert [int(r['token']) for r in selected] == doc['tokens'][doc['prefix_tokens']:]
    logprob = np.array([float(r['logprob']) for r in selected])
    top1 = np.array([int(r['top1']) for r in selected])
    assert np.isfinite(logprob).all() and (logprob <= 1e-10).all()
    return logprob, top1


def launch(base, model, mode, cases, label, teacher_batch=1, extra_env=None, compact=True):
    output = base/label
    output.mkdir(exist_ok=True)
    inputs = output/'inputs.tsv'
    suffix = '.csv' if compact else '.bin'
    inputs.write_text(''.join(ids+'\t'+str(output/(name+suffix))+'\n' for name,ids in cases))
    env = {k:v for k,v in os.environ.items() if not k.startswith('DS4_')}
    overrides = {'DS4_QWEN4_FT_TEACHER_BATCH':str(teacher_batch), 'DS4_QWEN4_GPU':'1', 'DS4_QWEN4_FT_LIST':str(inputs), 'DS4_QWEN4_FT_SCORES':'1'}
    if mode == 'q8': overrides['DS4_QWEN4_CPU_Q8'] = '1'
    overrides.update(extra_env or {})
    if not compact: overrides.pop('DS4_QWEN4_FT_SCORES', None)
    command = [str(ROOT/'ds4'),'--rocm','-m',str(model),'--first-token-test',
               '--gpu-memory-budget','14GB','--gpu-expert-layers','4',
               '--ctx','4096','--threads','16','-p','Perplexity regression']
    meta = {'command':command,'environment':overrides,'binary_sha256':digest(ROOT/'ds4'),
            'peak_vram_bytes':0,'peak_rss_kib':0,'major_faults':0,'completed_windows':0}
    start = time.monotonic()
    with (output/'stdout.txt').open('w') as out, (output/'stderr.log').open('w') as err:
        process = subprocess.Popen(command,cwd=ROOT,env=env|overrides,stdout=out,stderr=err)
        try:
            while process.poll() is None:
                for path in Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_used'):
                    meta['peak_vram_bytes'] = max(meta['peak_vram_bytes'],int(path.read_text()))
                try:
                    status = Path(f'/proc/{process.pid}/status').read_text()
                    rss = re.search(r'^VmRSS:\s+(\d+)',status,re.M)
                    if rss:meta['peak_rss_kib'] = max(meta['peak_rss_kib'],int(rss[1]))
                    stat = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
                    meta['major_faults'] = max(meta['major_faults'],int(stat[9]))
                except FileNotFoundError:
                    pass
                count = len((output/'stdout.txt').read_text().splitlines())
                if count != meta['completed_windows']:
                    meta['completed_windows']=count
                    print(label, count, '/', len(cases), flush=True)
                meta['elapsed_seconds']=time.monotonic()-start
                write_json(output/'run.json',meta)
                if meta['elapsed_seconds']>1800:raise TimeoutError(label)
                time.sleep(.5)
        finally:
            if process.poll() is None:process.kill();process.wait()
    meta['completed_windows'] = len((output/'stdout.txt').read_text().splitlines())
    meta['exit_code']=process.returncode
    meta['elapsed_seconds']=time.monotonic()-start
    meta['placement']=[line for line in (output/'stderr.log').read_text().splitlines() if 'resident placement:' in line]
    write_json(output/'run.json',meta)
    assert process.returncode == 0, label
    assert meta['peak_vram_bytes'] < 14*2**30, meta
    return output


def calibration(base, model, manifest):
    docs=[]
    for source in SOURCES:
        original = next(d for d in manifest['documents'] if d['source']==source and d['prefix_tokens']==128)
        doc = dict(original, name=source+'-calibration',scored_tokens=32,tokens=original['tokens'][:160])
        docs.append(doc)
    result = []
    cases = []
    for doc in docs:
        ids = doc['tokens']; p = doc['prefix_tokens']
        sequence = ','.join(map(str, ids[:p])) + '|' + ','.join(map(str, ids[p:]))
        cases.append((doc['name'], sequence))
        cases.append((doc['name'] + '-repeat', sequence))
    for mode in ('float', 'q8'):
        reference = launch(base, model, mode, cases, 'calibration-' + mode + '-t1', 1)
        batched = launch(base, model, mode, cases, 'calibration-' + mode + '-t8', 8)
        for doc in docs:
            teacher, teacher_top = load_scores(reference/(doc['name']+'.csv'),doc)
            batch, top = load_scores(batched/(doc['name']+'.csv'),doc)
            repeat, repeat_top = load_scores(batched/(doc['name']+'-repeat.csv'),doc)
            teacher_repeat, teacher_repeat_top = load_scores(reference/(doc['name']+'-repeat.csv'),doc)
            delta = np.abs(batch-teacher)
            item = {'mode':mode, 'source':doc['source'], 'target_batch':8,
                    'max_logprob_difference':float(delta.max()),
                    'mean_logprob_difference':float(delta.mean()),
                    'top1_equal':bool(np.array_equal(top,teacher_top)),
                    'repeat_max_difference':float(max(np.abs(batch-repeat).max(), np.abs(teacher-teacher_repeat).max())),
                    'repeat_top1_equal':bool(np.array_equal(top,repeat_top) and np.array_equal(teacher_top,teacher_repeat_top))}
            result.append(item)
            print('calibration', item, flush=True)
            item['batch_parity_passed'] = bool(delta.max() < .003 and delta.mean() < .0005)
            assert item['repeat_max_difference']<1e-6 and item['repeat_top1_equal'],item
        write_json(base/'calibration.json',result)
    return result


def analyze(base, manifest):
    documents=[];deltas=[];outliers=[]
    for doc in manifest['documents']:
        a,ta=load_scores(base/'float'/(doc['name']+'.csv'),doc)
        b,tb=load_scores(base/'q8'/(doc['name']+'.csv'),doc)
        delta=a-b # positive = Q8 has worse ground-truth NLL
        item={k:doc[k] for k in ('name','source','prefix_tokens','scored_tokens')}
        item.update({'float_nll':float(-a.sum()),'q8_nll':float(-b.sum()),
                     'float_ppl':float(np.exp(-a.mean())),'q8_ppl':float(np.exp(-b.mean())),
                     'delta_avg_nll':float(delta.mean()),'relative_ppl_change':float(np.expm1(delta.mean())),
                     'top1_changed_tokens':int((ta!=tb).sum()),
                     'q8_better_tokens':int((delta<0).sum()),'q8_worse_tokens':int((delta>0).sum())})
        documents.append(item);deltas.append(delta)
        for j in np.argsort(np.abs(delta))[-5:]:
            outliers.append({'document':doc['name'],'position':int(j+doc['prefix_tokens']),
                             'target_token':doc['tokens'][j+doc['prefix_tokens']],
                             'float_logprob':float(a[j]),'q8_logprob':float(b[j]),'delta_nll':float(delta[j])})
    rng=np.random.default_rng(20261004)

    def aggregate(items):
        n=sum(x['scored_tokens'] for x in items)
        fa=sum(x['float_nll'] for x in items)/n
        qb=sum(x['q8_nll'] for x in items)/n
        bootstrap=np.zeros(20000)
        # Stratification fixes domain mixture and prevents resampling individual
        # correlated tokens as if they were independent observations.
        strata = sorted({(d['source'], d['prefix_tokens']) for d in items})
        for source, prefix in strata:
            group = [d for d in items if d['source'] == source and d['prefix_tokens'] == prefix]
            values=np.array([d['delta_avg_nll'] for d in group])
            bootstrap+=values[rng.integers(0,len(values),(20000,len(values)))].mean(axis=1)*len(group)/len(items)
        bounds=np.quantile(bootstrap,[.025,.975])
        return {'windows':len(items),'tokens':n,'float_avg_nll':fa,'q8_avg_nll':qb,
                'float_ppl':math.exp(fa),'q8_ppl':math.exp(qb),'delta_avg_nll':qb-fa,
                'relative_ppl_change':math.expm1(qb-fa),
                'delta_avg_nll_bootstrap_95':bounds.tolist(),
                'relative_ppl_change_bootstrap_95':np.expm1(bounds).tolist(),
                'top1_change_rate':sum(d['top1_changed_tokens'] for d in items)/n,
                'windows_q8_worse':sum(d['delta_avg_nll']>0 for d in items)}
    all_delta=np.concatenate(deltas)
    summary={'completed':True,'protocol':manifest['protocol'],'documents':documents,
             'aggregate':aggregate(documents),
             'by_source':{source:aggregate([d for d in documents if d['source']==source]) for source in SOURCES},
             'by_context':{str(p):aggregate([d for d in documents if d['prefix_tokens']==p]) for p in (128,1536)},
             'token_delta_nll_quantiles':dict(zip(['min','p01','p05','p50','p95','p99','max'],np.quantile(all_delta,[0,.01,.05,.5,.95,.99,1]).tolist())),
             'largest_token_changes':sorted(outliers,key=lambda x:abs(x['delta_nll']),reverse=True)[:20],
             'calibration':json.loads((base/'calibration.json').read_text()),
             'manifest_sha256':digest(base/'manifest.json'),
             'source_sha256':digest(ROOT/'ds4.c'),'binary_sha256':digest(ROOT/'ds4'),
             'validation_script_sha256':digest(Path(__file__)),
             'runs':{mode:json.loads((base/mode/'run.json').read_text()) for mode in ('float','q8')},
             'repeat_controls':{mode:json.loads((base/mode/'repeat-controls.json').read_text()) for mode in ('float','q8')},
             'limitations':['Intervals describe these sampled windows; book windows share an author and text.',
                            'Raw-text perplexity does not establish unchanged chat/task accuracy.',
                            'This is a sampled WikiText regression, not a canonical published full-test perplexity.',
                            'Float and Q8 both use the same IQ3 weights; no FP16 weight-quality baseline.'],
             'production_default':'float',
             'repeatability_passed':all(c.get('passed',c['max_logprob_difference']<1e-6 and c['top1_equal'])
                                        for mode in ('float','q8')
                                        for c in json.loads((base/mode/'repeat-controls.json').read_text())),
             'interpretation':'Paired single-pass estimates. Failed repeat controls must be investigated; bootstrap covers sampled windows only.'}
    summary['model'] = manifest.get('model')
    summary['scope_complete'] = True
    summary['quality_validation_passed'] = (summary['repeatability_passed'] and
        all(c.get('batch_parity_passed', False) for c in summary['calibration']))
    for field, filename in [('reset_validation','reset-validation.json'),
                            ('environment','environment.json'),
                            ('export_validation','export-validation.json')]:
        if (base/filename).exists():
            summary[field] = json.loads((base/filename).read_text())
    if not summary['repeatability_passed']:
        summary['limitations'].append('Q8 repeatability failed; bootstrap intervals exclude execution variability.')
    write_json(base/'summary.json',summary)
    print(json.dumps(summary['aggregate'],indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'gguf/benchmarks/hybrid-cpu-q8-perplexity')
    parser.add_argument('--model',type=Path,default=ROOT/'gguf/Qwen3.8-Flash-Next-IQ3.gguf')
    parser.add_argument('--stage',choices=['prepare','calibrate','score','analyze','all'],default='all')
    args=parser.parse_args();base=args.output.resolve();model=args.model.resolve()
    base.mkdir(parents=True,exist_ok=True)
    if args.stage in ('prepare','all'):prepare(base,model)
    manifest=json.loads((base/'manifest.json').read_text())
    if args.stage in ('calibrate','all'):calibration(base,model,manifest)
    if args.stage in ('score','all'):
        checks = json.loads((base/'calibration.json').read_text())
        assert len(checks) == 8
        assert all(c['repeat_max_difference'] < 1e-6 and c['repeat_top1_equal'] for c in checks)
        teacher_batch = 8 if all(c['batch_parity_passed'] for c in checks) else 1
        manifest['protocol']['target_microbatch_tokens'] = teacher_batch
        manifest['protocol']['batch_parity_passed'] = teacher_batch == 8
        write_json(base/'manifest.json', manifest)
        docs=manifest['documents']
        cases = [(d['name'], ','.join(map(str,d['tokens'][:d['prefix_tokens']])) + '|'
                  + ','.join(map(str,d['tokens'][d['prefix_tokens']:]))) for d in docs]
        controls = [next(d for d in docs if d['prefix_tokens'] == prefix) for prefix in (128,1536)]
        for doc in controls:
            sequence = ','.join(map(str,doc['tokens'][:doc['prefix_tokens']])) + '|' + ','.join(map(str,doc['tokens'][doc['prefix_tokens']:]))
            cases.append((doc['name']+'-repeat',sequence))
        for mode in ('float','q8'):
            launch(base,model,mode,cases,mode,teacher_batch)
            repeats = []
            for doc in controls:
                first, top1 = load_scores(base/mode/(doc['name']+'.csv'),doc)
                repeat, repeat_top1 = load_scores(base/mode/(doc['name']+'-repeat.csv'),doc)
                item = {'document':doc['name'],'prefix_tokens':doc['prefix_tokens'],
                        'max_logprob_difference':float(np.abs(first-repeat).max()),
                        'top1_equal':bool(np.array_equal(top1,repeat_top1))}
                item['passed'] = bool(item['max_logprob_difference'] < 1e-6 and item['top1_equal'])
                repeats.append(item)
            write_json(base/mode/'repeat-controls.json',repeats)
    if args.stage in ('analyze','all'):analyze(base,manifest)


if __name__=='__main__':main()
