"""Check probability alignment and paired perplexity statistics without a model."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'run_hybrid_perplexity.py'
spec = importlib.util.spec_from_file_location('hybrid_perplexity', MODULE)
ppl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppl)


class PerplexityStatisticsTests(unittest.TestCase):
    def test_rejects_shifted_target_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scores.csv'
            path.write_text('position,token,logprob,top1\n2,99,-1,99\n')
            document = {'prefix_tokens':2, 'scored_tokens':1, 'tokens':[7,8,9]}
            with self.assertRaises(AssertionError):
                ppl.load_scores(path, document)

    def test_paired_nll_bootstrap_and_token_weighted_perplexity(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            documents = []
            for source in ppl.SOURCES:
                prefixes = [128,128,1536] if source == 'wiki' else [128,128]
                for i, prefix in enumerate(prefixes):
                    documents.append({'name':f'{source}-{i}', 'source':source,
                                      'prefix_tokens':prefix, 'scored_tokens':4,
                                      'tokens':[42]*prefix+[0,1,2,3]})
            for mode in ('float','q8'):
                folder = base/mode
                folder.mkdir()
                (folder/'run.json').write_text('{}')
                (folder/'repeat-controls.json').write_text(json.dumps([
                    {'max_logprob_difference':.1 if mode=='q8' else 0,
                     'top1_equal':mode=='float', 'passed':mode=='float'}]))
                for document in documents:
                    rows = ['position,token,logprob,top1']
                    # Different document losses expose incorrect arithmetic
                    # averaging of perplexities; Q8 adds exactly 0.01 nat.
                    scale = 1 if document['source']=='wiki' else 2
                    for t in range(4):
                        logprob = -(t+1)*scale - (0.01 if mode=='q8' else 0)
                        top1 = 10 if mode=='q8' and t==0 else t
                        rows.append(f"{document['prefix_tokens']+t},{t},{logprob},{top1}")
                    (folder/(document['name']+'.csv')).write_text('\n'.join(rows)+'\n')
            (base/'manifest.json').write_text('{}')
            (base/'calibration.json').write_text('[]')
            with patch.object(ppl, 'digest', return_value='unit-test'):
                ppl.analyze(base, {'documents':documents, 'protocol':{}})
            report = json.loads((base/'summary.json').read_text())
            self.assertFalse(report['repeatability_passed'])
            summary = report['aggregate']
            expected_nll = (3*2.5 + 6*5)/9
            self.assertAlmostEqual(summary['float_ppl'], math.exp(expected_nll))
            self.assertAlmostEqual(summary['relative_ppl_change'], math.expm1(.01))
            self.assertAlmostEqual(summary['delta_avg_nll'], .01)
            for bound in summary['delta_avg_nll_bootstrap_95']:
                self.assertAlmostEqual(bound, .01)
            self.assertEqual(summary['tokens'], 36)
            self.assertEqual(summary['top1_change_rate'], .25)


if __name__ == '__main__':
    unittest.main()
