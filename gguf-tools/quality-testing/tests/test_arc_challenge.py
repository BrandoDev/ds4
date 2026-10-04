import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1] / 'run_arc_challenge.py'
spec = importlib.util.spec_from_file_location('arc_runner', path)
arc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arc)


class ARCScoringTests(unittest.TestCase):
    def test_complete_model_hash_detects_damage_beyond_header(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.gguf'
            model.write_bytes(b'header' + b'\0' * (9 << 20))
            expected = arc.sha256(model)
            self.assertEqual(arc.verify_model(model, expected.upper()), expected)
            with model.open('r+b') as target:
                target.seek(8 << 20)
                target.write(b'bad weights')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                arc.verify_model(model, expected)

    def test_causal_alignment_and_teacher_forcing(self):
        # Two context tokens, then targets 2 and 3. Rows 1 and 2 predict
        # those targets. Row 0 must never contribute to continuation NLL.
        logits = np.array([[99, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]],
                          dtype='<f4')
        expected = np.array([2 - np.log(np.exp(2) + 3),
                             3 - np.log(np.exp(3) + 3)])
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory) / 'logits.bin'
            logits.tofile(dump)
            np.testing.assert_allclose(arc.continuation_logprobs(
                dump, [0, 1, 2], [2, 3], 2), expected, atol=1e-12)
            logits[1:].tofile(dump)
            np.testing.assert_allclose(arc.continuation_logprobs(
                dump, [0, 1, 2], [2, 3], 2, True), expected, atol=1e-12)

    def test_nonfinite_scores_and_truncated_dumps_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory) / 'logits.bin'
            np.array([[0, np.nan]], dtype='<f4').tofile(dump)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                arc.continuation_logprobs(dump, [0], [1], 1)
            dump.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'shape'):
                arc.continuation_logprobs(dump, [0], [1], 1)

    def test_normalization_uses_unicode_characters_not_tokens_or_bytes(self):
        record = {'id': 'numeric-label', 'answerKey': '2', 'question': {
            'choices': [{'label': '1', 'text': 'é'},
                        {'label': '2', 'text': 'long'}]}}
        result = arc.grade(record, [-2, -4])
        self.assertFalse(result['correct'])
        self.assertTrue(result['correct_norm'])
        self.assertEqual(result['choice_character_lengths'], [1, 4])
        with self.assertRaises(ValueError):
            arc.grade(record, [-2, float('nan')])

    def test_fixed_sample_can_be_extended_without_changing_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / 'test.jsonl'
            dataset.write_text('\n'.join(json.dumps({'id': str(i)})
                                         for i in range(1172)))
            small = arc.select_sample(dataset, 20, 42)
            full = arc.select_sample(dataset, 300, 42)
            self.assertEqual(small, full[:20])
            self.assertEqual(len({r['id'] for r in full}), 300)

    def test_partial_summary_does_not_claim_completion(self):
        summary = arc.summarize([{'correct': True, 'correct_norm': False}],
                                20, 10, None)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['acc'], 1)
        self.assertEqual(summary['acc_norm'], 0)
        low, high = summary['acc_wilson_95']
        self.assertLess(low, 1)
        self.assertAlmostEqual(high, 1)

    def test_resume_rejects_a_different_model(self):
        keys = ('task', 'num_fewshot', 'chat_template', 'target_delimiter',
                'acc_norm_length', 'scoring', 'dataset_sha256', 'sample_sha256',
                'seed', 'sample_size', 'backend', 'model', 'model_bytes',
                'model_mtime_ns', 'model_first_8MiB_sha256', 'binary_sha256')
        metadata = dict.fromkeys(keys, 'same')
        metadata['max_cases'] = 2
        record = {'id': 'q1', 'answerKey': 'A', 'question': {'choices': [
            {'label': 'A', 'text': 'yes'}, {'label': 'B', 'text': 'no'}]}}
        result = arc.grade(record, [-1, -2])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            arc.save_json(directory / 'run.json', metadata)
            arc.save_json(directory / 'summary.json', {
                'success': True, 'completed': 1, 'elapsed_seconds': 10})
            (directory / 'scores.jsonl').write_text(json.dumps(result) + '\n')
            previous, seconds = arc.load_previous(directory, metadata, [record])
            self.assertEqual(previous, [result])
            self.assertEqual(seconds, 10)
            with self.assertRaisesRegex(ValueError, 'model'):
                arc.load_previous(directory, {**metadata, 'model': 'different'}, [record])


if __name__ == '__main__':
    unittest.main()
