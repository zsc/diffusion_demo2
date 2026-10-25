import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.rank_face_candidates import read_key, select_nested, validate_score


class BestOfNTests(unittest.TestCase):
    def test_nested_groups_never_mix_and_ties_use_earliest(self):
        rows = [dict(candidate_id=i, score=s) for i, s in enumerate([10, 40, 40, 20, 90, 20, 95, 95])]
        selected = select_nested(rows, 4, [1, 2, 4])
        self.assertEqual([r['candidate_id'] for r in selected[4]], [1, 6])
        for group in range(2):
            scores = [selected[n][group]['score'] for n in [1, 2, 4]]
            self.assertEqual(scores, sorted(scores))
        with self.assertRaises(ValueError):
            select_nested(rows[:-1], 4, [4])

    def test_invalid_judge_numbers_are_rejected(self):
        good = dict(anatomy=4, naturalness=3, detail=2, coherence=5, issues='minor blur')
        self.assertEqual(validate_score(good)['score'], 68)
        for value in [True, -1, 6, float('nan'), float('inf'), '5', None]:
            with self.assertRaises(ValueError):
                validate_score({**good, 'anatomy': value})
        with self.assertRaises(ValueError):
            validate_score({k: v for k, v in good.items() if k != 'issues'})

    def test_env_file_is_parsed_without_shell_evaluation(self):
        with tempfile.TemporaryDirectory() as root, patch.dict('os.environ', {}, clear=True):
            p = Path(root) / '.env'
            p.write_text("export LLM_API_KEY='literal-$(never-execute)'\n")
            self.assertEqual(read_key(p), 'literal-$(never-execute)')


if __name__ == '__main__':
    unittest.main()
