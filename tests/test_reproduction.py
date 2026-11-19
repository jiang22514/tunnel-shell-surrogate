import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTest(unittest.TestCase):
    def test_closed_ring_numbers_agree_with_the_printed_values(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        run = subprocess.run([sys.executable, str(ROOT/'code'/'reproduce_tables.py'), '--family', 'closed_ring'],
                             capture_output=True, text=True, env=env)
        self.assertEqual(run.returncode, 0, run.stdout[-3000:] + run.stderr[-3000:])
        self.assertIn('All recomputed numbers agree', run.stdout)


if __name__ == '__main__':
    unittest.main()
