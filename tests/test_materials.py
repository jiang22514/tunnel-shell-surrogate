from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from materials import F_TK_C40


class MaterialConstantsTest(unittest.TestCase):
    def test_c40_tensile_strength_matches_the_manuscript(self) -> None:
        self.assertEqual(F_TK_C40, 2.39e6)


if __name__ == "__main__":
    unittest.main()
