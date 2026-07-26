from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("/home" + "/jiang", "/mnt" + "/d")


class PortablePathTest(unittest.TestCase):
    def test_active_scripts_do_not_embed_author_private_paths(self) -> None:
        offenders: list[str] = []
        for path in sorted((ROOT / "code").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for private_path in FORBIDDEN:
                if private_path in text:
                    offenders.append(f"{path.name}: {private_path}")

        self.assertEqual(
            offenders,
            [],
            "Active public scripts contain private absolute paths:\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
