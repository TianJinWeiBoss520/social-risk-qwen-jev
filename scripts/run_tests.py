"""Run public unit tests and archived synthetic/mock tests; no cloud/GPU work."""

import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "src"), str(root / "experiments/v1/src"), str(root / "experiments/v1/tests")]
suite = unittest.TestSuite()
for directory in (root / "tests", root / "experiments/v1/tests"):
    suite.addTests(unittest.defaultTestLoader.discover(str(directory), top_level_dir=str(directory)))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
