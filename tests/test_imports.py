"""Regression tests for package imports and the single public implementation."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPackageImports(unittest.TestCase):
    def test_package_does_not_modify_import_path(self):
        # Use a fresh interpreter so an earlier import cannot hide path changes.
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
            "before = sys.path.copy(); "
            "import fastad; "
            "from fastad.fastad import FastAD; "
            "from fastad.benchmark import FastAD as BenchmarkFastAD; "
            "assert sys.path == before, 'Importing fastad modified sys.path'; "
            "assert fastad.FastAD is FastAD is BenchmarkFastAD"
        )
        with tempfile.TemporaryDirectory() as workdir:
            result = subprocess.run(
                [sys.executable, "-B", "-c", code],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
