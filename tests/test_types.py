"""Checks the package's type hints with mypy. Runs without COMSOL."""
import subprocess
import sys
from pathlib import Path

import pytest

root = Path(__file__).parents[1]


def test_mypy():
    pytest.importorskip('mypy')
    run = subprocess.run([sys.executable, '-m', 'mypy', 'src/mphkit'],
                         cwd=root, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
