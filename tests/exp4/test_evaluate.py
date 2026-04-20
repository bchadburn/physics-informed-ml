import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from core.metrics import optimality_gap


def test_optimality_gap_negative_when_better():
    gap = optimality_gap(0.9, 1.0)
    assert gap == pytest.approx(-0.1)
