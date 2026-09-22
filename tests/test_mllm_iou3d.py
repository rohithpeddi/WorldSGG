"""Vendored IoU must equal the detector's compute_iou_3d_obb when that module imports."""
import numpy as np
import pytest

from lib.mllm.data.geometry import obb_to_corners
from lib.mllm.eval.iou3d import compute_iou_3d_obb as vendored


def _cases():
    rs = np.random.RandomState(0)
    for _ in range(50):
        a = obb_to_corners(rs.rand(3) * 2, rs.rand(3) + 0.2, rs.rand() * 3)
        b = obb_to_corners(rs.rand(3) * 2, rs.rand(3) + 0.2, rs.rand() * 3)
        yield a, b


def test_self_iou_is_one():
    c = obb_to_corners([0, 0, 0.5], [2, 1, 1], 0.3)
    assert abs(vendored(c, c) - 1.0) < 1e-9


def test_matches_original():
    orig = pytest.importorskip("lib.detector.monocular3d.evaluation.evaluate_3d")
    for a, b in _cases():
        assert abs(orig.compute_iou_3d_obb(a, b) - vendored(a, b)) < 1e-12
