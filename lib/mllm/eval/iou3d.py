"""
Oriented 3D IoU for floor-parallel OBBs -- verbatim copy of
lib/detector/monocular3d/evaluation/evaluate_3d.py (_polygon_clip, _polygon_area,
_convex_hull_2d, _cross, _corners_to_bottom_face, compute_iou_3d_obb) so the MLLM
evaluator does not import lib.detector.monocular3d (whose __init__ pulls in the
trainer, wandb and torch).  tests/test_mllm_iou3d.py checks equality against the
original whenever that module is importable.
"""
import numpy as np




def _polygon_clip(subject: list, clip: list) -> list:
    """Sutherland-Hodgman polygon clipping. subject and clip are lists of (x, y) tuples."""
    def _inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def _intersect(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = a;  x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = list(subject)
    for i in range(len(clip)):
        if len(output) == 0:
            return []
        a, b = clip[i - 1], clip[i]
        inp = list(output)
        output = []
        for j in range(len(inp)):
            p_cur = inp[j]
            p_prev = inp[j - 1]
            if _inside(p_cur, a, b):
                if not _inside(p_prev, a, b):
                    output.append(_intersect(p_prev, p_cur, a, b))
                output.append(p_cur)
            elif _inside(p_prev, a, b):
                output.append(_intersect(p_prev, p_cur, a, b))
    return output


def _polygon_area(poly: list) -> float:
    """Shoelace formula for area of a polygon given as list of (x,y)."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1]
        area -= poly[j][0] * poly[i][1]
    return abs(area) / 2.0


def _convex_hull_2d(points: list) -> list:
    """Andrew's monotone chain convex hull for 2D points."""
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    lower = []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _corners_to_bottom_face(corners: np.ndarray) -> list:
    """Extract the bottom 4 corners (lowest z) as convex hull for OBB IoU in XY plane."""
    corners = corners.reshape(8, 3)
    z_vals = corners[:, 2]
    z_mid = (z_vals.max() + z_vals.min()) / 2.0
    bottom = corners[z_vals <= z_mid]
    pts = [(float(p[0]), float(p[1])) for p in bottom]
    if len(pts) < 3:
        pts = [(float(p[0]), float(p[1])) for p in corners[:4]]
    return _convex_hull_2d(pts)


def compute_iou_3d_obb(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """Oriented 3D IoU between two sets of 8 corners (8, 3).
    Projects to XY plane for polygon intersection, then multiplies by Z overlap."""
    corners1 = np.asarray(corners1, dtype=np.float64).reshape(8, 3)
    corners2 = np.asarray(corners2, dtype=np.float64).reshape(8, 3)

    # Z overlap
    z_min1, z_max1 = corners1[:, 2].min(), corners1[:, 2].max()
    z_min2, z_max2 = corners2[:, 2].min(), corners2[:, 2].max()
    z_overlap = max(0.0, min(z_max1, z_max2) - max(z_min1, z_min2))
    if z_overlap <= 0:
        return 0.0

    # XY polygon intersection
    poly1 = _corners_to_bottom_face(corners1)
    poly2 = _corners_to_bottom_face(corners2)
    if len(poly1) < 3 or len(poly2) < 3:
        return 0.0

    inter_poly = _polygon_clip(poly1, poly2)
    inter_area = _polygon_area(inter_poly)

    area1 = _polygon_area(poly1)
    area2 = _polygon_area(poly2)

    inter_vol = inter_area * z_overlap
    h1 = z_max1 - z_min1
    h2 = z_max2 - z_min2
    vol1 = area1 * h1
    vol2 = area2 * h2
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0
