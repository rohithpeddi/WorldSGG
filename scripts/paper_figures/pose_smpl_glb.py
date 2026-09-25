"""Recover the per-keyframe posed PromptHMR (SMPL-X / Meshcapade) meshes from the animated,
skinned ``world4d.glb`` of video 00T1E, using numpy only (no glTF library).

Pipeline
--------
1. Parse the GLB container (12-byte header, JSON chunk, BIN chunk) and read accessors through
   their bufferViews (componentType / type / byteStride / normalized).
2. Build the node hierarchy and each node's local TRS; evaluate ``Scene_animation`` at
   keyframe k (sampler output taken at index k when the sampler has exactly the master
   keyframe count, otherwise linear / slerp interpolation on the time axis).
3. Node world matrices -> per-skin joint matrices
   ``J_j = inv(world(meshNode)) @ world(joint_j) @ inverseBind_j``.
4. Morph targets: ``v = POSITION + sum_i w_i * target_i.POSITION`` with the weights coming from
   the animated ``weights`` channel of the mesh node (falls back to ``mesh.weights``).
5. Linear blend skinning with JOINTS_0 / WEIGHTS_0, then the mesh node's world transform, giving
   the posed vertices in the glb scene frame.
6. The AnimatedCamera node's camera-to-world matrix per keyframe and ``cameras[0]`` intrinsics.

Outputs
-------
``outputs/scene_pipeline/00T1E/smpl_posed.npz``
    verts (2, 77, 11307, 3) float32, faces_0 / faces_1 int32, cam_c2w (77, 4, 4), cam_yfov,
    cam_aspect, keyframe_times (77,), plus ``glb_to_floorsim`` (4x4; the floor-preserving yaw+xz
    Kabsch from the mesh centroids to the pi3 person points over the 8 clip views, see
    ``glb_to_floorsim_note``) and the alternative fits ``glb_to_floorsim_{rigid,similarity,yaw_xz}``
    with their per-view median nearest-neighbour errors ``check_nn_*``.  ``verts`` themselves are
    left in the raw glb scene frame (which already has floorsim conventions: metric, y-up, floor y=0).
``outputs/scene_pipeline/panels/s4_smpl_check.png``
    keyframes 0 / 43 / 65: posed mesh vertices (green) with the pi3 person points of that view
    in the floorsim frame.

Run from the repo root:  ``python -m scripts.paper_figures.pose_smpl_glb``
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.paper_figures.scene_common import BUNDLE_DIR, PANEL_DIR, bundle, to_floorsim, cfg, view_ids  # noqa: E402

GLB_PATH = BUNDLE_DIR / "world4d.glb"
OUT_NPZ = BUNDLE_DIR / "smpl_posed.npz"
OUT_PNG = PANEL_DIR / "s4_smpl_check.png"

CHECK_KEYFRAMES = tuple(cfg("views3"))
VIEW_KEYS = tuple(view_ids())
PI3_HW = (672, 378)

# ----------------------------------------------------------------------------- GLB parsing
_CTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class GLB:
    def __init__(self, path: Path):
        raw = path.read_bytes()
        magic, version, length = struct.unpack_from("<III", raw, 0)
        if magic != 0x46546C67:
            raise ValueError(f"{path} is not a GLB (magic {magic:#x})")
        off, self.json, self.bin = 12, None, None
        while off < length:
            clen, ctype = struct.unpack_from("<II", raw, off)
            chunk = raw[off + 8: off + 8 + clen]
            if ctype == 0x4E4F534A:
                self.json = json.loads(chunk.decode("utf-8"))
            elif ctype == 0x004E4942:
                self.bin = chunk
            off += 8 + clen
        if self.json is None or self.bin is None:
            raise ValueError("GLB is missing the JSON or BIN chunk")
        self.oddities: list[str] = []
        self._cache: dict[int, np.ndarray] = {}

    def accessor(self, idx: int) -> np.ndarray:
        if idx in self._cache:
            return self._cache[idx]
        a = self.json["accessors"][idx]
        dt = np.dtype(_CTYPE[a["componentType"]])
        ncomp = _NCOMP[a["type"]]
        count = a["count"]
        if "bufferView" not in a:                       # all zeros per spec
            self.oddities.append(f"accessor {idx} has no bufferView (zeros)")
            out = np.zeros((count, ncomp), dt)
        else:
            bv = self.json["bufferViews"][a["bufferView"]]
            if bv.get("buffer", 0) != 0:
                raise ValueError("only the embedded BIN buffer is supported")
            base = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
            elem = dt.itemsize * ncomp
            stride = bv.get("byteStride", elem)
            if stride == elem:
                out = np.frombuffer(self.bin, dt, count * ncomp, base).reshape(count, ncomp)
            else:
                # interleaved vertex buffer: view as a strided array of rows
                rows = np.frombuffer(self.bin, np.uint8, stride * (count - 1) + elem, base)
                out = np.lib.stride_tricks.as_strided(rows, (count, elem), (stride, 1)).copy()
                out = out.view(dt).reshape(count, ncomp)
        if a.get("sparse"):
            self.oddities.append(f"accessor {idx} uses sparse storage (unsupported, ignored)")
        if a.get("normalized"):
            self.oddities.append(f"accessor {idx} normalized {dt}")
            info = np.iinfo(dt)
            out = out.astype(np.float32) / (info.max if info.min < 0 else info.max)
            if info.min < 0:
                out = np.maximum(out, -1.0)
        if a["type"].startswith("MAT"):                 # column-major in glTF
            n = int(np.sqrt(ncomp))
            out = out.reshape(count, n, n).transpose(0, 2, 1)
        out = np.ascontiguousarray(out)
        # sanity checks against the declared min/max of the accessor
        if "min" in a and out.size and not a["type"].startswith("MAT"):
            lo = np.asarray(a["min"], np.float64)
            if np.any(out.min(0) < lo - 1e-4 * (1 + np.abs(lo))):
                self.oddities.append(f"accessor {idx} data below declared min")
        self._cache[idx] = out
        return out


# ----------------------------------------------------------------------------- math helpers
def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """glTF quaternion (x, y, z, w) -> 3x3 rotation matrix (batched on the leading axes)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def trs(t, q, s) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = quat_to_mat(q) * np.asarray(s, np.float64)[None, :]
    M[:3, 3] = t
    return M


def slerp(q0, q1, u):
    q0 = np.asarray(q0, np.float64); q1 = np.asarray(q1, np.float64)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


# ----------------------------------------------------------------------------- scene evaluation
class Scene:
    def __init__(self, g: GLB):
        self.g = g
        self.js = g.json
        self.nodes = self.js["nodes"]
        self.parent = {}
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.roots = self.js["scenes"][self.js.get("scene", 0)]["nodes"]
        # static local TRS
        self.base = []
        for n in self.nodes:
            self.base.append(dict(
                matrix=np.asarray(n["matrix"], np.float64).reshape(4, 4).T if "matrix" in n else None,
                translation=np.asarray(n.get("translation", [0, 0, 0]), np.float64),
                rotation=np.asarray(n.get("rotation", [0, 0, 0, 1]), np.float64),
                scale=np.asarray(n.get("scale", [1, 1, 1]), np.float64),
            ))
        # animation: (node, path) -> (times, values, interpolation)
        self.channels: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, str]] = {}
        self.times: np.ndarray | None = None
        anims = self.js.get("animations", [])
        if len(anims) != 1:
            g.oddities.append(f"{len(anims)} animations present (using the first)")
        if anims:
            an = anims[0]
            for ch in an["channels"]:
                sm = an["samplers"][ch["sampler"]]
                t_in = g.accessor(sm["input"])[:, 0].astype(np.float64)
                out = g.accessor(sm["output"]).astype(np.float64)
                key = (ch["target"]["node"], ch["target"]["path"])
                if key in self.channels:
                    g.oddities.append(f"duplicate animation channel {key}")
                self.channels[key] = (t_in, out, sm.get("interpolation", "LINEAR"))
                if self.times is None or len(t_in) > len(self.times):
                    self.times = t_in
            for (node, path), (t_in, out, interp) in self.channels.items():
                if interp == "CUBICSPLINE":
                    g.oddities.append(f"CUBICSPLINE sampler on node {node} {path} (evaluated as its keyframe values)")
                if path == "weights":
                    nt = len(self.js["meshes"][self.nodes[node]["mesh"]].get("weights", [])) or \
                        len(self.js["meshes"][self.nodes[node]["mesh"]]["primitives"][0].get("targets", []))
                    if out.shape[0] != len(t_in) * nt:
                        g.oddities.append(f"weights channel of node {node}: {out.shape[0]} values != {len(t_in)} x {nt}")

    # -- sampling ---------------------------------------------------------------------------
    def _sample(self, key, k: int, ncomp_hint=None):
        t_in, out, interp = self.channels[key]
        path = key[1]
        t = self.times[k]
        if interp == "CUBICSPLINE":                    # (in-tangent, value, out-tangent) triples
            out = out.reshape(len(t_in), 3, -1)[:, 1, :]
        if path == "weights":
            out = out.reshape(len(t_in), -1)
        if len(t_in) == len(self.times) and abs(t_in[k] - t) < 1e-6:
            return out[k]
        # generic time interpolation
        j = int(np.searchsorted(t_in, t, side="right") - 1)
        j = max(0, min(j, len(t_in) - 1))
        if j >= len(t_in) - 1 or interp == "STEP":
            return out[j]
        u = (t - t_in[j]) / max(t_in[j + 1] - t_in[j], 1e-12)
        if path == "rotation":
            return slerp(out[j], out[j + 1], u)
        return (1 - u) * out[j] + u * out[j + 1]

    def local_matrix(self, i: int, k: int) -> np.ndarray:
        b = self.base[i]
        animated = [p for p in ("translation", "rotation", "scale") if (i, p) in self.channels]
        if b["matrix"] is not None and not animated:
            return b["matrix"]
        t = self._sample((i, "translation"), k) if (i, "translation") in self.channels else b["translation"]
        q = self._sample((i, "rotation"), k) if (i, "rotation") in self.channels else b["rotation"]
        s = self._sample((i, "scale"), k) if (i, "scale") in self.channels else b["scale"]
        return trs(t, q, s)

    def world_matrices(self, k: int) -> np.ndarray:
        W = np.zeros((len(self.nodes), 4, 4))
        seen = np.zeros(len(self.nodes), bool)

        def rec(i, parent_w):
            W[i] = parent_w @ self.local_matrix(i, k)
            seen[i] = True
            for c in self.nodes[i].get("children", []):
                rec(c, W[i])

        for r in self.roots:
            rec(r, np.eye(4))
        if not seen.all():
            self.g.oddities.append(f"{(~seen).sum()} nodes are not reachable from the scene roots")
            for i in np.where(~seen)[0]:
                W[i] = self.local_matrix(int(i), k)
        return W

    def morph_weights(self, node: int, k: int) -> np.ndarray | None:
        mesh = self.js["meshes"][self.nodes[node]["mesh"]]
        if (node, "weights") in self.channels:
            return self._sample((node, "weights"), k)
        if "weights" in mesh:
            return np.asarray(mesh["weights"], np.float64)
        return None

    # -- posing -------------------------------------------------------------------------------
    def pose_mesh_node(self, node: int, k: int, W: np.ndarray):
        """Posed vertices (N, 3) of a skinned mesh node in the scene frame, plus faces."""
        n = self.nodes[node]
        mesh = self.js["meshes"][n["mesh"]]
        g = self.g
        V_all, F_all, off = [], [], 0
        w_morph = self.morph_weights(node, k)
        for prim in mesh["primitives"]:
            if prim.get("mode", 4) != 4:
                g.oddities.append(f"mesh {n['mesh']} primitive mode {prim.get('mode')} (not TRIANGLES)")
            att = prim["attributes"]
            V = g.accessor(att["POSITION"]).astype(np.float64)
            targets = prim.get("targets", [])
            if targets and w_morph is not None:
                nz = np.where(np.abs(w_morph[:len(targets)]) > 0)[0]
                for i in nz:
                    if "POSITION" in targets[i]:
                        V = V + w_morph[i] * g.accessor(targets[i]["POSITION"]).astype(np.float64)
            if "skin" in n and "JOINTS_0" in att:
                skin = self.js["skins"][n["skin"]]
                joints = skin["joints"]
                ibm = g.accessor(skin["inverseBindMatrices"]).astype(np.float64) if "inverseBindMatrices" in skin \
                    else np.tile(np.eye(4), (len(joints), 1, 1))
                inv_mesh = np.linalg.inv(W[node])
                JM = np.einsum("ab,jbc,jcd->jad", inv_mesh, W[joints], ibm)          # (J, 4, 4)
                J = g.accessor(att["JOINTS_0"]).astype(np.int64)                    # (N, 4)
                Wt = g.accessor(att["WEIGHTS_0"]).astype(np.float64)                # (N, 4)
                ws = Wt.sum(1)
                if np.any(np.abs(ws - 1) > 1e-3):
                    g.oddities.append(f"mesh {n['mesh']}: {int((np.abs(ws-1) > 1e-3).sum())} vertices with skin weights not summing to 1 (renormalised)")
                    Wt = Wt / np.maximum(ws, 1e-8)[:, None]
                if "WEIGHTS_1" in att:
                    g.oddities.append(f"mesh {n['mesh']} has >4 skin influences (WEIGHTS_1 ignored)")
                M = np.einsum("ni,nijk->njk", Wt, JM[J])                             # (N, 4, 4)
                Vh = np.concatenate([V, np.ones((len(V), 1))], 1)
                V = np.einsum("njk,nk->nj", M, Vh)[:, :3]
            Vh = np.concatenate([V, np.ones((len(V), 1))], 1)
            V = (W[node] @ Vh.T).T[:, :3]
            F = g.accessor(prim["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in prim \
                else np.arange(len(V)).reshape(-1, 3)
            V_all.append(V); F_all.append(F + off); off += len(V)
        return np.concatenate(V_all), np.concatenate(F_all)


# ----------------------------------------------------------------------------- verification data
def person_points_floorsim(k: int) -> np.ndarray:
    """pi3 points of clip view k that fall inside the largest SAM2 person component, in floorsim."""
    import matplotlib.image as mimg
    from scipy import ndimage
    b = bundle()
    fid = int(b["sampled_idx"][k])
    m = mimg.imread(BUNDLE_DIR / "masks" / f"{fid:06d}.png")
    if m.ndim == 3:
        m = m[..., 0]
    m = m > 0.5
    H, Wd = PI3_HW
    rr = np.clip((np.arange(H) + 0.5) * m.shape[0] / H, 0, m.shape[0] - 1).astype(int)
    cc = np.clip((np.arange(Wd) + 0.5) * m.shape[1] / Wd, 0, m.shape[1] - 1).astype(int)
    mr = m[rr][:, cc]
    lab, n = ndimage.label(mr)
    if n == 0:
        return np.zeros((0, 3))
    sizes = ndimage.sum(mr, lab, range(1, n + 1))
    keep = lab == (int(np.argmax(sizes)) + 1)
    pix = b[f"pi3_pix_{k}"].astype(np.int64)
    sel = keep.reshape(-1)[pix]
    return to_floorsim(b[f"pi3_points_{k}"][sel])


def kabsch_similarity(P: np.ndarray, Q: np.ndarray, with_scale=True):
    """Similarity Q ~ s R P + t (Umeyama).  Returns 4x4."""
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(Y.T @ X)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (X ** 2).sum() if with_scale else 1.0
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = mq - s * R @ mp
    return T, s, R


def kabsch_yaw_xz(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Floor-preserving fit Q ~ R_y(yaw) P + (tx, 0, tz): 2-D Kabsch in the xz plane (y untouched)."""
    A, B = P[:, [0, 2]], Q[:, [0, 2]]
    ma, mb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    d = np.sign(np.linalg.det(U @ Vt))
    R2 = U @ np.diag([1, d]) @ Vt
    t2 = mb - R2 @ ma
    T = np.eye(4)
    T[0, 0], T[0, 2], T[2, 0], T[2, 2] = R2[0, 0], R2[0, 1], R2[1, 0], R2[1, 1]
    T[0, 3], T[2, 3] = t2
    return T


# ----------------------------------------------------------------------------- main
def main():
    if not GLB_PATH.exists():
        raise SystemExit(f"missing {GLB_PATH}")
    g = GLB(GLB_PATH)
    sc = Scene(g)
    js = g.json
    times = sc.times
    K = len(times)
    mesh_nodes = [i for i, n in enumerate(js["nodes"]) if "mesh" in n]
    cam_nodes = [i for i, n in enumerate(js["nodes"]) if "camera" in n]
    print(f"scene '{js['scenes'][0].get('name')}': {len(js['nodes'])} nodes, {K} keyframes "
          f"({times[0]:.2f}..{times[-1]:.2f} s), mesh nodes {mesh_nodes}, camera nodes {cam_nodes}")
    has_weights = [(i, (i, "weights") in sc.channels) for i in mesh_nodes]
    print("animated morph-weight channel per mesh node:", has_weights)

    verts = np.zeros((len(mesh_nodes), K, 0, 3), np.float32)
    faces = []
    cam_c2w = np.zeros((K, 4, 4))
    V0 = None
    for k in range(K):
        W = sc.world_matrices(k)
        if cam_nodes:
            cam_c2w[k] = W[cam_nodes[0]]
        for mi, node in enumerate(mesh_nodes):
            V, F = sc.pose_mesh_node(node, k, W)
            if V0 is None:
                V0 = V
                verts = np.zeros((len(mesh_nodes), K, len(V), 3), np.float32)
            verts[mi, k] = V.astype(np.float32)
            if k == 0:
                faces.append(F.astype(np.int32))
        if k % 20 == 0 or k == K - 1:
            print(f"  keyframe {k:3d}/{K}: mesh0 centroid {verts[0, k].mean(0).round(3)}, "
                  f"y-range [{verts[0, k][:, 1].min():.3f}, {verts[0, k][:, 1].max():.3f}]")
    cam = js["cameras"][0]["perspective"]
    yfov, aspect = float(cam["yfov"]), float(cam.get("aspectRatio", np.nan))

    # ---- frame comparison with the pi3 person points --------------------------------------
    print("\nframe check (glb scene frame vs floorsim person points)")
    P_mesh, P_pts, rows = [], [], []
    for k in VIEW_KEYS:
        pts = person_points_floorsim(k)
        if len(pts) < 50:
            print(f"  view {k}: only {len(pts)} person points, skipped")
            continue
        cp = pts.mean(0)
        ds = [np.linalg.norm(verts[mi, k].mean(0) - cp) for mi in range(len(mesh_nodes))]
        best = int(np.argmin(ds))
        rows.append((k, best, ds, cp, verts[best, k].mean(0), verts[best, k][:, 1].min(), pts[:, 1].min()))
        P_mesh.append(verts[best, k].mean(0)); P_pts.append(cp)
        print(f"  view {k:2d}: person pts {len(pts):5d} centroid {cp.round(2)} y-min {pts[:,1].min():.2f} | "
              + " | ".join(f"mesh{mi} centroid {verts[mi,k].mean(0).round(2)} feet-y {verts[mi,k][:,1].min():.2f} d={ds[mi]:.2f}" for mi in range(len(mesh_nodes))))
    P_mesh, P_pts = np.asarray(P_mesh), np.asarray(P_pts)
    T_sim, s_fit, R_fit = kabsch_similarity(P_mesh, P_pts, with_scale=True)
    T_rig, _, _ = kabsch_similarity(P_mesh, P_pts, with_scale=False)
    res_id = np.linalg.norm(P_mesh - P_pts, axis=1)
    res_sim = np.linalg.norm((T_sim[:3, :3] @ P_mesh.T).T + T_sim[:3, 3] - P_pts, axis=1)
    res_rig = np.linalg.norm((T_rig[:3, :3] @ P_mesh.T).T + T_rig[:3, 3] - P_pts, axis=1)
    print(f"  centroid residual  identity: mean {res_id.mean():.3f} m (max {res_id.max():.3f})")
    print(f"  centroid residual  rigid   : mean {res_rig.mean():.3f} m (max {res_rig.max():.3f})")
    print(f"  centroid residual  similar.: mean {res_sim.mean():.3f} m (max {res_sim.max():.3f}), scale {s_fit:.4f}")
    print(f"  fitted rotation:\n{np.round(R_fit, 4)}\n  translation {np.round(T_sim[:3, 3], 4)}")
    feet = np.asarray([verts[0, k][:, 1].min() for k in range(K)])
    print(f"  mesh0 feet y over all keyframes: min {feet.min():.3f} max {feet.max():.3f} (floor y=0 in floorsim)")

    # nearest-neighbour check (robust to the visible-surface bias of the pi3 person points):
    # median distance from each person point to the closest mesh vertex, per view
    from scipy.spatial import cKDTree
    T_xz = kabsch_yaw_xz(P_mesh, P_pts)                 # floor-preserving: yaw about y + xz shift
    cands = {"identity": np.eye(4), "yaw_xz": T_xz, "rigid": T_rig, "similarity": T_sim}
    nn = {name: [] for name in cands}
    view_pts = {k: person_points_floorsim(k) for k in VIEW_KEYS}
    for (k, best, *_) in rows:
        for name, M in cands.items():
            Vm = (M[:3, :3] @ verts[best, k].T).T + M[:3, 3]
            nn[name].append(float(np.median(cKDTree(Vm).query(view_pts[k])[0])))
    for name in cands:
        print(f"  median NN dist pts->mesh  {name:10s}: mean {np.mean(nn[name]):.3f} m  per view {np.round(nn[name], 3)}")
    # decision: the axes/scale/floor of the glb frame coincide with floorsim (y-up, feet at y~0, metric,
    # camera height matches), so treat it as the same frame when the identity NN error is already small.
    # Otherwise prefer the floor-preserving yaw+xz fit (feet stay on y=0); fall back to the full rigid /
    # similarity Kabsch only when one of them is clearly (>20 %) better.
    if np.mean(nn["identity"]) < 0.12:
        choice = "identity"
    else:
        choice = "yaw_xz"
        for alt in ("rigid", "similarity"):
            if np.mean(nn[alt]) < 0.8 * np.mean(nn[choice]):
                choice = alt
    glb_to_floorsim = cands[choice]
    yaw_deg = float(np.degrees(np.arctan2(T_xz[2, 0], T_xz[0, 0])))
    note = (f"glb scene frame has floorsim conventions (metric, y-up, floor y=0, camera height {cam_c2w[0, 1, 3]:.2f} m) "
            f"but the PromptHMR camera stays near the origin while the pi3 cameras move, so the body drifts w.r.t. the "
            f"pi3 person points; glb_to_floorsim = {choice} Kabsch on {len(P_mesh)} view centroids "
            f"(median NN pts->mesh {np.mean(nn['identity']):.3f} m -> {np.mean(nn[choice]):.3f} m"
            + (f", yaw {yaw_deg:.1f} deg, xz shift {np.round(T_xz[[0, 2], 3], 2)}" if choice == "yaw_xz" else "")
            + (f", scale {s_fit:.4f}" if choice == "similarity" else "") + ")")
    print("  ->", note)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ, verts=verts, faces_0=faces[0], faces_1=faces[1] if len(faces) > 1 else faces[0],
        cam_c2w=cam_c2w.astype(np.float64), cam_yfov=np.float64(yfov), cam_aspect=np.float64(aspect),
        keyframe_times=times.astype(np.float64),
        glb_to_floorsim=glb_to_floorsim.astype(np.float64), glb_to_floorsim_note=np.str_(note),
        glb_to_floorsim_rigid=T_rig.astype(np.float64), glb_to_floorsim_similarity=T_sim.astype(np.float64),
        glb_to_floorsim_yaw_xz=T_xz.astype(np.float64), check_nn_yaw_xz=np.asarray(nn["yaw_xz"], np.float32),
        mesh_names=np.asarray([js["meshes"][js["nodes"][i]["mesh"]].get("name", "") for i in mesh_nodes]),
        check_view_keys=np.asarray([r[0] for r in rows]), check_best_mesh=np.asarray([r[1] for r in rows]),
        check_person_centroids=P_pts.astype(np.float32), check_mesh_centroids=P_mesh.astype(np.float32),
        check_centroid_offset=(P_pts - P_mesh).astype(np.float32),
        check_nn_identity=np.asarray(nn["identity"], np.float32), check_nn_rigid=np.asarray(nn["rigid"], np.float32),
        check_nn_similarity=np.asarray(nn["similarity"], np.float32),
    )
    print("wrote", OUT_NPZ)

    # ---- verification figure: top row raw glb frame, bottom row after glb_to_floorsim ------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    person_mesh = int(np.bincount([r[1] for r in rows]).argmax())
    other = [mi for mi in range(len(mesh_nodes)) if mi != person_mesh]
    fig = plt.figure(figsize=(12, 8))
    for row_i, (M, lab) in enumerate(((np.eye(4), "raw glb frame"), (glb_to_floorsim, f"after glb_to_floorsim ({choice})"))):
        for j, k in enumerate(CHECK_KEYFRAMES):
            ax = fig.add_subplot(2, len(CHECK_KEYFRAMES), row_i * len(CHECK_KEYFRAMES) + j + 1, projection="3d")
            pts = view_pts[k] if k in view_pts else person_points_floorsim(k)
            V = (M[:3, :3] @ verts[person_mesh, k].T).T + M[:3, 3]
            ax.scatter(V[::4, 0], V[::4, 2], V[::4, 1], s=0.6, c="#57c75a", alpha=0.6, label=f"glb mesh {person_mesh} (posed)")
            ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=1.2, c="#e07a2f", alpha=0.7, label="pi3 person pts (floorsim)")
            A = np.concatenate([pts, V])
            c = A.mean(0); r = max(np.ptp(A, 0).max() / 2, 1.0)
            ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[2] - r, c[2] + r); ax.set_zlim(0, 2 * r)
            ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (up)")
            d = float(np.median(cKDTree(V).query(pts)[0]))
            oth = "; ".join(f"mesh{mi} centroid {verts[mi, k].mean(0).round(1)}" for mi in other)
            ax.set_title(f"k={k} (frame {int(bundle()['sampled_idx'][k]):06d}) {lab}\nmedian NN {d:.2f} m, feet y={V[:, 1].min():.2f}"
                         + (f"\n[{oth} not drawn]" if oth else ""), fontsize=8)
            ax.view_init(elev=18, azim=-62)
            if row_i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=7, markerscale=6)
    fig.suptitle(note, fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PNG, dpi=150)
    print("wrote", OUT_PNG)

    print("\naccessor / structure oddities:")
    for o in sorted(set(g.oddities)):
        print("  -", o)
    if not g.oddities:
        print("  (none)")


if __name__ == "__main__":
    main()
