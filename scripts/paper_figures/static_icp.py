"""Export the pi3 static background S and run the paper's trimmed ICP (Alg. sup:alg:trimmed_icp)
for every dynamic frame of one video.  Writes <out>/static_icp.npz."""
import sys, time, numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
v, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
S = np.load(f"/data/rohith/ag/ag4D/static_scenes/pi3_static/{v}_10/predictions.npz", allow_pickle=True)
Dz = np.load(f"/data2/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic/{v}_10/predictions.npz", allow_pickle=True)
D = {k: Dz[k] for k in ("local_points", "points", "conf", "camera_poses")}   # load each array once
sp, sc = S["static_points"].astype(np.float32), S["static_colors"]
ok = np.isfinite(sp).all(1); sp, sc = sp[ok], sc[ok]
print("static S points", len(sp), "colors dtype", sc.dtype, sc.max())
tgt = sp[rng.choice(len(sp), min(600_000, len(sp)), replace=False)]
tree = cKDTree(tgt)

def wkabsch(A, B, w):
    w = w / w.sum(); mA, mB = (w[:, None] * A).sum(0), (w[:, None] * B).sum(0)
    H = ((A - mA) * w[:, None]).T @ (B - mB)
    U, _, Vt = np.linalg.svd(H); V = Vt.T
    R = V @ np.diag([1, 1, np.linalg.det(V @ U.T)]) @ U.T
    return R, mB - R @ mA

T_START = time.time()
T = D["camera_poses"].shape[0]
Ticp = np.tile(np.eye(4), (T, 1, 1)); iters = np.zeros(T, int); mse_hist = []; n_src = np.zeros(T, int)
for t in range(T):
    L = D["local_points"][t]; d = L[..., 2]
    edge = (ndimage.maximum_filter(d, 3) - ndimage.minimum_filter(d, 3)) / np.maximum(np.abs(d), 1e-6) > 0.03
    conf = np.where(edge, 0.0, D["conf"][t][..., 0])
    P = D["points"][t].reshape(-1, 3); c = conf.reshape(-1)
    m = np.isfinite(P).all(1) & (c > 0.01)
    idx = np.nonzero(m)[0]; idx = rng.choice(idx, min(40_000, len(idx)), replace=False)
    A, w = P[idx].astype(np.float64), c[idx].astype(np.float64); n_src[t] = len(A)
    Rt, tt = np.eye(3), np.zeros(3); prev = None; hist = []
    for it in range(100):
        dist, nn = tree.query(A, k=1, workers=-1)   # all cores; identical result
        cut = np.percentile(dist, 80); V = dist <= cut
        if V.sum() < 10: break
        R, tr = wkabsch(A[V], tgt[nn[V]].astype(np.float64), w[V])
        Rt, tt = R @ Rt, R @ tt + tr
        A = A @ R.T + tr
        e = float(np.mean(np.sum((A[V] - tgt[nn[V]]) ** 2, 1))); hist.append(e)
        if prev is not None and abs(prev - e) < 1e-5: break
        prev = e
    Ticp[t, :3, :3], Ticp[t, :3, 3] = Rt, tt; iters[t] = len(hist); mse_hist.append(np.array(hist))
    ang = np.degrees(np.arccos(np.clip((np.trace(Rt) - 1) / 2, -1, 1)))
    if t % 5 == 0 or t == T - 1:
        print(f"t={t:2d} iters {len(hist):3d} mse {hist[0]:.2e}->{hist[-1]:.2e} |tau| {np.linalg.norm(tt):.4f} rot {ang:.2f} deg")
keep = rng.choice(len(sp), min(400_000, len(sp)), replace=False)
H = max(len(h) for h in mse_hist); M = np.full((T, H), np.nan)
for t, h in enumerate(mse_hist): M[t, :len(h)] = h
np.savez_compressed(f"{out}/static_icp.npz", static_points=sp[keep], static_colors=sc[keep], static_poses=S["camera_poses"],
                    static_image_view=(S["images"][int(sys.argv[3]) if len(sys.argv) > 3 else 0] * 255).clip(0, 255).astype(np.uint8), T_icp=Ticp, iters=iters, mse=M, n_src=n_src,
                    n_static_total=len(sp))
print("wrote", f"{out}/static_icp.npz", f"(ICP over {T} frames: {time.time() - T_START:.1f} s)")
