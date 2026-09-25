"""Extract a compact, figure-ready bundle of the 4D scene-construction intermediates
of one Action Genome video (default 00T1E) from the CS93371 data roots.

Nothing is recomputed: every array is read from artifacts the annotation pipeline
already wrote. The bundle is small enough to render locally with matplotlib.

Usage (on CS93371)::

    ~/anaconda3/envs/scene4cast/bin/python scripts/paper_figures/extract_scene_pipeline.py \
        --video 00T1E --out /data3/rohith/ag/runs/scene_pipeline/00T1E

Bundle contents (``bundle.npz`` + PNG/JPG copies):

* ``sampled_idx``            raw-frame indices of the adaptively sampled frames
* ``pi3_frame_ids``          raw-frame ids of the 77 pi3 views (clip index k -> file {id+1:06d}.png,
                             the loader off-by-one the annotations were built with)
* ``pi3_points_k``, ``pi3_colors_k``, ``pi3_conf_k``   subsampled world points per selected view k
* ``pi3_all_points/colors``  a confidence-filtered, subsampled union of *all* views (the unaligned world cloud)
* ``pi3_poses``              initial pi3 camera-to-world poses (77, 4, 4)
* ``refined_poses``, ``refined_stems``   ICP-refined poses of the 36 annotated frames
* ``global_floor_sim_{s,R,t}``, ``world_to_final_{origin,A}``  the world -> floor-aligned transforms
* ``floor_verts/faces/colors``  the floor grid mesh
* ``vggt_points/colors``     the VGGT static background cloud
* ``smpl_verts_i``, ``smpl_faces_i``   the PromptHMR SMPL meshes from world4d.glb
* ``obb.json``               per-frame per-object candidates, chosen box, final corrected corners
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

AG = Path("/data/rohith/ag")
AG2 = Path("/data2/rohith/ag")


def _sub(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n, size=min(n, k), replace=False) if n > k else np.arange(n)


def _tolist(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _tolist(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_tolist(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def read_ply_xyzrgb(path: Path):
    """Minimal ASCII/binary PLY reader for x y z (+ r g b) vertex clouds."""
    try:
        import trimesh
        m = trimesh.load(str(path), process=False)
        pts = np.asarray(m.vertices, dtype=np.float32)
        cols = None
        if getattr(m, "colors", None) is not None and len(m.colors):
            cols = np.asarray(m.colors)[:, :3].astype(np.uint8)
        elif hasattr(m, "visual") and getattr(m.visual, "vertex_colors", None) is not None:
            cols = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.uint8)
        return pts, cols
    except Exception as e:  # pragma: no cover
        print("trimesh PLY read failed:", e)
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="00T1E")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=8, help="number of pi3 views to keep per-view clouds for")
    ap.add_argument("--pts-per-view", type=int, default=60000)
    ap.add_argument("--pts-union", type=int, default=400000)
    ap.add_argument("--conf-q", type=float, default=0.3, help="drop points below this confidence quantile")
    args = ap.parse_args()

    v = args.video
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    B: dict = {}

    # ---------------- frame sampling ----------------
    sampled_idx = np.load(AG / "sampled_frames_idx" / f"{v}.npy", allow_pickle=True).astype(int)
    B["sampled_idx"] = sampled_idx
    raw_frames = sorted(os.listdir(AG / "frames" / f"{v}.mp4"))
    B["n_raw_frames"] = np.array(len(raw_frames))
    (out / "frames_raw").mkdir(exist_ok=True)
    (out / "frames_sampled").mkdir(exist_ok=True)
    (out / "frames_static").mkdir(exist_ok=True)
    (out / "frames_annotated").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    # a handful of consecutive raw frames around the first two sampled ones (for the SIFT panel)
    for i in range(min(len(raw_frames), 40)):
        shutil.copy(AG / "frames" / f"{v}.mp4" / raw_frames[i], out / "frames_raw" / raw_frames[i])
    for name in sorted(os.listdir(AG / "sampled_frames_jpg" / f"{v}.mp4")):
        shutil.copy(AG / "sampled_frames_jpg" / f"{v}.mp4" / name, out / "frames_sampled" / name)
    for name in sorted(os.listdir(AG / "ag4D" / "static_frames" / f"{v}.mp4")):
        shutil.copy(AG / "ag4D" / "static_frames" / f"{v}.mp4" / name, out / "frames_static" / name)
    ann_frames = sorted(os.listdir(AG / "frames_annotated" / f"{v}.mp4"))
    for name in ann_frames:
        shutil.copy(AG / "frames_annotated" / f"{v}.mp4" / name, out / "frames_annotated" / name)
    mdir = AG / "segmentation" / "masks" / "combined_1" / f"{v}.mp4"
    for name in sorted(os.listdir(mdir)):
        shutil.copy(mdir / name, out / "masks" / name)
    odir = AG / "segmentation" / "masks" / "overlayed_frames" / f"{v}.mp4"
    if odir.exists():
        (out / "masks_overlay").mkdir(exist_ok=True)
        for name in sorted(os.listdir(odir)):
            shutil.copy(odir / name, out / "masks_overlay" / name)

    # ---------------- pi3 dynamic reconstruction ----------------
    z = np.load(AG2 / "ag4D" / "dynamic_scenes" / "pi3_dynamic" / f"{v}_10" / "predictions.npz", allow_pickle=True)
    T = z["camera_poses"].shape[0]
    B["pi3_poses"] = z["camera_poses"].astype(np.float32)
    # clip index k -> sampled frame id (the annotation-time loader indexed the sorted
    # frame list by the sampled id, i.e. clip k shows file {sampled_idx[k]+1:06d}.png)
    ids = sampled_idx[:T]
    B["pi3_frame_ids"] = ids
    B["pi3_frame_files"] = np.array([f"{i + 1:06d}.png" for i in ids])
    conf = z["conf"][..., 0]
    thr = np.quantile(conf, args.conf_q)
    B["pi3_conf_thr"] = np.array(thr)
    keep_views = np.linspace(0, T - 1, args.views).round().astype(int)
    B["pi3_view_ids"] = keep_views
    union_p, union_c, union_k, union_pix = [], [], [], []
    per_view_budget = args.pts_union // T
    for k in range(T):
        P = z["points"][k].reshape(-1, 3)
        C = (z["images"][k].reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
        F = conf[k].reshape(-1)
        ok = np.isfinite(P).all(1) & (F >= thr)
        idx = np.nonzero(ok)[0]
        if k in keep_views:
            s = idx[_sub(len(idx), args.pts_per_view, rng)]
            B[f"pi3_points_{k}"] = P[s]
            B[f"pi3_colors_{k}"] = C[s]
            B[f"pi3_conf_{k}"] = F[s].astype(np.float16)
            # local (camera-frame) points too, for the "unaligned" per-frame panel
            B[f"pi3_local_{k}"] = z["local_points"][k].reshape(-1, 3)[s]
            B[f"pi3_pix_{k}"] = s.astype(np.int32)   # flat pixel index into the (H, W) pi3 image
            # small image for thumbnails
            from PIL import Image
            Image.fromarray((z["images"][k] * 255).clip(0, 255).astype(np.uint8)).save(out / f"pi3_view_{k:02d}.jpg", quality=88)
        s = idx[_sub(len(idx), per_view_budget, rng)]
        union_p.append(P[s]); union_c.append(C[s]); union_k.append(np.full(len(s), k, np.int16))
        union_pix.append(s.astype(np.int32))
    B["pi3_all_points"] = np.concatenate(union_p)
    B["pi3_all_colors"] = np.concatenate(union_c)
    B["pi3_all_view"] = np.concatenate(union_k)
    B["pi3_all_pix"] = np.concatenate(union_pix)
    B["pi3_image_hw"] = np.array(z["images"].shape[1:3])

    # ---------------- floor / world->final transforms / refined poses ----------------
    obbf = pickle.load(open(AG / "world_annotations" / "bbox_annotations_3d_obb_final" / f"{v}.pkl", "rb"))
    d4 = pickle.load(open(AG / "world_annotations" / "bbox_annotations_4d" / f"{v}.pkl", "rb"))
    g = obbf["global_floor_sim"]
    B["global_floor_sim_s"] = np.array(g["s"]); B["global_floor_sim_R"] = np.asarray(g["R"]); B["global_floor_sim_t"] = np.asarray(g["t"])
    w2f = obbf["world_to_final"]
    B["world_to_final_origin"] = np.asarray(w2f["origin_world"]); B["world_to_final_A"] = np.asarray(w2f["A_world_to_final"])
    B["floor_verts"] = np.asarray(obbf["gv"]); B["floor_faces"] = np.asarray(obbf["gf"]); B["floor_colors"] = np.asarray(obbf["gc"])
    ff = obbf["frames_final"]
    B["refined_poses"] = np.asarray(ff["camera_poses"], dtype=np.float32)
    B["refined_stems"] = np.array(ff["frame_stems"])
    fl = ff["floor"]
    B["floor_final_verts"] = np.asarray(fl["vertices"]); B["floor_final_faces"] = np.asarray(fl["faces"]); B["floor_final_colors"] = np.asarray(fl["colors"])
    B["d4_camera_poses"] = np.asarray(d4["camera_poses"], dtype=np.float32)
    B["d4_frame_stems"] = np.array(d4["frame_stems"])
    pfs = obbf["per_frame_sims"]
    B["per_frame_sim_keys"] = np.array([str(k) for k in sorted(pfs)])
    B["per_frame_sim_s"] = np.array([pfs[k]["s"] for k in sorted(pfs)], dtype=np.float32)
    B["per_frame_sim_R"] = np.array([np.asarray(pfs[k]["R"]) for k in sorted(pfs)], dtype=np.float32)
    B["per_frame_sim_t"] = np.array([np.asarray(pfs[k]["t"]) for k in sorted(pfs)], dtype=np.float32)

    # ---------------- OBB candidates / chosen / final ----------------
    obb_json = {"labels": d4["meta"]["labels"], "frames": {}}
    for fk in sorted(d4["frames"]):
        objs = d4["frames"][fk]["objects"]
        objs = objs if isinstance(objs, list) else list(objs.values())
        rec = []
        for o in objs:
            rec.append({
                "label": o.get("label"),
                "source": o.get("source"),
                "gt_bbox_xyxy": _tolist(o.get("gt_bbox_xyxy")),
                "corners_world": _tolist(o.get("corners_world")),
                "corners_final": _tolist(o.get("corners_final")),
                "aabb_floor_aligned": _tolist(o.get("aabb_floor_aligned")),
                "obb_floor_parallel": _tolist(o.get("obb_floor_parallel")),
                "obb_arbitrary": _tolist(o.get("obb_arbitrary")),
                "candidates": _tolist(o.get("candidates")),
                "world4d_filled": o.get("world4d_filled"),
                "world4d_fill_method": o.get("world4d_fill_method"),
            })
        obb_json["frames"][fk] = rec
    # raw (pre-correction) multi-scale candidates with point counts
    raw = pickle.load(open(AG / "world_annotations" / "bbox_annotations_3d_obb" / f"{v}.pkl", "rb"))
    obb_json["raw_frames"] = {}
    for fk in sorted(raw["frames"]):
        objs = raw["frames"][fk]["objects"]
        objs = objs if isinstance(objs, list) else list(objs.values())
        obb_json["raw_frames"][fk] = [{
            "label": o.get("label"), "gt_bbox_xyxy": _tolist(o.get("gt_bbox_xyxy")),
            "aabb_floor_aligned": _tolist(o.get("aabb_floor_aligned")),
            "multi_scale_candidates": _tolist(o.get("multi_scale_candidates")),
        } for o in objs]
    json.dump(obb_json, open(out / "obb.json", "w"))
    # per-frame box meshes of the raw pipeline (verts/faces/color/label)
    fbm = raw["frame_bbox_meshes"]
    meshes = {str(k): [{"label": m["label"], "color": _tolist(m["color"]), "verts": _tolist(m["verts"]), "faces": _tolist(m["faces"])} for m in fbm[k]] for k in fbm}
    json.dump(meshes, open(out / "frame_bbox_meshes.json", "w"))

    # ---------------- VGGT static background ----------------
    ply = AG / "ag4D" / "static_scenes" / "vggt" / f"{v}.mp4" / "sparse" / "points.ply"
    if ply.exists():
        pts, cols = read_ply_xyzrgb(ply)
        if pts is not None:
            s = _sub(len(pts), args.pts_union, rng)
            B["vggt_points"] = pts[s]
            if cols is not None:
                B["vggt_colors"] = cols[s]
            print("vggt cloud", pts.shape)

    # ---------------- PromptHMR SMPL meshes ----------------
    try:
        import trimesh
        sc = trimesh.load(str(AG / "ag4D" / "human" / f"{v}.mp4" / "world4d.glb"), process=False)
        for i, (name, geom) in enumerate(sc.geometry.items()):
            B[f"smpl_verts_{i}"] = np.asarray(geom.vertices, dtype=np.float32)
            B[f"smpl_faces_{i}"] = np.asarray(geom.faces, dtype=np.int32)
            print("smpl", name, geom.vertices.shape)
        B["smpl_names"] = np.array(list(sc.geometry.keys()))
    except Exception as e:
        print("glb:", e)
    try:
        import joblib
        r = joblib.load(AG / "ag4D" / "human" / f"{v}.mp4" / "results.pkl")
        cam = r.get("camera", {})
        info = {k: (np.asarray(val).shape if hasattr(val, "shape") or isinstance(val, (list, tuple)) else val) for k, val in (cam.items() if isinstance(cam, dict) else [])}
        print("results.pkl camera:", info)
        for k in ("pred_cam_R", "pred_cam_T", "img_focal", "img_center", "R", "T", "focal"):
            if isinstance(cam, dict) and k in cam:
                B[f"hmr_cam_{k}"] = np.asarray(cam[k])
        people = r.get("people", {})
        if isinstance(people, dict):
            for pid, pd in list(people.items())[:2]:
                for k in ("frames", "smpl_world", "pred_smpl_params_world", "cam_smpl", "smpl_cam"):
                    pass
                print("person", pid, list(pd.keys())[:12] if isinstance(pd, dict) else type(pd))
    except Exception as e:
        print("results.pkl:", e)

    np.savez_compressed(out / "bundle.npz", **B)
    tot = sum(os.path.getsize(p) for p in out.rglob("*") if p.is_file())
    print(f"bundle written to {out}  ({tot/1e6:.1f} MB)  keys={len(B)}")


if __name__ == "__main__":
    main()
