"""Export per-frame scene-graph data (ground truth + predictions + outcomes) for the supplementary
qualitative figures, so they can be rendered off the server in the paper's style.

Reuses generate_qualitative.build_views, i.e. the metric's own matcher: ``recalled`` holds the
ground-truth (object, predicate) keys a method recalls at R@K, ``said`` what it predicted per object
in the protocol's own reading (one predicate per head with constraint).  Nothing is re-scored here.

    cd <code snapshot with lib/mllm> && python export_sg_json.py --video 12XD3 UG4M2 \
        --out /data3/rohith/ag/runs/qualitative_sg --mode predcls \
        --methods worldwise_pp worldwise w_dsgdetr_pp caption_all rag_all track_a track_b

Writes <out>/<video>/<video>_<mode>_sg.json and copies every annotated raw frame to <out>/<video>/frames/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(os.getcwd()) / "scripts" / "paper_figures"))
import generate_qualitative as GQ  # noqa: E402
from lib.mllm.core.config_loader import get_path, load_config  # noqa: E402
from lib.mllm.data.worldbbox import WorldBBoxTestSet  # noqa: E402


def box(b):
    return None if b is None else [float(x) for x in b]


def export(video_id, args, cfg, test_set):
    specs = [GQ.METHODS[k] for k in args.methods]
    video = test_set.load(video_id)
    cache_dir = Path(args.cache_dir or (Path(get_path(cfg, "outputs.dumps")) / "qualitative" / "cache"))
    views = GQ.build_views(video, specs, args.mode, cfg, cache_dir, args.k, 0.15, args.constraint)
    out_dir = Path(args.out) / video_id
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    frames = []
    for fr in video.frames:
        raw = GQ.raw_frame_path(video, cfg, fr.file)
        if raw.exists() and not (out_dir / "frames" / fr.file).exists():
            shutil.copy2(raw, out_dir / "frames" / fr.file)
        objs = [{"label": o.label, "cls": o.cls, "observed": bool(o.observed), "visible": bool(o.visible),
                 "source": o.source, "bbox_2d": box(o.bbox_2d), "attention": list(o.attention),
                 "spatial": list(o.spatial), "contacting": list(o.contacting)} for o in fr.objects]
        gt_rel = GQ.gt_relations_of(fr)
        per_method = {}
        for spec in specs:
            v = views[spec.key]
            present = fr.file in v.records
            rec = v.recalled.get(fr.file)
            per_method[spec.key] = {
                "present": present,
                "said": v.said.get(fr.file, {}),
                "recalled": sorted([list(k) for k in (rec or set())]),
                "outcome": {l: list(GQ.outcome_for(v, fr.file, l, gt_rel.get(l, []))) for l in gt_rel},
            }
        frames.append({"frame_num": int(fr.frame_num), "file": fr.file,
                       "person_bbox_2d": box(fr.person_bbox_2d), "objects": objs, "methods": per_method,
                       "raw_frame": fr.file if (out_dir / "frames" / fr.file).exists() else None})
    blob = {"video_id": video_id, "mode": args.mode, "k": args.k, "constraint": args.constraint,
            "methods": {s.key: {"label": s.label, "kind": s.kind, "backbone": s.backbone,
                                "available": views[s.key].available, "source": views[s.key].source}
                        for s in specs},
            "frames": frames}
    path = out_dir / f"{video_id}_{args.mode}_sg.json"
    path.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    n_uo = sum(1 for f in frames for o in f["objects"] if not o["observed"])
    print(f"{video_id}: {len(frames)} frames, {n_uo} UO object-frames, methods available "
          f"{[k for k, m in blob['methods'].items() if m['available']]} -> {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", nargs="+", required=True)
    p.add_argument("--mode", default="predcls", choices=("predcls", "sgdet"))
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--constraint", default="with", choices=("with", "no"))
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    test_set = WorldBBoxTestSet(cfg)
    for vid in args.video:
        try:
            export(vid, args, cfg, test_set)
        except Exception as e:  # one bad video must not stop the batch
            print(f"{vid}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
