"""CPU pre-flight for the sandbox (run with WSGG_MLLM_CONFIG=<sandbox>/config.yaml, CUDA hidden).

Loads every split video through the runners' own loader, then builds the Track A payload of every
frame in both modes, which fills the sandbox BEV / detection / lift caches exactly as the runner
would. Any data problem surfaces here instead of after a model load on the GPU.
"""
import logging
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.expanduser("~/CODE/Scene4Cast_mllm"))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

from lib.mllm.core.config_loader import get_path, load_config  # noqa: E402
from lib.mllm.data.worldbbox import WorldBBoxTestSet  # noqa: E402
from lib.mllm.methods.track_a_prompt.runner import TrackAContext, build_payload  # noqa: E402

cfg = load_config()
assert "sandbox" in open(os.environ["WSGG_MLLM_CONFIG"]).read(), "not the sandbox config"
ts = WorldBBoxTestSet(cfg)
print("split:", ts.video_ids, "| bev cache:", get_path(cfg, "worldbbox.caches.bev"))
for vid in ts.video_ids:
    t0 = time.time()
    v = ts.load(vid)
    frs = v.frames
    print(f"\n== {vid}: {len(frs)} frames, pi3 start {v.pi3_start}, "
          f"{sum(f.pi3_index is not None for f in frs)} mapped to pi3, objects {v.video_objects()}")
    for mode in ("predcls", "sgdet"):
        ctx = TrackAContext(v, cfg, mode)
        n_obj = n_obb = n_img = n_chr = 0
        for fr in frs:
            p = build_payload(ctx, fr, 2)
            n_obj += len(p["objects"])
            n_obb += sum(o["obb"] is not None for o in p["objects"])
            n_img += len(p["images"])
            n_chr += len(p["text"])
        ctx.save()
        print(f"  {mode:7s}: {n_obj} ids over {len(frs)} frames ({n_obb} with an OBB), "
              f"{n_img / len(frs):.1f} images/frame, prompt ~{n_chr // len(frs)} chars; bev "
              f"{ctx.bev_img.size} at {ctx.bev_meta['px_per_m']:.0f} px/m")
    print(f"  {time.time() - t0:.0f}s")
for sub in ("bev", "detections", "lifted3d"):
    d = get_path(cfg, f"worldbbox.caches.{sub}")
    print(sub, sorted(os.listdir(d)))
