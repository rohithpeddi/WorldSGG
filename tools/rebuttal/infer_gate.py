"""INFER gate: does a new dump reproduce the saved dump it corresponds to? (CPU, read-only)

Compares every record of ``--new`` (usually a 2-video ``dump_predictions.py --videos ...`` dump) with the record of
the saved full-split dump that has the same (video_id, frame_t), key by key, over the keys the saved dump has
(the non-3-D fields). Integer / bool arrays must be equal; float arrays must agree to ``--atol``. Also reports the
3-D keys the new dump adds, and how many slots changed visibility. For a WSGG_VISIBILITY_FROM_ANNOTATION=0 dump use
``--expect-vis-change``: then visibility and the predicted distributions may differ, all other fields must match.

    $PY tools/rebuttal/infer_gate.py --new <gate dump> --saved <saved dump> [--atol 1e-4] [--out gate.json]
"""
import argparse
import json
import pickle
import sys

import numpy as np

PRED_KEYS = ("attention_distribution", "spatial_distribution", "contacting_distribution")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--saved", required=True)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--expect-vis-change", action="store_true",
                    help="variant dump (WSGG_VISIBILITY_FROM_ANNOTATION=0): visibility_mask and the three predicted "
                         "distributions may differ (reported under variant_changed_keys); every other key must match")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    new = pickle.load(open(a.new, "rb"))
    saved = pickle.load(open(a.saved, "rb"))
    vids = {r["video_id"] for r in new["records"]}
    ref = {(r["video_id"], int(r["frame_t"])): r for r in saved["records"] if r["video_id"] in vids}
    res = {"new": a.new, "saved": a.saved, "new_meta": new.get("meta"), "saved_meta": saved.get("meta"),
           "videos": sorted(vids), "n_new": len(new["records"]), "n_saved_same_videos": len(ref),
           "missing_in_saved": 0, "key_max_abs_diff": {}, "key_mismatch": {}, "keys_only_in_new": set(),
           "keys_only_in_saved": set(), "visibility_changed_slots": 0, "valid_slots": 0}
    for r in new["records"]:
        s = ref.get((r["video_id"], int(r["frame_t"])))
        if s is None:
            res["missing_in_saved"] += 1
            continue
        res["keys_only_in_new"] |= set(r) - set(s)
        res["keys_only_in_saved"] |= set(s) - set(r)
        vm = np.asarray(r["valid_mask"]).astype(bool)
        res["valid_slots"] += int(vm.sum())
        res["visibility_changed_slots"] += int((np.asarray(r["visibility_mask"]).astype(bool)
                                                 != np.asarray(s["visibility_mask"]).astype(bool))[vm].sum())
        for k, v in s.items():
            if k not in r:
                continue
            x, y = r[k], v
            if isinstance(y, np.ndarray) or isinstance(x, np.ndarray):
                x, y = np.asarray(x), np.asarray(y)
                if x.shape != y.shape:
                    res["key_mismatch"][k] = res["key_mismatch"].get(k, 0) + 1
                    continue
                if np.issubdtype(y.dtype, np.floating) or np.issubdtype(x.dtype, np.floating):
                    d = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.size else 0.0
                    res["key_max_abs_diff"][k] = max(res["key_max_abs_diff"].get(k, 0.0), d)
                    if d > a.atol:
                        res["key_mismatch"][k] = res["key_mismatch"].get(k, 0) + 1
                elif not np.array_equal(x, y):
                    if k == "visibility_mask" and a.expect_vis_change:
                        continue
                    res["key_mismatch"][k] = res["key_mismatch"].get(k, 0) + 1
            elif x != y:
                res["key_mismatch"][k] = res["key_mismatch"].get(k, 0) + 1
    res["keys_only_in_new"] = sorted(res["keys_only_in_new"])
    res["keys_only_in_saved"] = sorted(res["keys_only_in_saved"])
    mismatch = dict(res["key_mismatch"])
    if a.expect_vis_change:
        # variant dump: the model input (visibility) and therefore its outputs may differ; every other field
        # (slots, boxes, pairs, GT labels) must still be identical to the saved default-path dump
        res["variant_changed_keys"] = {k: mismatch.pop(k) for k in PRED_KEYS if k in mismatch}
    res["pass"] = (res["missing_in_saved"] == 0 and not mismatch and not res["keys_only_in_saved"]
                   and res["n_new"] == res["n_saved_same_videos"])
    print(json.dumps({k: v for k, v in res.items() if k not in ("new_meta", "saved_meta")}, indent=1, default=str))
    print("new meta:", json.dumps(res["new_meta"], default=str))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    print("GATE", "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
