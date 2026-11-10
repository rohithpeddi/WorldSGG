"""Table 3 re-scores from saved zero-shot predictions (CPU only, local E: drive; no MLLM inference).

For the five Table 3 backbones and three context arms (Caption = ``subtitle``, RAG = ``rag``, and the no-context
control = ``zero_shot``), on the exact Table 3 video sets (``wsg_2d_augmentations_filtered_{predcls,sgdet}``:
184 PredCls / 14 SGDet videos):

  1. ``evaluate()`` from WorldSceneGraphAnnotationTool (the Table 3 scorer, unchanged). Gate: the Caption and RAG
     JSONs reproduce the saved ``predictions/{mode}/{method}_{model}.json`` micro/macro F1 that Table 3 prints.
  2. SGDet under the rule App H.3.1 states (C7): relation predictions for predicted objects that are not in the
     frame's ground truth count as false positives. Same matching as ``docs/paper_audit/probes/zs_sgdet_fp.py``.

The filtered GT folders are only read (``filter_augmented_pkls`` is never called, so they are not regenerated).
Outputs: ``--out``/{predcls,sgdet}/{method}_{model}.json and ``--out``/table3_rescore.json.

Usage (local):  C:/Users/rohit/anaconda3/envs/scene4cast/python.exe tools/rebuttal/zs_table3_rescore.py
"""
import argparse
import contextlib
import glob
import io
import json
import logging
import os
import pickle
import sys

WSG = r"C:\Users\rohit\PycharmProjects\WorldSceneGraphAnnotationTool"
sys.path.insert(0, WSG)
sys.dont_write_bytecode = True
from backend.evaluation.evaluate_relationships import evaluate  # noqa: E402
from backend.evaluation.label_constants import normalise_object_label, normalise_relationship_label  # noqa: E402
from backend.evaluation.load_method_outputs import MethodOutputLoader  # noqa: E402

AG = r"E:\DATA\OPEN\SceneGraphs\action_genome"
M = os.path.join(AG, "mllms")
SAVED = os.path.join(M, "predictions")
GT = {mode: os.path.join(AG, f"wsg_2d_augmentations_filtered_{mode}") for mode in ("predcls", "sgdet")}
PRED_DIR = {"subtitle": "subtitle_all_objects_results", "rag": "rag_all_objects_results", "zero_shot": "zero_shot_results"}
MODELS = ["internvl25_8b", "keyevl15_8b", "minicpmv45", "ovis25_9b", "qwen25vl_7b"]


def rel_sets(pred):
    a = pred.get("attention", "")
    a = {normalise_relationship_label(x) for x in (a if isinstance(a, list) else [a]) if x}
    c = pred.get("contacting", []); c = [c] if isinstance(c, str) else c
    s = pred.get("spatial", []); s = [s] if isinstance(s, str) else s
    return a, {normalise_relationship_label(x) for x in c if x}, {normalise_relationship_label(x) for x in s if x}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def sgdet_stated_rule(method, model):
    """Micro P/R/F1 over all axes, as coded and with extra-object false positives (App H.3.1)."""
    L = MethodOutputLoader(predictions_dir=os.path.join(M, PRED_DIR[method]), mode="sgdet", model_name=model)
    tp = fp = fn = fp_extra = n_extra_obj = 0
    for ap in sorted(glob.glob(os.path.join(GT["sgdet"], "*.pkl"))):
        with open(ap, "rb") as f:
            rec = pickle.load(f)
        vid = rec.get("video_id", os.path.basename(ap).replace(".pkl", ""))
        pv = L.load_video(vid)
        if pv is None:
            continue
        for fk, fd in rec.get("frames", {}).items():
            stem = fk.split("/")[-1].replace(".png", "").replace(".jpg", "")
            pf = pv.get(stem, {})
            gt_objs = fd.get("objects", [])
            names = {normalise_object_label(o.get("class", "")) for o in gt_objs if o.get("class")}
            for o in gt_objs:
                cls = normalise_object_label(o.get("class", ""))
                if not cls:
                    continue
                ps = rel_sets(pf.get(cls, {"attention": "", "contacting": [], "spatial": []}))
                gs = ({normalise_relationship_label(x) for x in o.get("attention", []) if x},
                      {normalise_relationship_label(x) for x in o.get("contacting", []) if x},
                      {normalise_relationship_label(x) for x in o.get("spatial", []) if x})
                for P, G in zip(ps, gs):
                    tp += len(P & G); fp += len(P - G); fn += len(G - P)
            for name, p in pf.items():
                if name not in names:
                    n_extra_obj += 1
                    fp_extra += sum(len(x) for x in rel_sets(p))
    return {"as_coded": prf(tp, fp, fn), "stated_rule": prf(tp, fp + fp_extra, fn),
            "extra_object_predictions": n_extra_obj, "extra_object_relation_fps": fp_extra}


def main(out):
    logging.disable(logging.CRITICAL)
    n_gt = {m: len(glob.glob(os.path.join(GT[m], "*.pkl"))) for m in GT}
    print("GT videos", n_gt)
    table, gate = {}, []
    for mode in ("predcls", "sgdet"):
        os.makedirs(os.path.join(out, mode), exist_ok=True)
        for method in PRED_DIR:
            for model in MODELS:
                with contextlib.redirect_stdout(io.StringIO()):
                    r = evaluate(augmented_dir=GT[mode], predictions_dir=os.path.join(M, PRED_DIR[method]),
                                 mode=mode, model_name=model)
                with open(os.path.join(out, mode, f"{method}_{model}.json"), "w") as f:
                    json.dump(r, f, indent=1, default=str)
                g = r["gt_plus_corrections"]
                row = {"micro_f1": g["micro_avg"]["f1"], "macro_f1": g["macro_avg"]["f1"],
                       "n_videos_evaluated": r.get("n_videos_evaluated"), "n_videos_skipped": r.get("n_videos_skipped"),
                       "n_pairs": g.get("n_pairs")}
                if mode == "sgdet":
                    row["object_detection"] = {k: r["object_detection"][k] for k in ("precision", "recall", "f1")}
                    row.update(sgdet_stated_rule(method, model))
                saved = os.path.join(SAVED, mode, f"{method}_{model}.json")
                if method != "zero_shot" and os.path.exists(saved):
                    s = json.load(open(saved))["gt_plus_corrections"]
                    d = max(abs(s["micro_avg"]["f1"] - row["micro_f1"]), abs(s["macro_avg"]["f1"] - row["macro_f1"]))
                    gate.append((mode, method, model, d))
                table[f"{mode}|{method}|{model}"] = row
                print(f"{mode:7s} {method:9s} {model:14s} muF1={100 * row['micro_f1']:.1f} MF1={100 * row['macro_f1']:.1f}"
                      f" videos={row['n_videos_evaluated']}/{row['n_videos_skipped']}"
                      + (f" | stated-rule muF1={100 * row['stated_rule']['f1']:.1f}"
                         f" objdet F1={100 * row['object_detection']['f1']:.1f}" if mode == "sgdet" else ""), flush=True)
    worst = max(gate, key=lambda x: x[3]) if gate else None
    print("gate: max |F1 diff| vs saved Table 3 JSONs =", worst)
    with open(os.path.join(out, "table3_rescore.json"), "w") as f:
        json.dump({"gt_videos": n_gt, "gate_max_diff": worst, "rows": table}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Users\rohit\PycharmProjects\WorldSGG\docs\rebuttal_rescore\table3")
    main(ap.parse_args().out)
