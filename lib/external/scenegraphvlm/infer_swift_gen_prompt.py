#!/usr/bin/env python3
"""
Real-time GEN inference via ms-swift inference engines (vLLM or Transformers) on Swift JSONL.

Frame chaining logic:
- frame t=0: uses original user prompt from jsonl.
- frame t>=1: replaces "Previous frame ..." block with model prediction from frame t-1
  (or GT previous assistant if --prev-source gt).

Input format (Swift jsonl):
  {"messages": [{"role":"user","content":"<image>\n..."}, {"role":"assistant", ...}], "images": ["..."]}

Output: {output_dir}/{run_name}.jsonl with fields compatible with eval tooling.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

try:
    from swift.model import get_model_processor, get_processor
    from swift.template import get_template
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine, VllmEngine
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "ms-swift is required. Install with e.g.:\n"
        "  pip install 'ms-swift[llm]' -U\n"
        "  pip install vllm   # for --infer-backend vllm\n"
        f"Import error: {e}"
    ) from e

PREV_HEADER_LEGACY = "Previous frame scene graph (TOON):\n"
PREV_HEADER_TAGS = "Previous frame ground-truth scene graph (TOON), for reference:\n"
PREV_HEADERS = (PREV_HEADER_LEGACY, PREV_HEADER_TAGS)

_NOW_MARKERS = (
    "\n\n\nNow, generate the complete scene graph for the provided image. Write your response only between <answer> and </answer> tags.",
    "\n\nNow, generate the complete scene graph for the provided image. Write your response only between <answer> and </answer> tags.",
    "\n\n\nNow, generate the complete scene graph for the provided image. Write your response only between <answer> and </answer> tags.\n",
    "\n\nNow, generate the complete scene graph for the provided image. Write your response only between <answer> and </answer> tags.\n",
    "\n\n\nNow, generate the complete scene graph for the provided image. Wrap your scene graph in <answer>...</answer> tags.",
    "\n\nNow, generate the complete scene graph for the provided image. Wrap your scene graph in <answer>...</answer> tags.",
    "\n\n\nNow, generate the complete scene graph for the provided image. Wrap your scene graph in <answer>...</answer> tags.\n",
    "\n\nNow, generate the complete scene graph for the provided image. Wrap your scene graph in <answer>...</answer> tags.\n",
    "\n\n\nNow, generate the complete scene graph for the provided image:\n",
    "\n\n\nNow, generate the complete scene graph for the provided image:",
    "\n\nNow, generate the complete scene graph for the provided image:\n",
    "\n\nNow, generate the complete scene graph for the provided image:",
)

_THINK_END = "</think>"
_SG_OBJ_HDR = re.compile(r"^\s*obj\[\d+\]\{id,name", re.MULTILINE)
_SG_REL_HDR = re.compile(r"^\s*rel\[\d+\]\{subj,pred,obj\}", re.MULTILINE)
_SG_REL_PAIRS_HDR = re.compile(r"^\s*rel_pairs\[\d+\]\{subj,attention", re.MULTILINE)


def _find_now_suffix_start_in_rest(rest: str) -> int:
    for marker in _NOW_MARKERS:
        idx = rest.find(marker)
        if idx != -1:
            return idx
    return -1


def strip_thinking_suffix(text: str) -> str:
    s = text.strip()
    while _THINK_END in s:
        idx = s.find(_THINK_END)
        prefix = s[:idx]
        suffix = s[idx + len(_THINK_END) :].lstrip()
        if _SG_OBJ_HDR.search(prefix) or _SG_REL_HDR.search(prefix) or _SG_REL_PAIRS_HDR.search(prefix):
            s = (prefix + "\n" + suffix).strip() if suffix else prefix.strip()
            break
        if not suffix:
            return prefix.strip() if prefix.strip() else ""
        s = suffix
    return s


def truncate_first_answer_block(text: str) -> str:
    s = text.strip()
    k = s.find("</answer>")
    if k != -1:
        return s[: k + len("</answer>")].strip()
    if "<answer>" in s:
        return s.rstrip() + "\n</answer>\n"
    return s


def finalize_vl_prediction(raw: str) -> str:
    return truncate_first_answer_block(strip_thinking_suffix(raw))


def has_obj_header(text: str) -> bool:
    return bool(_SG_OBJ_HDR.search(text or ""))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_user_text(sample: Dict[str, Any]) -> str:
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    raise ValueError("No user message in sample")


def set_user_text(sample: Dict[str, Any], new_text: str) -> None:
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            msg["content"] = new_text
            return
    raise ValueError("No user message in sample")


def get_assistant_text(sample: Dict[str, Any]) -> str:
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            return (msg.get("content") or "").strip()
    raise ValueError("No assistant message in sample")


def resolve_image_path(sample: Dict[str, Any], images_base: Optional[str]) -> str:
    imgs = sample.get("images") or []
    if not imgs:
        raise ValueError("No images in sample")
    p = imgs[0]
    if os.path.isabs(p):
        return p
    if not images_base:
        return p
    return os.path.join(images_base, p)


def parse_video_and_frame(sample: Dict[str, Any], images_base: Optional[str]) -> Tuple[str, int]:
    p = Path(resolve_image_path(sample, images_base))
    vid = p.parent.name
    try:
        t = int(p.stem)
    except ValueError:
        t = 0
    return vid, t


def parse_torch_dtype(name: str) -> torch.dtype:
    d = getattr(torch, name, None)
    if d is None or not isinstance(d, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {name}")
    return d


def build_prev_frame_template(samples: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], bool]:
    for sample in samples:
        user = get_user_text(sample)
        for header in PREV_HEADERS:
            if header not in user:
                continue
            before, rest = user.split(header, 1)
            idx = _find_now_suffix_start_in_rest(rest)
            if idx == -1:
                continue
            prefix = before + header
            suffix = rest[idx:]
            wrap = header == PREV_HEADER_TAGS
            return prefix, suffix, wrap
    return None, None, False


def build_prompt_with_prev(prefix: str, suffix: str, prev_prediction: str, wrap_prev_in_answer: bool) -> str:
    prev = (prev_prediction or "").strip()
    if wrap_prev_in_answer and prev and not prev.lstrip().lower().startswith("<answer"):
        prev = f"<answer>\n{prev}\n</answer>"
    return prefix + prev + suffix


def build_swift_engine(
    *,
    model_id_or_path: str,
    infer_backend: str,
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    template_type: Optional[str],
    enable_thinking: bool,
    response_prefix: Optional[str],
    torch_dtype: torch.dtype,
) -> Union[VllmEngine, TransformersEngine]:
    if infer_backend == "vllm":
        processor = get_processor(
            model_id_or_path=model_id_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        tmpl = get_template(
            processor,
            template_type=template_type,
            enable_thinking=enable_thinking,
            response_prefix=response_prefix,
        )
        return VllmEngine(
            model_id_or_path,
            template=tmpl,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 1},
        )

    if infer_backend == "transformers":
        model, processor = get_model_processor(
            model_id_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        tmpl = get_template(
            processor,
            template_type=template_type,
            enable_thinking=enable_thinking,
            response_prefix=response_prefix,
        )
        return TransformersEngine(model, template=tmpl, max_batch_size=max(1, batch_size))

    raise ValueError(f"Unknown infer_backend: {infer_backend}")


def _error_record(
    sample: Dict[str, Any],
    *,
    model_path: str,
    run_name: str,
    dataset_tag: str,
    checkpoint_step: str,
    test_jsonl: Path,
    err: str,
    img_path: str,
    prev_source: str,
) -> Dict[str, Any]:
    try:
        gt = get_assistant_text(sample)
    except Exception:
        gt = ""
    try:
        user_prompt = get_user_text(sample)
    except Exception:
        user_prompt = ""
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": ""},
        ],
        "images": sample.get("images"),
        "content": gt,
        "predict": "",
        "model_name": run_name,
        "gen_time_sec": None,
        "predict_error": err,
    }


def _prev_chain_value(prev_source: str, sample: Dict[str, Any], pred_clean: str) -> str:
    if prev_source == "gt":
        try:
            return finalize_vl_prediction(get_assistant_text(sample))
        except Exception:
            return (pred_clean or "").strip()
    return (pred_clean or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swift real-time GEN inference (t -> t+1 previous-frame chaining) on Swift JSONL.",
    )
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    parser.add_argument("--test-jsonl", required=True, type=Path, help="Swift jsonl with messages/images")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--run-name", required=True, help="Output basename: {run_name}.jsonl")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per wave")
    parser.add_argument("--infer-backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=8192, help="vLLM only")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45, help="vLLM only; default: 0.45")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM only")
    parser.add_argument("--images-base", default="", help="Prefix for relative image paths")
    parser.add_argument("--dataset-tag", default="swift_sgg", help="dataset_name field")
    parser.add_argument("--checkpoint-step", default="", help="metadata checkpoint_step")
    parser.add_argument("--template-type", default="", help="Optional swift template_type")
    parser.add_argument("--response-prefix", default="", help="Optional template response_prefix")
    parser.add_argument(
        "--prev-source",
        choices=("model", "gt"),
        default="model",
        help="Source for previous-frame block: model prediction (real-time) or gt assistant",
    )
    parser.add_argument(
        "--auto-obj-prefix-fallback",
        action="store_true",
        help="If output has no obj[...] header, retry sample with response_prefix='obj['.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", help="torch dtype name")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode in template")
    parser.add_argument("--no-stop-on-answer-close", action="store_true")
    parser.add_argument("--stop-sequence", action="append", default=None, metavar="STR")
    parser.add_argument("--force", action="store_true", help="Overwrite output file")
    args = parser.parse_args()

    images_base = args.images_base or None
    template_type = args.template_type.strip() or None
    response_prefix = args.response_prefix.strip() or None
    ckpt_step = args.checkpoint_step.strip() or "inference"
    torch_dtype = parse_torch_dtype(args.torch_dtype.strip())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.run_name}.jsonl"
    if out_path.exists() and not args.force:
        print(f"[skip] exists: {out_path} (use --force)", file=sys.stderr)
        sys.exit(0)

    gpu_mem = args.gpu_memory_utilization

    enable_thinking = bool(args.enable_thinking)
    print(f"[init] model={args.model!r} backend={args.infer_backend} thinking={enable_thinking} prev={args.prev_source}")
    engine = build_swift_engine(
        model_id_or_path=args.model,
        infer_backend=args.infer_backend,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=args.tensor_parallel_size,
        template_type=template_type,
        enable_thinking=enable_thinking,
        response_prefix=response_prefix,
        torch_dtype=torch_dtype,
    )

    fallback_engine: Optional[Union[VllmEngine, TransformersEngine]] = None
    if args.auto_obj_prefix_fallback and (response_prefix or "") != "obj[":
        print("[init] auto obj-prefix fallback is enabled")
        fallback_engine = build_swift_engine(
            model_id_or_path=args.model,
            infer_backend=args.infer_backend,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=gpu_mem,
            tensor_parallel_size=args.tensor_parallel_size,
            template_type=template_type,
            enable_thinking=enable_thinking,
            response_prefix="obj[",
            torch_dtype=torch_dtype,
        )

    samples = read_jsonl(args.test_jsonl)
    print(f"[init] samples={len(samples)} from {args.test_jsonl}")

    stop_list: Optional[List[str]] = None
    if args.no_stop_on_answer_close:
        stop_list = list(args.stop_sequence) if args.stop_sequence else None
    else:
        stop_list = ["</answer>"]
        if args.stop_sequence:
            stop_list = stop_list + list(args.stop_sequence)

    req_cfg = RequestConfig(max_tokens=args.max_new_tokens, temperature=args.temperature, stop=stop_list or [])

    prefix, suffix, wrap_prev_in_answer = build_prev_frame_template(samples)
    if prefix is None:
        print("[warn] previous-frame template not found; all rows run with original prompts")

    by_video: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for i, sample in enumerate(samples):
        try:
            vid, t = parse_video_and_frame(sample, images_base)
        except Exception:
            vid, t = f"_row{i}", i
        by_video[vid].append((t, i))
    for vid in by_video:
        by_video[vid].sort(key=lambda x: x[0])

    video_ids = list(by_video.keys())
    max_wave = max((len(v) for v in by_video.values()), default=0)
    predictions: List[Optional[str]] = [None] * len(samples)
    gen_times: List[Optional[float]] = [None] * len(samples)
    messages_used: List[Optional[List[Any]]] = [None] * len(samples)
    user_text_used: List[Optional[str]] = [None] * len(samples)
    prev_preds: Dict[str, Optional[str]] = {vid: None for vid in video_ids}

    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fout:
        pbar = tqdm(total=len(samples), desc="realtime infer", unit="sample")

        for wave_idx in range(max_wave):
            wave_items: List[Tuple[int, str, int]] = []
            for vid in video_ids:
                frames = by_video[vid]
                if wave_idx < len(frames):
                    t, gidx = frames[wave_idx]
                    wave_items.append((gidx, vid, t))

            wave_samples: List[Dict[str, Any]] = []
            wave_paths: List[str] = []
            wave_gidx: List[int] = []
            wave_vids: List[str] = []

            for gidx, vid, t in wave_items:
                sample = samples[gidx]
                try:
                    img_path = resolve_image_path(sample, images_base)
                    if not os.path.isfile(img_path):
                        raise FileNotFoundError(img_path)

                    if wave_idx == 0 or prefix is None:
                        run_sample = sample
                    else:
                        run_sample = copy.deepcopy(sample)
                        new_user = build_prompt_with_prev(
                            prefix,
                            suffix or "",
                            prev_preds[vid] or "",
                            wrap_prev_in_answer,
                        )
                        set_user_text(run_sample, new_user)

                    messages_used[gidx] = copy.deepcopy(run_sample.get("messages"))
                    user_text_used[gidx] = get_user_text(run_sample)
                    wave_samples.append(run_sample)
                    wave_paths.append(img_path)
                    wave_gidx.append(gidx)
                    wave_vids.append(vid)
                except Exception as e:
                    messages_used[gidx] = copy.deepcopy(sample.get("messages"))
                    rec = _error_record(
                        sample,
                        model_path=args.model,
                        run_name=args.run_name,
                        dataset_tag=args.dataset_tag,
                        checkpoint_step=ckpt_step,
                        test_jsonl=args.test_jsonl,
                        err=str(e),
                        img_path="",
                        prev_source=args.prev_source,
                    )
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    predictions[gidx] = ""
                    gen_times[gidx] = None
                    prev_preds[vid] = _prev_chain_value(args.prev_source, sample, "")
                    pbar.update(1)

            bs = max(1, args.batch_size)
            for b0 in range(0, len(wave_samples), bs):
                b_samples = wave_samples[b0 : b0 + bs]
                b_paths = wave_paths[b0 : b0 + bs]
                b_gidx = wave_gidx[b0 : b0 + bs]
                b_vids = wave_vids[b0 : b0 + bs]

                requests: List[InferRequest] = []
                meta: List[Tuple[int, Dict[str, Any], str, str]] = []
                for run_sample, path, gidx, vid in zip(b_samples, b_paths, b_gidx, b_vids):
                    try:
                        gt = get_assistant_text(samples[gidx])
                        user = get_user_text(run_sample)
                        requests.append(InferRequest(messages=[{"role": "user", "content": user}], images=[path]))
                        meta.append((gidx, samples[gidx], gt, vid))
                    except Exception as e:
                        rec = _error_record(
                            samples[gidx],
                            model_path=args.model,
                            run_name=args.run_name,
                            dataset_tag=args.dataset_tag,
                            checkpoint_step=ckpt_step,
                            test_jsonl=args.test_jsonl,
                            err=str(e),
                            img_path=path,
                            prev_source=args.prev_source,
                        )
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        predictions[gidx] = ""
                        gen_times[gidx] = None
                        prev_preds[vid] = _prev_chain_value(args.prev_source, samples[gidx], "")
                        pbar.update(1)

                if not requests:
                    continue

                t0 = time.perf_counter()
                try:
                    resp_list = engine.infer(requests, req_cfg)
                except Exception as e:
                    for gidx, sample, _gt, vid in meta:
                        rec = _error_record(
                            sample,
                            model_path=args.model,
                            run_name=args.run_name,
                            dataset_tag=args.dataset_tag,
                            checkpoint_step=ckpt_step,
                            test_jsonl=args.test_jsonl,
                            err=f"engine.infer failed: {e}",
                            img_path=resolve_image_path(sample, images_base) if sample.get("images") else "",
                            prev_source=args.prev_source,
                        )
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        predictions[gidx] = ""
                        gen_times[gidx] = None
                        prev_preds[vid] = _prev_chain_value(args.prev_source, sample, "")
                        pbar.update(1)
                    continue

                elapsed = time.perf_counter() - t0
                per = elapsed / max(1, len(resp_list))

                for (gidx, sample, gt, vid), resp in zip(meta, resp_list):
                    raw = resp.choices[0].message.content or ""
                    pred_clean = finalize_vl_prediction(raw)

                    if fallback_engine is not None and not has_obj_header(pred_clean):
                        try:
                            fb_req = InferRequest(
                                messages=[{"role": "user", "content": user_text_used[gidx] or get_user_text(sample)}],
                                images=[resolve_image_path(sample, images_base)],
                            )
                            fb_resp = fallback_engine.infer([fb_req], req_cfg)[0]
                            fb_pred = finalize_vl_prediction(fb_resp.choices[0].message.content or "")
                            if has_obj_header(fb_pred):
                                pred_clean = fb_pred
                        except Exception:
                            pass

                    predictions[gidx] = pred_clean
                    gen_times[gidx] = per
                    prev_preds[vid] = _prev_chain_value(args.prev_source, sample, pred_clean)
                    pbar.update(1)

        pbar.close()

        n_ok = 0
        for idx, sample in enumerate(samples):
            try:
                gt = get_assistant_text(sample)
            except Exception:
                gt = ""
            pred = predictions[idx] or ""
            if pred:
                n_ok += 1
            user_prompt = user_text_used[idx] or get_user_text(sample)
            record = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": pred},
                ],
                "images": sample.get("images"),
                "content": gt,
                "predict": pred,
                "model_name": args.run_name,
                "gen_time_sec": gen_times[idx],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp.replace(out_path)
    print(f"[done] non-empty predictions: {n_ok}/{len(samples)} -> {out_path}")


if __name__ == "__main__":
    main()
