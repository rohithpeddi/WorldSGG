"""
Manifest of cached artifacts for the worldbbox / WorldFormer / MLLM campaigns.

Every cached artifact (token grids, derived ROI features, MLLM caches,
prediction dumps) is registered here so that, when the annotations are
revised, only the annotation-DEPENDENT artifacts are regenerated.

    manifest.json = {
      "annotation_versions": {"<name>": "<md5>", ...},   # from make_test_split_worldbbox.py
      "artifacts": {
        "<key>": {"path": str, "inputs": [str], "annotation_version": str|None,
                  "code_commit": str, "created": str, "note": str}
      }
    }

Usage (library):
    from tools.cache_manifest import register, is_current
    register("tokens/dinov3l/test", path, inputs=["frames"], annotation_version=None)
    register("roi_derived/predcls/fused/test_worldbbox", path, inputs=[...],
             annotation_version=current_annotation_version("test_worldbbox"))

CLI:
    python tools/cache_manifest.py --manifest /data3/rohith/ag/cache/manifest.json list
    python tools/cache_manifest.py --manifest ... stale --annotation test_worldbbox
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_MANIFEST = "/data3/rohith/ag/cache/manifest.json"


def _load(manifest: str) -> dict:
    p = Path(manifest)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"annotation_versions": {}, "artifacts": {}}


def _save(manifest: str, data: dict) -> None:
    p = Path(manifest)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, p)


def _code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def set_annotation_version(name: str, md5: str, manifest: str = DEFAULT_MANIFEST) -> None:
    data = _load(manifest)
    data["annotation_versions"][name] = md5
    _save(manifest, data)


def current_annotation_version(name: str, manifest: str = DEFAULT_MANIFEST) -> Optional[str]:
    return _load(manifest)["annotation_versions"].get(name)


def register(key: str, path: str, inputs: Iterable[str] = (),
             annotation_version: Optional[str] = None, note: str = "",
             manifest: str = DEFAULT_MANIFEST) -> None:
    data = _load(manifest)
    data["artifacts"][key] = {
        "path": str(path),
        "inputs": list(inputs),
        "annotation_version": annotation_version,
        "code_commit": _code_commit(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
    }
    _save(manifest, data)


def is_current(key: str, annotation_name: Optional[str] = None,
               manifest: str = DEFAULT_MANIFEST) -> bool:
    """True if the artifact exists and (if annotation-dependent) was built
    against the current version of ``annotation_name``."""
    data = _load(manifest)
    art = data["artifacts"].get(key)
    if art is None or not Path(art["path"]).exists():
        return False
    if art["annotation_version"] is None:
        return True
    if annotation_name is None:
        return True
    return art["annotation_version"] == data["annotation_versions"].get(annotation_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    st = sub.add_parser("stale")
    st.add_argument("--annotation", required=True)
    args = ap.parse_args()

    data = _load(args.manifest)
    if args.cmd == "list":
        print(json.dumps(data["annotation_versions"], indent=2))
        for k, v in sorted(data["artifacts"].items()):
            dep = v["annotation_version"] or "-"
            print(f"{k:55s} annot={dep[:10]:10s} {v['created']}  {v['path']}")
    elif args.cmd == "stale":
        cur = data["annotation_versions"].get(args.annotation)
        for k, v in sorted(data["artifacts"].items()):
            if v["annotation_version"] is not None and v["annotation_version"] != cur:
                print(k)


if __name__ == "__main__":
    main()
