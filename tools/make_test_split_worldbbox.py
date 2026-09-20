"""
Lock the worldbbox test set: write the sorted video-id list and an
``annotation_version`` (md5 over the sorted per-file md5s of the PKLs) into
the cache manifest, so every derived artifact can record which annotation
revision it was built from.

    python tools/make_test_split_worldbbox.py \
        --data_path /data/rohith/ag \
        --annot_dir world4d_rel_annotations_worldbbox \
        --out /data3/rohith/ag/splits/test_worldbbox_1511.txt \
        --manifest /data3/rohith/ag/cache/manifest.json
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.cache_manifest import set_annotation_version  # noqa: E402


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/data/rohith/ag")
    ap.add_argument("--annot_dir", default="world4d_rel_annotations_worldbbox")
    ap.add_argument("--phase", default="test")
    ap.add_argument("--name", default=None, help="manifest key (default: <phase>_<annot_dir suffix>)")
    ap.add_argument("--out", default="/data3/rohith/ag/splits/test_worldbbox_1511.txt")
    ap.add_argument("--manifest", default="/data3/rohith/ag/cache/manifest.json")
    args = ap.parse_args()

    annot_dir = Path(args.data_path) / args.annot_dir / args.phase
    pkls = sorted(annot_dir.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"no PKLs under {annot_dir}")

    vids, per_file = [], []
    for p in pkls:
        vid = p.name[:-4]
        if vid.endswith(".mp4"):
            vid = vid[:-4]
        vids.append(vid)
        per_file.append(f"{p.name}:{md5_file(p)}")
    version = hashlib.md5("\n".join(per_file).encode()).hexdigest()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sorted(vids)) + "\n", encoding="utf-8")
    (out.with_suffix(".md5s.txt")).write_text("\n".join(per_file) + "\n", encoding="utf-8")

    name = args.name or f"{args.phase}_{args.annot_dir.split('world4d_rel_annotations')[-1].strip('_') or 'legacy'}"
    set_annotation_version(name, version, manifest=args.manifest)
    print(f"{len(vids)} videos -> {out}")
    print(f"annotation_version[{name}] = {version}  (manifest: {args.manifest})")


if __name__ == "__main__":
    main()
