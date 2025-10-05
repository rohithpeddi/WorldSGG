import json
import pickle


def convert_pkl_to_json():
    pkl_file_path = r"C:\Users\rohit\PycharmProjects\Scene4Cast\datasets\4d_video_frame_id_list.pkl"
    json_file_path = r"C:\Users\rohit\PycharmProjects\Scene4Cast\datasets\4d_video_frame_id_list.json"
    with open(pkl_file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    sorted_unique_data = {}
    for key, value in data.items():
        sorted_unique_data[key] = sorted(set(value))

    with open(json_file_path, 'w') as json_file:
        json.dump(sorted_unique_data, json_file)


#!/usr/bin/env python3
"""
Copy .zip/.pkl files from input -> output (preserving folders), verify, then delete source.

Defaults:
  input:  /data/rohith/captain_cook/videos/hololens/
  output: /data2/rohith/captain_cook/videos/hololens/

Verification:
  - default: size match
  - optional: --verify sha256   (strong integrity check, slower)
  - optional: --verify none     (not recommended)

Usage:
  python transfer_copy_then_delete.py
  python transfer_copy_then_delete.py --overwrite
  python transfer_copy_then_delete.py --verify sha256
  python transfer_copy_then_delete.py --dry-run
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable

ALLOWED_EXTS = {".zip", ".pkl"}


def iter_target_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            yield p


def sha256sum(path: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_copy(src: Path, dst: Path, mode: str) -> bool:
    if mode == "none":
        return True
    if mode == "size":
        try:
            return src.stat().st_size == dst.stat().st_size
        except FileNotFoundError:
            return False
    if mode == "sha256":
        try:
            return sha256sum(src) == sha256sum(dst)
        except FileNotFoundError:
            return False
    raise ValueError(f"Unknown verify mode: {mode}")


def copy_atomically(src: Path, dst: Path, overwrite: bool) -> Path:
    """
    Copy to dst.parent as a temporary .partial file, then os.replace to final path.
    Returns the final destination path when the copy step completes (before verification).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")

    if dst.exists():
        if not overwrite:
            return dst  # signal to caller that final exists; they decide to skip
        try:
            dst.unlink()
        except FileNotFoundError:
            pass

    # Clean any stale partial
    if tmp.exists():
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass

    shutil.copy2(src, tmp)  # copy metadata too
    # Atomically flip into place (same directory -> same filesystem)
    os.replace(tmp, dst)
    return dst


def main():
    parser = argparse.ArgumentParser(description="Copy .zip/.pkl from input to output, verify, then delete source.")
    parser.add_argument("-i", "--input", default="/data/rohith/captain_cook/videos/hololens/",
                        help="Input root directory")
    parser.add_argument("-o", "--output", default="/data2/rohith/captain_cook/videos/hololens/",
                        help="Output root directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination files if they already exist")
    parser.add_argument("--verify", choices=["size", "sha256", "none"], default="size",
                        help="Verification mode before deleting source")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without copying/deleting")
    args = parser.parse_args()

    in_root = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()

    if not in_root.is_dir():
        print(f"ERROR: input dir not found: {in_root}", file=sys.stderr)
        sys.exit(1)
    if in_root == out_root:
        print("ERROR: input and output cannot be the same directory.", file=sys.stderr)
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Mode: COPY → VERIFY({args.verify}) → DELETE")
    print(f"From: {in_root}\nTo:   {out_root}\nOverwrite: {args.overwrite}  Dry-run: {args.dry_run}\n")

    found = copied = skipped = deleted = errors = 0

    for src in iter_target_files(in_root):
        found += 1
        rel = src.relative_to(in_root)
        dst = out_root / rel

        if args.dry_run:
            exists = dst.exists()
            if exists and not args.overwrite:
                print(f"[SKIP   ] {src} -> {dst} (exists)")
                skipped += 1
            else:
                print(f"[COPY   ] {src} -> {dst}")
                print(f"[DELETE ] {src} (after verify)")
            continue

        try:
            if dst.exists() and not args.overwrite:
                skipped += 1
                print(f"[SKIP   ] {dst} exists (use --overwrite)")
                continue

            copy_atomically(src, dst, overwrite=args.overwrite)

            if not verify_copy(src, dst, args.verify):
                errors += 1
                print(f"[ERROR  ] Verification failed for {src} -> {dst}. Source kept.", file=sys.stderr)
                continue

            copied += 1
            try:
                src.unlink()
                deleted += 1
                print(f"[OK     ] Copied+Verified+Deleted: {src.name}")
            except Exception as e:
                errors += 1
                print(f"[ERROR  ] Copied+Verified but failed to delete source {src}: {e}", file=sys.stderr)

        except Exception as e:
            errors += 1
            print(f"[ERROR  ] {src} -> {dst}: {e}", file=sys.stderr)

    if args.dry_run:
        print(f"\nDry run complete. Matched: {found}")
    else:
        print(
            f"\nDone. Matched: {found} | Copied: {copied} | Deleted: {deleted} | Skipped: {skipped} | Errors: {errors}")
        print(f"Destination root: {out_root}")


if __name__ == "__main__":
    main()
