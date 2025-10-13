import os
import argparse
from boxsdk import OAuth2, Client, CCGAuth
from tqdm import tqdm

#!/usr/bin/env python3
import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Iterable, Optional, Set

from boxsdk import Client, CCGAuth
from boxsdk.exception import BoxAPIException
from tqdm import tqdm


# ------------------------- Helpers ------------------------- #

def human_gb(nbytes: int) -> str:
    return f"{nbytes / 1e9:.2f} GB"


def should_include(path: str, include_ext: Optional[Set[str]], exclude_ext: Optional[Set[str]]) -> bool:
    if include_ext is None and exclude_ext is None:
        return True
    ext = Path(path).suffix.lower()
    if include_ext is not None and ext not in include_ext:
        return False
    if exclude_ext is not None and ext in exclude_ext:
        return False
    return True


def backoff_sleep(attempt: int, retry_after: Optional[float] = None):
    # If Box returns Retry-After, respect it; else exponential backoff with jitter
    if retry_after is not None:
        time.sleep(retry_after)
    else:
        base = 0.75
        delay = min(30.0, base * (2 ** attempt))
        time.sleep(delay)


# ------------------------- Sync Service ------------------------- #

class SyncService:
    def __init__(self, local_root: str, box_folder_id: str, mode: str,
                 workers: int = 4, dry_run: bool = False,
                 include_ext: Optional[Set[str]] = None,
                 exclude_ext: Optional[Set[str]] = None):
        self.local_root = str(Path(local_root).resolve())
        self.box_folder_id = box_folder_id
        self.mode = mode
        self.workers = max(1, workers)
        self.dry_run = dry_run
        self.include_ext = include_ext
        self.exclude_ext = exclude_ext

        # ---- CCG (JWT/Server-to-Server) Auth; credentials via env ----

        user_id = '23441227496'
        root_folder_id = '317554414852'
        client_id = 'krr2b0dmxvnqn83ikpe6ufs58jg9t82b'
        client_secret = 'TTsVwLrnv9EzmKJv67yrCyUM09wJSriK'

        if not all([client_id, client_secret, user_id]):
            print("ERROR: Please export BOX_CLIENT_ID, BOX_CLIENT_SECRET, and BOX_USER_ID.", file=sys.stderr)
            sys.exit(1)

        ccg_auth = CCGAuth(
            client_id=client_id,
            client_secret=client_secret,
            user=user_id
        )
        self.client = Client(ccg_auth)

        # Cache {folder_id: {child_folder_name: child_folder_id}}
        self._folder_children_cache: Dict[str, Dict[str, str]] = {}

    # --------------- Robust Box calls with retries --------------- #

    def _with_retries(self, fn, *args, **kwargs):
        max_attempts = 7
        for attempt in range(max_attempts):
            try:
                return fn(*args, **kwargs)
            except BoxAPIException as e:
                # Retry on typical transient statuses
                if e.status in (429, 500, 502, 503, 504):
                    retry_after = None
                    try:
                        retry_after = float(e.network_response.headers.get("Retry-After", ""))
                    except Exception:
                        pass
                    backoff_sleep(attempt, retry_after)
                    continue
                raise

    # --------------- Local scanning --------------- #

    def get_local_files(self) -> Dict[str, int]:
        """
        Walk the local tree and return {rel_posix_path: size_in_bytes}.
        """
        local: Dict[str, int] = {}
        for dirpath, _, files in os.walk(self.local_root):
            for fname in files:
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, self.local_root)
                rel_posix = Path(rel).as_posix()
                if should_include(rel_posix, self.include_ext, self.exclude_ext):
                    try:
                        local[rel_posix] = os.path.getsize(full)
                    except FileNotFoundError:
                        # File disappeared between list and stat; skip
                        continue
        return local

    # --------------- Box listing (paginated + minimal fields) --------------- #

    def _iter_box_folder_items(self, folder_id: str, fields: Iterable[str]) -> Iterable:
        """
        Iterate items in a Box folder using offset pagination.
        """
        limit = 1000
        offset = 0
        fields = list(fields)
        while True:
            items_page = self._with_retries(
                self.client.folder(folder_id=folder_id).get_items,
                limit=limit, offset=offset, fields=fields
            )
            items_list = list(items_page)
            if not items_list:
                break
            for it in items_list:
                yield it
            if len(items_list) < limit:
                break
            offset += len(items_list)

    def get_box_files(self, root_folder_id: str) -> Dict[str, Tuple[int, str]]:
        """
        Recursively list Box from the given root folder.
        Returns {rel_posix_path: (size_in_bytes, file_id)}.
        """
        out: Dict[str, Tuple[int, str]] = {}
        stack = [(root_folder_id, "")]  # (folder_id, parent_rel)
        fields = ["id", "name", "type", "size", "sha1"]

        with tqdm(desc="Scanning Box (recursive)", unit="item") as pbar:
            while stack:
                folder_id, parent = stack.pop()
                for item in self._iter_box_folder_items(folder_id, fields):
                    pbar.update(1)
                    rel = (Path(parent) / item.name).as_posix()
                    if item.type == "file":
                        size = item.size or 0
                        out[rel] = (size, item.id)
                    elif item.type == "folder":
                        stack.append((item.id, rel))

        return out

    # --------------- Folder children cache --------------- #

    def _get_folder_children(self, folder_id: str) -> Dict[str, str]:
        """
        Return {child_folder_name: child_folder_id} for a Box folder, cached.
        """
        if folder_id in self._folder_children_cache:
            return self._folder_children_cache[folder_id]

        children: Dict[str, str] = {}
        for it in self._iter_box_folder_items(folder_id, fields=["id", "name", "type"]):
            if it.type == "folder":
                children[it.name] = it.id

        self._folder_children_cache[folder_id] = children
        return children

    def ensure_box_folder_path(self, root_id: str, rel_dir: str) -> str:
        """
        Ensure that 'rel_dir' exists under Box 'root_id'. Returns target folder_id.
        Uses cache to avoid repeated list calls.
        """
        rel_dir = Path(rel_dir).as_posix()
        if rel_dir in ("", "."):
            return root_id

        current = root_id
        for part in Path(rel_dir).parts:
            children = self._get_folder_children(current)
            if part in children:
                current = children[part]
                continue
            # Create subfolder
            if self.dry_run:
                # Simulate an ID for dry run, but keep cache coherent
                created_id = f"DRYRUN_{part}"
            else:
                created = self._with_retries(self.client.folder(current).create_subfolder, part)
                created_id = created.id
            # Update cache
            children = self._get_folder_children(current)
            children[part] = created_id
            current = created_id
        return current

    # --------------- Upload / Download (parallel) --------------- #

    def _upload_one(self, dest_root_id: str, rel_path: str):
        src = Path(self.local_root) / rel_path
        dest_dir = Path(rel_path).parent.as_posix()
        target_folder_id = self.ensure_box_folder_path(dest_root_id, dest_dir)
        if self.dry_run:
            return
        self._with_retries(self.client.folder(target_folder_id).upload, str(src), file_name=src.name)

    def _download_one(self, fid: str, rel_path: str):
        dst = Path(self.local_root) / rel_path
        if self.dry_run:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            self._with_retries(self.client.file(fid).download_to, f)

    def upload_new(self, root_id: str, local_files: Dict[str, int], box_files: Dict[str, Tuple[int, str]]):
        to_upload = [rel for rel in local_files.keys() if rel not in box_files]
        if not to_upload:
            print("Nothing to upload.")
            return

        print(f"Uploading {len(to_upload)} new files to Box (workers={self.workers})...")
        with ThreadPoolExecutor(max_workers=self.workers) as ex, tqdm(total=len(to_upload), unit="file") as pbar:
            futures = [ex.submit(self._upload_one, root_id, rel) for rel in to_upload]
            for fut in as_completed(futures):
                _ = fut.result()
                pbar.update(1)

    def download_new(self, root_id: str, local_files: Dict[str, int], box_files: Dict[str, Tuple[int, str]]):
        to_download = [(rel, fid) for rel, (_, fid) in box_files.items() if rel not in local_files]
        if not to_download:
            print("Nothing to download.")
            return

        print(f"Downloading {len(to_download)} new files from Box (workers={self.workers})...")
        with ThreadPoolExecutor(max_workers=self.workers) as ex, tqdm(total=len(to_download), unit="file") as pbar:
            futures = [ex.submit(self._download_one, fid, rel) for (rel, fid) in to_download]
            for fut in as_completed(futures):
                _ = fut.result()
                pbar.update(1)

    # --------------- Summary & Orchestration --------------- #

    def print_summary(self, local_files: Dict[str, int], box_files: Dict[str, Tuple[int, str]]):
        n_local, s_local = len(local_files), sum(local_files.values())
        n_box, s_box = len(box_files), sum(size for size, _ in box_files.values())
        print("\n-------------------- SUMMARY --------------------")
        print(f"Local: {n_local} files, {human_gb(s_local)}")
        print(f"Box  : {n_box} files, {human_gb(s_box)}")
        print("-------------------------------------------------")

    def process_files(self):
        print("Scanning local files…")
        local_files = self.get_local_files()

        print("Scanning Box files…")
        box_files = self.get_box_files(self.box_folder_id)

        # Optional extension filters apply to both ends
        if self.include_ext or self.exclude_ext:
            local_files = {k: v for k, v in local_files.items()
                           if should_include(k, self.include_ext, self.exclude_ext)}
            box_files = {k: v for k, v in box_files.items()
                         if should_include(k, self.include_ext, self.exclude_ext)}

        self.print_summary(local_files, box_files)

        if self.mode in ("download", "sync"):
            print("\nDownloading new files from Box…")
            self.download_new(self.box_folder_id, local_files, box_files)

        if self.mode in ("upload", "sync"):
            print("\nUploading new files to Box…")
            self.upload_new(self.box_folder_id, local_files, box_files)


# ------------------------- CLI ------------------------- #

def parse_ext_list(val: Optional[str]) -> Optional[Set[str]]:
    if not val:
        return None
    exts = {e.strip().lower() for e in val.split(",") if e.strip()}
    exts = {e if e.startswith(".") else f".{e}" for e in exts}
    return exts


def main():
    p = argparse.ArgumentParser(description="Fast parallel sync between local folder and a Box folder.")
    p.add_argument("--local_dir", required=False, default="/data/rohith/ag/static_videos")
    p.add_argument("--box_folder_id", required=False, default="345011408563")
    p.add_argument("--mode", choices=["upload", "download", "sync"], default="sync")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers for up/downloading (default: 4)")
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not perform writes.")
    p.add_argument("--include-ext", type=str, default=None,
                   help="Comma-separated list of file extensions to include (e.g. 'mp4,mov').")
    p.add_argument("--exclude-ext", type=str, default=None,
                   help="Comma-separated list of file extensions to exclude.")
    args = p.parse_args()

    Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    include_ext = parse_ext_list(args.include_ext)
    exclude_ext = parse_ext_list(args.exclude_ext)

    svc = SyncService(
        local_root=args.local_dir,
        box_folder_id=args.box_folder_id,
        mode=args.mode,
        workers=args.workers,
        dry_run=args.dry_run,
        include_ext=include_ext,
        exclude_ext=exclude_ext,
    )
    svc.process_files()


if __name__ == "__main__":
    main()