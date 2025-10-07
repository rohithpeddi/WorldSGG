import os
import argparse
from boxsdk import OAuth2, Client, CCGAuth
from tqdm import tqdm


class SyncService:
    def __init__(self, local_root, box_folder_id, mode):
        self.local_root = local_root
        self.box_folder_id = box_folder_id
        self.mode = mode

        user_id = '23441227496'
        root_folder_id = '317554414852'
        client_id = 'krr2b0dmxvnqn83ikpe6ufs58jg9t82b'
        client_secret = 'TTsVwLrnv9EzmKJv67yrCyUM09wJSriK'
        ccg_auth = CCGAuth(
            client_id=client_id,
            client_secret=client_secret,
            user=user_id
        )
        self.client = Client(ccg_auth)

    def get_local_files(self):
        """Walk local tree, return {rel_path: size_in_bytes}."""
        local = {}
        for dirpath, _, files in os.walk(self.local_root):
            for fname in tqdm(files, desc=f"Scanning {self.local_root}", unit='file'):
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, self.local_root)
                local[rel] = os.path.getsize(full)
        return local

    def get_box_files(self, folder_id, parent_path=''):
        """
        Recursively list Box folder.
        Returns {rel_path: (size_in_bytes, file_id)}.
        """
        box = {}
        items = self.client.folder(folder_id=folder_id).get_items(limit=1000, offset=0)
        for item in tqdm(items, desc=f"Scanning Box folder {folder_id}", unit='file'):
            rel = os.path.join(parent_path, item.name)
            if item.type == 'file':
                info = self.client.file(item.id).get()
                box[rel] = (info.size, item.id)
            else:  # folder
                box.update(self.get_box_files(item.id, rel))
        return box

    def ensure_box_folder_path(self, root_id, path):
        """
        Ensure nested folders exist on Box.
        Returns the folder_id of `path`.
        """
        parts = path.split(os.sep)
        current = root_id
        for p in parts:
            # look for existing subfolder
            entries = self.client.folder(current).get_items(limit=1000)
            sub = next((e for e in entries if e.type == 'folder' and e.name == p), None)
            if sub:
                current = sub.id
            else:
                sub = self.client.folder(current).create_subfolder(p)
                current = sub.id
        return current

    def upload_new(self, root_id, local_root, local_files, box_files):
        for rel, size in tqdm(local_files.items(), desc=f"Uploading to Box folder {root_id}", unit='file'):
            if rel not in box_files:
                src = os.path.join(local_root, rel)
                parent = os.path.dirname(rel)
                dest_id = self.ensure_box_folder_path(root_id, parent) if parent else root_id
                print(f"Uploading {rel} ({size} bytes)…")
                self.client.folder(dest_id).upload(src, file_name=os.path.basename(rel))

    def download_new(self, root_id, local_root, local_files, box_files):
        for rel, (size, fid) in tqdm(box_files.items()):
            if rel not in local_files:
                dst = os.path.join(local_root, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                print(f"Downloading {rel} ({size} bytes)…")
                with open(dst, 'wb') as f:
                    self.client.file(fid).download_to(f)

    def print_summary(self, local_files, box_files):
        print("-------------------------------------------------")
        print("Printing summary:")
        n_local = len(local_files)
        n_box = len(box_files)
        s_local = sum(local_files.values())
        s_box = sum(s for s, _ in box_files.values())

        # Summary in human-readable format of GB
        print(f"\nSummary:")
        print(f"  Local files: {n_local} ({s_local / 1e9:.2f} GB)")
        print(f"  Box files: {n_box} ({s_box / 1e9:.2f} GB)")
        print("-------------------------------------------------")

    def process_files(self):
        local_files = self.get_local_files()
        box_files = self.get_box_files(self.box_folder_id)
        self.print_summary(local_files, box_files)

        print("-------------------------------------------------")
        if self.mode in ('download', 'sync'):
            print("Downloading new files from Box…")
            self.download_new(self.box_folder_id, self.local_root, local_files, box_files)

        if self.mode in ('upload', 'sync'):
            print("Uploading new files to Box…")
            self.upload_new(self.box_folder_id, self.local_root, local_files, box_files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--local_dir', default="/data/rohith/ag/static_videos", help='Path to local directory')
    p.add_argument('--box_folder_id', default='345011408563', help='Box folder ID')
    p.add_argument('--mode', choices=['upload', 'download', 'sync'], default='sync')
    args = p.parse_args()
    os.makedirs(args.local_dir, exist_ok=True)

    sync_service = SyncService(args.local_dir, args.box_folder_id, args.mode)
    sync_service.process_files()


if __name__ == '__main__':
    main()
