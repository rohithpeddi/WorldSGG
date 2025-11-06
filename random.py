import subprocess
import tempfile
import os
import posixpath
from pathlib import PurePosixPath

# ========== CONFIG ==========
M1_HOST = "cs93371.utdallas.edu"
M1_USER = "user1"
M1_DIR  = "/path/on/machine1/folder_path"

M2_HOST = "machine2.example.com"
M2_USER = "user2"
M2_DIR  = "/path/on/machine2/folder_path2"
# ============================


def run_ssh(user, host, cmd):
    """Run an ssh command and return stdout as text."""
    full_cmd = ["ssh", f"{user}@{host}", cmd]
    return subprocess.check_output(full_cmd, text=True)


def list_remote_files(user, host, base_dir):
    """
    Returns a dict: rel_path -> absolute_remote_path
    Uses `find` over ssh to list files recursively.
    """
    # we cd first so find prints ./sub/.. style paths
    cmd = f'cd "{base_dir}" && find . -type f'
    out = run_ssh(user, host, cmd)
    files = {}
    for line in out.splitlines():
        rel = line.strip()
        if rel == ".":
            continue
        # normalize ./a/b.txt -> a/b.txt
        rel = rel[2:] if rel.startswith("./") else rel
        abs_path = posixpath.join(base_dir, rel)
        files[rel] = abs_path
    return files


def get_remote_mtime(user, host, abs_path):
    """
    Get remote mtime (epoch seconds) using stat.
    """
    cmd = f'stat -c %Y "{abs_path}"'
    out = run_ssh(user, host, cmd)
    return int(out.strip())


def sftp_get(user, host, remote_path, local_path):
    """
    Download remote_path -> local_path using sftp batch mode.
    """
    # we use -b - to read commands from stdin
    sftp_cmd = ["sftp", "-b", "-", f"{user}@{host}"]
    batch = f'get "{remote_path}" "{local_path}"\n'
    subprocess.run(sftp_cmd, input=batch, text=True, check=True)


def sftp_put(user, host, local_path, remote_path):
    """
    Upload local_path -> remote_path, creating directories as needed.
    We’ll create dirs using ssh mkdir -p first.
    """
    remote_dir = posixpath.dirname(remote_path)
    run_ssh(user, host, f'mkdir -p "{remote_dir}"')

    sftp_cmd = ["sftp", "-b", "-", f"{user}@{host}"]
    batch = f'put "{local_path}" "{remote_path}"\n'
    subprocess.run(sftp_cmd, input=batch, text=True, check=True)


def merge_dirs():
    # 1) list files on both machines (relative to their base dirs)
    files1 = list_remote_files(M1_USER, M1_HOST, M1_DIR)
    files2 = list_remote_files(M2_USER, M2_HOST, M2_DIR)

    all_rel = set(files1.keys()) | set(files2.keys())

    for rel in sorted(all_rel):
        in_1 = rel in files1
        in_2 = rel in files2

        if in_1 and not in_2:
            # copy 1 -> 2
            src_remote = files1[rel]
            dst_remote = posixpath.join(M2_DIR, rel)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                local_tmp = tmp.name
            try:
                print(f"[+] {rel}: present on M1 only, copying to M2")
                sftp_get(M1_USER, M1_HOST, src_remote, local_tmp)
                sftp_put(M2_USER, M2_HOST, local_tmp, dst_remote)
            finally:
                if os.path.exists(local_tmp):
                    os.remove(local_tmp)

        elif in_2 and not in_1:
            # copy 2 -> 1
            src_remote = files2[rel]
            dst_remote = posixpath.join(M1_DIR, rel)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                local_tmp = tmp.name
            try:
                print(f"[+] {rel}: present on M2 only, copying to M1")
                sftp_get(M2_USER, M2_HOST, src_remote, local_tmp)
                sftp_put(M1_USER, M1_HOST, local_tmp, dst_remote)
            finally:
                if os.path.exists(local_tmp):
                    os.remove(local_tmp)

        else:
            # present on both: pick newer
            m1_path = files1[rel]
            m2_path = files2[rel]
            m1_mtime = get_remote_mtime(M1_USER, M1_HOST, m1_path)
            m2_mtime = get_remote_mtime(M2_USER, M2_HOST, m2_path)

            if m1_mtime > m2_mtime:
                # overwrite M2 from M1
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    local_tmp = tmp.name
                try:
                    print(f"[~] {rel}: newer on M1 -> updating M2")
                    sftp_get(M1_USER, M1_HOST, m1_path, local_tmp)
                    sftp_put(M2_USER, M2_HOST, local_tmp, m2_path)
                finally:
                    if os.path.exists(local_tmp):
                        os.remove(local_tmp)
            elif m2_mtime > m1_mtime:
                # overwrite M1 from M2
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    local_tmp = tmp.name
                try:
                    print(f"[~] {rel}: newer on M2 -> updating M1")
                    sftp_get(M2_USER, M2_HOST, m2_path, local_tmp)
                    sftp_put(M1_USER, M1_HOST, local_tmp, m1_path)
                finally:
                    if os.path.exists(local_tmp):
                        os.remove(local_tmp)
            else:
                # same mtime, do nothing
                pass


if __name__ == "__main__":
    merge_dirs()
