#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Buckets in MB
BUCKETS_MB = [10, 20, 30, 40, 50]


def human_readable_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def bucket_label_from_size_mb(size_mb: float, buckets_mb):
    prev = 0
    for b in buckets_mb:
        if size_mb < b:
            return f"< {b} MB" if prev == 0 else f"{prev}–{b} MB"
        prev = b
    return f">= {buckets_mb[-1]} MB"


def main():
    # you can make these args if you want
    static_video_dir_path = "/data3/rohith/ag/ag4D/static_scenes/pi3_static"
    charades_train_data_filename = "Charades_v1_train.csv"
    charades_test_data_filename = "Charades_v1_test.csv"

    # 1) collect file sizes
    files_with_sizes = []
    video_id_filesize = {}  # video_id -> size_bytes

    for static_video_name in os.listdir(static_video_dir_path):
        # e.g. dir: ABCD1234 and file inside: ABCD12.glb ?
        # you're doing static_video_name[:-3] so preserve your logic
        glb_path = os.path.join(
            static_video_dir_path,
            static_video_name,
            f"{static_video_name[:-3]}.glb"
        )
        if not os.path.exists(glb_path):
            # skip if that file doesn't actually exist
            continue
        size_bytes = os.path.getsize(glb_path)
        files_with_sizes.append((glb_path, size_bytes))
        # use the same id key you used originally
        video_id = static_video_name[:-3]
        video_id_filesize[video_id] = size_bytes

    # 2) read charades CSVs to get quality per video_id
    charades_train_data = pd.read_csv(charades_train_data_filename)
    charades_test_data = pd.read_csv(charades_test_data_filename)

    video_id_quality = {}

    for _, row in charades_train_data.iterrows():
        video_id = row["id"]
        video_quality = row["quality"]
        video_id_quality[video_id] = video_quality

    for _, row in charades_test_data.iterrows():
        video_id = row["id"]
        video_quality = row["quality"]
        video_id_quality[video_id] = video_quality

    # 3) build a table: video_id, size_mb, size_bucket, quality
    rows = []
    for video_id, size_bytes in video_id_filesize.items():
        if video_id not in video_id_quality:
            # your static names might not match Charades ids; skip quietly
            continue
        quality = video_id_quality[video_id]
        size_mb = size_bytes / (1024 * 1024)
        size_bucket = bucket_label_from_size_mb(size_mb, BUCKETS_MB)
        rows.append(
            {
                "video_id": video_id,
                "size_mb": size_mb,
                "size_bucket": size_bucket,
                "quality": quality,
            }
        )

    if not rows:
        print("No overlapping video IDs between filesystem and Charades CSVs.")
        return

    df = pd.DataFrame(rows)

    # 4) make sure buckets are ordered
    ordered_labels = []
    prev = 0
    for b in BUCKETS_MB:
        if prev == 0:
            lbl = f"< {b} MB"
        else:
            lbl = f"{prev}–{b} MB"
        ordered_labels.append(lbl)
        prev = b
    ordered_labels.append(f">= {BUCKETS_MB[-1]} MB")

    df["size_bucket"] = pd.Categorical(df["size_bucket"], ordered=True, categories=ordered_labels)

    # 5) pivot to counts: rows=quality, cols=size_bucket, values=count
    # you could also do aggfunc="mean" on size_mb if quality were numeric, but it’s probably string
    pivot_table = pd.pivot_table(
        df,
        index="quality",
        columns="size_bucket",
        values="video_id",
        aggfunc="count",
        fill_value=0,
    )

    # 6) plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar_kws={"label": "Num videos"},
    )
    plt.title("Number of videos by quality and file-size bucket")
    plt.xlabel("File-size bucket")
    plt.ylabel("Video quality")
    plt.tight_layout()
    plt.show()

    # (optional) your earlier bar plot by size bucket — now using what we already computed
    counts_by_bucket = df["size_bucket"].value_counts().reindex(ordered_labels, fill_value=0)
    plt.figure()
    counts_by_bucket.plot(kind="bar")
    plt.title("Number of video files by size range")
    plt.xlabel("Size range")
    plt.ylabel("Count of videos")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
