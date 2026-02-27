#!/usr/bin/env python3
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Set, Optional, Any, Dict

# Adjust the import paths if your project structure differs.
# from dataloader.standard.action_genome.ag_dataset import StandardAG

# import seaborn as sns

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Buckets in MB
BUCKETS_MB = [10, 20, 30, 40, 50]



#
# def _extract_video_ids(video_list: List[List[str]]) -> List[str]:
#     ids: Set[str] = set()
#     for frames in video_list:
#         if not frames:
#             continue
#         # frame relpath looks like: video_id/frame_num.jpg (or similar)
#         video_id = frames[0].split("/")[0]
#         ids.add(video_id)
#     return sorted(ids)


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--out",
#         default="/data/rohith/ag/video_splits.json",
#         help="Output JSON path.",
#     )
#
#     args = parser.parse_args()
#
#     test_dataset = StandardAG(
#         phase="test",
#         mode="sgdet",
#         datasize="large",
#         data_path="/data/rohith/ag",
#         filter_nonperson_box_frame=True,
#         filter_small_box=False,
#     )
#
#     test_video_ids = _extract_video_ids(test_dataset.video_list)
#
#     train_dataset = StandardAG(
#         phase="train",
#         mode="sgdet",
#         datasize="large",
#         data_path="/data/rohith/ag",
#         filter_nonperson_box_frame=True,
#         filter_small_box=False,
#     )
#
#     train_video_ids = _extract_video_ids(train_dataset.video_list)
#
#     out_obj = {
#         "train": train_video_ids,
#         "test": test_video_ids,
#     }
#
#     os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
#     with open(args.out, "w", encoding="utf-8") as f:
#         json.dump(out_obj, f, indent=2)
#
#     print(f"Wrote {args.out} (train={len(out_obj['train'])}, test={len(out_obj['test'])})")



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
        glb_path = os.path.join(
            static_video_dir_path,
            static_video_name,
            f"{static_video_name[:-3]}.glb"
        )
        if not os.path.exists(glb_path):
            continue
        size_bytes = os.path.getsize(glb_path)
        files_with_sizes.append((glb_path, size_bytes))
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

    # ---------------------------------------------------------------------
    # NEW: write an Excel file with one sheet per size bucket
    # ---------------------------------------------------------------------
    # output_excel = "charades_video_sizes_by_bucket.xlsx"
    # with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
    #     # summary sheet first (the full df)
    #     df.sort_values(["size_bucket", "quality", "video_id"]).to_excel(
    #         writer, sheet_name="all_videos", index=False
    #     )
    #
    #     # per-bucket sheets
    #     for bucket in ordered_labels:
    #         bucket_df = df[df["size_bucket"] == bucket].copy()
    #         # sort to make it nice: quality -> video_id
    #         bucket_df = bucket_df.sort_values(["quality", "video_id"])
    #         # keep only the relevant columns; you can add/remove as needed
    #         cols = ["video_id", "quality", "size_mb"]
    #         bucket_df.to_excel(
    #             writer,
    #             sheet_name=bucket[:31],  # Excel sheet name max length = 31
    #             index=False,
    #             columns=cols,
    #         )
    # print(f"Wrote Excel report to {output_excel}")
    # ---------------------------------------------------------------------

    # 5) pivot to counts: rows=quality, cols=size_bucket, values=count
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

    # bar plot by size bucket
    counts_by_bucket = df["size_bucket"].value_counts().reindex(ordered_labels, fill_value=0)
    plt.figure()
    counts_by_bucket.plot(kind="bar")
    plt.title("Number of video files by size range")
    plt.xlabel("Size range")
    plt.ylabel("Count of videos")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()



def load_rag_predictions(
    rag_results_dir: Path, mode: str, model_name: str, video_id: str,
) -> Optional[Dict[str, Any]]:
    """Load the RAG prediction pkl for a single video."""
    pkl_path = Path(rag_results_dir) / mode / model_name / f"{video_id}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    mllms_dir = Path("/data/rohith/ag/mllms")
    rag_results_dir = mllms_dir / "rag_results"
    rag_all_objects_results_dir = mllms_dir / "rag_all_objects_results"
    subtitle_all_objects_results_dir = mllms_dir / "subtitle_all_objects_results"

    mode = "predcls"
    model_name = "internvl"
    video_id = "3C1ZN.mp4"

    predcls_rag_preds = load_rag_predictions(subtitle_all_objects_results_dir, mode, model_name, video_id)
    print(predcls_rag_preds)