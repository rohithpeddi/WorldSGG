import os
import sys
import torch
import argparse

from tqdm import tqdm

from datasets.preprocess.utils import load_file_pkl_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Object Labels:
0 - Background
1 - Person
2...36 --> Object Classes

Total Number of Classes: 37

1. We have constructed object class level masks following SAM2.
Where we have for each object id --> All frames from a starting point and the corresponding pixels of the mask.
2. Now we need to construct a single tensor with the shape of (H, W) and fill the value of the each pixel with the object id at each frame.
"""


class SAM2Processor:

    def __init__(
            self,
            ag_root_directory,
    ):
        self.ag_root_directory = ag_root_directory
        self.segmentation_root_dir = os.path.join(self.ag_root_directory, "segmentation")

        # Segmentation storage paths
        self.train_vid_segmentation_path = os.path.join(self.segmentation_root_dir, "train", "predcls")
        self.test_vid_segmentation_path = os.path.join(self.segmentation_root_dir, "test", "predcls")

        self.processed_segmentation_path = os.path.join(self.segmentation_root_dir, "processed")
        if not os.path.exists(self.processed_segmentation_path):
            os.makedirs(self.processed_segmentation_path)

    def construct_video_segmentation_tensor(self, video_id, phase):
        video_sam2_output_path = os.path.join(self.segmentation_root_dir, phase, "predcls", f"{video_id[:-4]}.pkl")
        sam2_video_data = load_file_pkl_file(video_sam2_output_path)

        segmentation_mask_tensor = None
        # 1. Construct a tensor based on shape of the frame
        # -1 : {
        # 		"video_id": "video_id",
        # 		"object_id": {
        # 		       	"frame_id": {"object_id": mask} --> (1, H, W)
        # 		  	}
        #       }
        for key, value in sam2_video_data.items():
            if key == "video_id":
                assert sam2_video_data["video_id"] == video_id[
                    :-4], "The video id should be the same as the one in the file name"
            else:
                # key --> object_id
                # value --> {}
                object_id = key
                frames_with_object_mask_dict = value
                if segmentation_mask_tensor is None:
                    frame_id_list = list(frames_with_object_mask_dict.keys())
                    B, H, W = frames_with_object_mask_dict[frame_id_list[0]][object_id].shape
                    num_frames_video = len(
                        os.listdir(os.path.join(self.ag_root_directory, "frames", f"{video_id[:-4]}.mp4")))
                    segmentation_mask_tensor = torch.zeros((num_frames_video, H, W), dtype=torch.int32)
                # 2. Fill the tensor with the object id at each frame
                for frame_id, mask_dict in frames_with_object_mask_dict.items():
                    mask_array = mask_dict[object_id].squeeze()
                    # 2a. Add the object label to the True positions of the mask in frame_id, H, W [segmentation_mask_tensor]
                    # Correcting for the person label in the objects
                    object_label = int(object_id) if object_id != -1 else 1
                    segmentation_mask_tensor[frame_id][mask_array == 1] = object_label

        # 3. Save the segmentation mask tensor to a file
        segmentation_mask_tensor_file_path = os.path.join(self.processed_segmentation_path, f"{video_id[:-4]}.pkl")
        with open(segmentation_mask_tensor_file_path, 'wb') as file:
            torch.save(segmentation_mask_tensor, file)

    def process_all_videos(self, phase):
        """
		Process all videos in the specified phase (train/test) and save the segmentation masks.
		"""
        video_ids = os.listdir(os.path.join(self.segmentation_root_dir, phase, "predcls"))
        for video_id in tqdm(video_ids):
            self.construct_video_segmentation_tensor(video_id, phase)
            print(f"Processed {video_id} for {phase} phase.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag/", help="Path to the SAM2 masks file")
    args = parser.parse_args()

    sam2_processor = SAM2Processor(ag_root_directory=args.ag_root_directory)
    sam2_processor.process_all_videos(phase="train")
    sam2_processor.process_all_videos(phase="test")


if __name__ == "__main__":
    main()
