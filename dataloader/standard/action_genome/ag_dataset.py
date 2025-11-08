from dataloader.base_ag_dataset import BaseAG


class StandardAG(BaseAG):

    def __init__(self, phase, mode, datasize, data_path=None, filter_nonperson_box_frame=True, filter_small_box=False):
        super().__init__(phase, mode, datasize, data_path, filter_nonperson_box_frame, filter_small_box)
        print("Total number of ground truth annotations: ", len(self.gt_annotations))

    def __getitem__(self, index):
        frame_names = self.video_list[index]
        video_id = frame_names[0].split("/")[0]
        return {
            "video_id": video_id,
            "frame_names": frame_names,
            "gt_annotations": self.gt_annotations[index]
        }

def cuda_collate_fn(batch):
    return batch[0]
