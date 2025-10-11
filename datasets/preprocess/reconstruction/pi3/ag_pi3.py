import os

import torch

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge


class AgPi3:

    def __init__(
            self,
            root_dir_path,
            output_dir_path=None,
    ):
        self.model = None
        self.root_dir_path = root_dir_path
        self.output_dir_path = output_dir_path if output_dir_path is not None else root_dir_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.load_model()

    def load_model(self):
        self.model = Pi3.from_pretrained("yyfz233/Pi3").to(self.device).eval()

    def preprocess_image_list(self, data_path, is_video=False):
        interval = 10 if is_video else 1
        print(f'Sampling interval: {interval}')
        imgs = load_images_as_tensor(data_path, interval=interval).to(self.device)  # (N, 3, H, W)
        return imgs

    def infer(self, video_id):
        data_path = f'{self.root_dir_path}/videos/{video_id}.mp4'
        video_save_dir = os.path.join(self.output_dir_path, "results", video_id)
        os.makedirs(video_save_dir, exist_ok=True)
        save_path = f'{video_save_dir}/{video_id}.ply'

        imgs = self.preprocess_image_list(data_path, is_video=True)
        print("Running model inference...")
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                res = self.model(imgs[None])

        # 4. Process mask
        masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]

        # 5. Save points
        print(f"Saving point cloud to: {save_path}")
        write_ply(res['points'][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], save_path)
        print("Done.")


def main():
    pass


if __name__ == "__main__":
    main()
