import json
import numpy as np
import torch
from matplotlib import pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)


def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().tolist()
    return tensor


# --- Visualization Helper ---
def get_color_map(num_colors):
    if num_colors <= 0: return []
    """Generates a list of distinct colors for visualization."""
    colors = plt.cm.get_cmap('hsv', num_colors)
    return [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors(range(num_colors))]