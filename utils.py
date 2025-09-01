import json
import numpy as np
import torch


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