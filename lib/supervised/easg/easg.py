import torch
from torch.nn import Module, Linear


class EASG(Module):
    def __init__(self, verbs, objs, rels, dim_clip_feat=2304, dim_obj_feat=1024):
        super().__init__()
        self.fc_verb = Linear(dim_clip_feat, len(verbs))
        self.fc_objs = Linear(dim_obj_feat, len(objs))
        self.fc_rels = Linear(dim_clip_feat+dim_obj_feat, len(rels))

    def forward(self, clip_feat, obj_feats):
        out_verb = self.fc_verb(clip_feat)
        out_objs = self.fc_objs(obj_feats)

        clip_feat_expanded = clip_feat.expand(obj_feats.shape[0], -1)
        out_rels = self.fc_rels(torch.cat((clip_feat_expanded, obj_feats), dim=1))
        return out_verb, out_objs, out_rels


def intersect_2d(out, gt):
    return (out[..., None] == gt.T[None, ...]).all(1)