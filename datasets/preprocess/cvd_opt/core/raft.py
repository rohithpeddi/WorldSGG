# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""RAFT network for MegaSaM."""

from .corr import AlternateCorrBlock, CorrBlock
from .extractor import BasicEncoder, SmallEncoder
import torch
from torch import nn
import torch.nn.functional as F
from .update import BasicUpdateBlock
from .update import SmallUpdateBlock
from .utils.utils import coords_grid, upflow8

try:
  autocast = torch.cuda.amp.autocast
except:  # pylint: disable=bare-except
  # dummy autocast for PyTorch < 1.6
  class autocast:  # pylint: disable=invalid-name

    def __init__(self, enabled):
      pass

    def __enter__(self):
      pass

    def __exit__(self, *args):
      pass


class RAFT(nn.Module):
  """RAFT network for MegaSaM."""

  def __init__(self, args):
    super(RAFT, self).__init__()
    self.args = args
    self.mixed_precision = True
    if args.small:
      self.hidden_dim = hdim = 96
      self.context_dim = cdim = 64
      args.corr_levels = 4
      args.corr_radius = 3

    else:
      self.hidden_dim = hdim = 128
      self.context_dim = cdim = 128
      args.corr_levels = 4
      args.corr_radius = 4

    if 'dropout' not in self.args:
      self.args.dropout = 0

    if 'alternate_corr' not in self.args:
      self.args.alternate_corr = False

    # feature network, context network, and update block
    if args.small:
      self.fnet = SmallEncoder(
          output_dim=128, norm_fn='instance', dropout=args.dropout
      )
      self.cnet = SmallEncoder(
          output_dim=hdim + cdim, norm_fn='none', dropout=args.dropout
      )
      self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

    else:
      self.fnet = BasicEncoder(
          output_dim=256, norm_fn='instance', dropout=args.dropout
      )
      self.cnet = BasicEncoder(
          output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout
      )
      self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

  def freeze_bn(self):
    for m in self.modules():
      if isinstance(m, nn.BatchNorm2d):
        m.eval()

  def initialize_flow(self, img):
    """Flow is represented as difference between two coordinate grids flow = coords1 - coords0."""
    # pylint: disable=invalid-name
    N, _, H, W = img.shape
    coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
    coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

    # optical flow computed as difference: flow = coords1 - coords0
    return coords0, coords1

  def upsample_flow(self, flow, mask):
    """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination."""
    # pylint: disable=invalid-name
    N, _, H, W = flow.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * flow, [3, 3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 2, 8 * H, 8 * W)

  def forward(
      self,
      image1,
      image2,
      iters=12,
      flow_init=None,
      upsample=True,
      test_mode=False,
  ):
    """Estimate optical flow between pair of frames."""

    image1 = 2 * (image1 / 255.0) - 1.0
    image2 = 2 * (image2 / 255.0) - 1.0

    image1 = image1.contiguous()
    image2 = image2.contiguous()

    hdim = self.hidden_dim
    cdim = self.context_dim

    # run the feature network
    with autocast(enabled=self.mixed_precision):
      fmap1, fmap2 = self.fnet([image1, image2])

    fmap1 = fmap1.float()
    fmap2 = fmap2.float()
    if self.args.alternate_corr:
      corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
    else:
      corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

    # run the context network
    with autocast(enabled=self.mixed_precision):
      cnet = self.cnet(image1)
      net, inp = torch.split(cnet, [hdim, cdim], dim=1)
      net = torch.tanh(net)
      inp = torch.relu(inp)

    coords0, coords1 = self.initialize_flow(image1)

    if flow_init is not None:
      coords1 = coords1 + flow_init

    flow_predictions = []
    flow_up = None
    for _ in range(iters):
      coords1 = coords1.detach()
      corr = corr_fn(coords1)  # index correlation volume

      flow = coords1 - coords0
      with autocast(enabled=self.mixed_precision):
        net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

      # F(t+1) = F(t) + \Delta(t)
      coords1 = coords1 + delta_flow

      # upsample predictions
      if up_mask is None:
        flow_up = upflow8(coords1 - coords0)
      else:
        flow_up = self.upsample_flow(coords1 - coords0, up_mask)

      flow_predictions.append(flow_up)

    if test_mode:
      if flow_up is None:
        raise ValueError('flow_up is None')
      return coords1 - coords0, flow_up, net

    return flow_predictions
