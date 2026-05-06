"""
W-DSGDetr Loss — reuses the LKS Buffer loss pattern.
======================================================

Same bucketed VLM noisy-label training as LKSLoss:
  Vis-Vis: full loss on clean manual labels
  Vis-Unseen / Unseen-Unseen: λ_vlm weighted, smoothed
"""

from lib.supervised.worldsgg.lks_buffer.loss import LKSLoss

# W-DSGDetr uses the same loss function as LKS Buffer.
# Aliased here for clean per-method imports.
WDSGDetrLoss = LKSLoss
