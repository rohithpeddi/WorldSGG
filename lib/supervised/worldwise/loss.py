"""
WorldWise Loss — reuses the AMWAE loss pattern with reconstruction.
=====================================================================

Same as AMWAELoss: bucketed VLM noisy-label training + reconstruction loss.
"""

from lib.supervised.worldsgg.amwae.loss import AMWAELoss

# WorldWise uses the same loss as AMWAE (includes reconstruction component).
WorldWiseLoss = AMWAELoss
