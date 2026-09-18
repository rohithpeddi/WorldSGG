"""Backward-compatible shim: "WorldFormer C1" was renamed WorldWise+.

The model now lives in ``lib.supervised.worldwise_plus``; running jobs and the
scoring scripts that still import ``WorldFormerC1`` from here keep working.
"""
from lib.supervised.worldwise_plus.model import (  # noqa: F401
    GatedFusionProjector, WorldWisePlus, WorldFormerC1,
)
