"""Backward-compatible shim: "WorldFormer C2" became WorldWise++.

The trainable method lives in ``lib.supervised.worldwise_pp`` (``WorldWisePP``,
``WorldWisePPLoss``, ``WorldAGGrid``). The original stand-alone scaffold
(``WorldFormerC2`` / ``WorldFormerC2Loss``, exercised by
tests/test_worldformer_c2_smoke.py) is kept here unchanged for reference.
"""
from lib.supervised.worldwise_pp.model import WorldWisePP  # noqa: F401
from lib.supervised.worldwise_pp.loss import WorldWisePPLoss  # noqa: F401
from .model import WorldFormerC2, WorldFormerC2Loss, hungarian_match  # noqa: F401
