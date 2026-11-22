"""PUF (arXiv 2607.07170) adapted to monocular per-timestamp WSGG.

Pieces:
  puf_core  vendored PUF / FROSS math (Apache-2.0, see NOTICE)
  geometry  Pi3 lifting of 2D slots into Gaussians in the floor-aligned frame
  prior     class-conditional relation prior fitted on the AG training split
  fusion    the per-video causal graph (arms: lks, fross, puf, puf+prior)
The driver is tools/ext_puf.py; the design note is setup/EXT_PUF.md.
"""
