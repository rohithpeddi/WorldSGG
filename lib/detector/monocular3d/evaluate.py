#!/usr/bin/env python3
"""
Monocular3D Evaluation Entry Point

Usage:
    python -m lib.detector.monocular3d.evaluate \
        --checkpoint /path/to/checkpoint_XX \
        --data_path /path/to/Datasets/action_genome
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.detector.monocular3d.evaluation.evaluate_3d import main

if __name__ == "__main__":
    main()
