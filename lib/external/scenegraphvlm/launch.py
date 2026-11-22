"""
Run the vendored ``infer_swift_gen_prompt.py`` unmodified, with one compatibility shim.

ms-swift 4.5 can no longer infer ``model_type`` from the released checkpoint directory
(``Failed to automatically match model_type ... ['qwen3_5', 'ovis_ocr2', 'qwen3_5_emb',
'wemm_embedding']``); the authors' container used an older ms-swift where the match was
unique.  The checkpoint's config is ``Qwen3_5ForConditionalGeneration`` /
``model_type: qwen3_5``, so we pin ``model_type='qwen3_5'`` in ``swift.model.get_processor``
/ ``get_model_processor`` before the driver imports them.  Nothing else is touched.

    python lib/external/scenegraphvlm/launch.py <driver args...>
"""
import os
import runpy
import sys

import swift.model as _sm

_MODEL_TYPE = os.environ.get("SGVLM_MODEL_TYPE", "qwen3_5")
_orig_gp, _orig_gmp = _sm.get_processor, _sm.get_model_processor


def _gp(*a, **k):
    k.setdefault("model_type", _MODEL_TYPE)
    return _orig_gp(*a, **k)


def _gmp(*a, **k):
    k.setdefault("model_type", _MODEL_TYPE)
    return _orig_gmp(*a, **k)


_sm.get_processor, _sm.get_model_processor = _gp, _gmp

if __name__ == "__main__":
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer_swift_gen_prompt.py")
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
