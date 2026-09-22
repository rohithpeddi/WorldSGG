"""lib.mllm package init.

Only job: make CUDA_VISIBLE_DEVICES readable by vLLM before any runner imports
it.  PBS Pro hands a job its GPUs as UUIDs::

    CUDA_VISIBLE_DEVICES=GPU-3dbb932a-e1cd-64f4-963c-cd6128360b36

which CUDA itself accepts, but vLLM 0.15's
``Platform.device_id_to_physical_device_id`` does ``int()`` on the entry and
dies with ``invalid literal for int() with base 10``.  vLLM reports that as
"Model architectures [...] failed to be inspected", so it looks like a model
problem rather than an environment one -- it hit every architecture we run
(2026-09-22, pragya).

Rewriting the UUIDs to the driver's own indices keeps the same physical cards
and costs one nvidia-smi call per process.  A numeric or empty value is left
alone, so CS93371 is unaffected.
"""

import os


def _normalize_cuda_visible_devices():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "GPU-" not in cvd:
        return

    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except Exception:
        return

    by_uuid = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            by_uuid[parts[1]] = parts[0]

    resolved = [by_uuid.get(d.strip(), d.strip()) for d in cvd.split(",") if d.strip()]
    if resolved and all(r.isdigit() for r in resolved):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(resolved)


_normalize_cuda_visible_devices()
