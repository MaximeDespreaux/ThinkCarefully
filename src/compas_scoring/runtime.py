"""Process setup that must run before any numerical library is imported.

XGBoost and PyTorch each ship their own OpenMP runtime. On macOS, loading both into one
process and then letting both spin up thread pools deadlocks: the interpreter parks at 0%
CPU with no exception and no traceback, and because the block is inside a C-level lock,
even SIGALRM cannot break it. That is exactly what happens when the tuned tree model and a
foundation model are fitted in the same script.

Pinning every native library to a single thread avoids the collision. The cost is close to
zero here -- the modelling table is 4,320 x 10 -- and it buys determinism, which matters
more for a deliverable whose whole argument is reproducibility. Cross-validation keeps its
parallelism at the process level via joblib, where it does not interact with OpenMP.

Import this first, before anything else:

    from compas_scoring import runtime  # noqa: F401  (must precede numeric imports)
"""

from __future__ import annotations

import os

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_configured = False


def configure(threads: int = 1) -> None:
    """Pin native thread pools. Idempotent, and a no-op if the user set the vars already."""
    global _configured
    if _configured:
        return

    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(threads))
    # Tolerate the second OpenMP runtime rather than aborting when both are present.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # TabPFN and TabICL are CPU-bound here; keep HF quiet about anonymous downloads.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:  # pragma: no cover - torch absent or already configured
        pass

    _configured = True


configure()
