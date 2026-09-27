"""Make the repo root importable so tests can reach `scripts/`, and each model folder's modules.

Model folders have no __init__.py (a `tabpfn/` package would shadow the installed tabpfn
library), so their modules are imported by name, e.g. `from pfn_metrics import ...`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tabpfn"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
