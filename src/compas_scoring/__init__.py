"""COMPAS scoring analysis: white-box vs ML vs tabular foundation model.

Compared across the four trustworthy-AI dimensions -- predictive performance (statistical
and economic), interpretability, stability and fairness.
"""

# Pin native thread pools before any numeric library loads. See runtime.py: without
# this, XGBoost and torch deadlock when used in the same process on macOS.
from compas_scoring import runtime as runtime  # noqa: I001  (must be imported first)
from compas_scoring.config import CONFIG, Config, load_config, project_root

__all__ = ["CONFIG", "Config", "load_config", "project_root", "runtime"]
__version__ = "0.1.0"
