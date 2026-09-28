"""COMPAS scoring: LogReg vs XGBoost vs TabPFN. Run with `make app`."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="COMPAS model comparison", page_icon="⚖️", layout="wide")

pages = st.navigation(
    [
        st.Page("pages/overview.py", title="Overview", default=True),
        st.Page("pages/performance.py", title="Performance"),
        st.Page("pages/interpretability.py", title="Interpretability"),
        st.Page("pages/stability.py", title="Stability"),
        st.Page("pages/fairness.py", title="Fairness"),
    ]
)
with st.sidebar:
    st.caption(
        "COMPAS two-year cohort (6,172 defendants, Broward County 2013-14). "
        "Three models compared on one shared test set; the COMPAS tool's own "
        "medium/high-risk decision is the benchmark. ISAF Group 9, HEC Paris."
    )
pages.run()
