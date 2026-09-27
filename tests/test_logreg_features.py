"""Guardrails on the white-box feature engineering in logreg/features.py."""

from __future__ import annotations

from compas_scoring.data import build_dataset
from logreg.features import engineered


def test_engineered_features_are_additive_and_finite():
    data = build_dataset("race_aware")
    out = engineered(data.X)
    assert set(data.X.columns) <= set(out.columns)
    assert out.notna().all().all()
    assert (out["log_priors"] >= 0).all()
    assert out["priors_capped"].max() <= 10
