"""XPER subsampling, wiring, saving and loading in xgboost/performance.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import performance
from performance import (
    calculate_xper,
    load_xper,
    save_xper,
    scores,
    xper_paths,
    xper_sample,
)

# XPER's AUC cost grows ~N^2-N^3 in rows and linearly in coalitions, so the real-XPER tests
# stay tiny: they check wiring and the decomposition, not the real values.
XPER_ROWS = 60
XPER_COALITIONS = 30


class FakeModelPerformance:
    """Records how XPER is called and returns random values of the right shape."""

    calls: list[dict] = []

    def __init__(self, X_train, y_train, X_test, y_test, model, **kwargs):
        self.n, self.p = X_test.shape
        FakeModelPerformance.calls.append(
            {
                "X_test": X_test,
                "y_test": y_test,
                "model": model,
                "n_jobs": model.get_params()["n_jobs"],
                **kwargs,
            }
        )

    def calculate_XPER_values(self, Eval_Metric, **kwargs):
        FakeModelPerformance.calls[-1].update(Eval_Metric=Eval_Metric, **kwargs)
        rng = np.random.default_rng(0)
        return rng.normal(size=self.p + 1), rng.normal(size=(self.n, self.p + 1))


@pytest.fixture
def fake_xper(monkeypatch):
    FakeModelPerformance.calls = []
    monkeypatch.setattr(performance, "ModelPerformance", FakeModelPerformance)
    return FakeModelPerformance


@pytest.fixture
def xper_frames(small_test):
    columns = ["benchmark", *small_test.X.columns]
    rng = np.random.default_rng(1)
    phi = pd.Series(rng.normal(size=len(columns)), index=columns, name="xper_auc")
    phi_i = pd.DataFrame(
        rng.normal(size=(len(small_test), len(columns))), index=small_test.X.index, columns=columns
    )
    return phi, phi_i


# --- xper_sample() -------------------------------------------------------------------------


def test_sample_size_and_origin(small_test):
    sample = xper_sample(small_test, n=50)

    assert len(sample) == 50
    assert sample.X.index.isin(small_test.X.index).all()
    assert sample.X.index.is_unique
    assert sample.X.index.is_monotonic_increasing


def test_sample_keeps_rows_aligned(small_test):
    """X, y, incumbent and groups must still describe the same defendants."""
    sample = xper_sample(small_test, n=50)

    for part in (sample.y, sample.incumbent, sample.groups):
        assert part.index.equals(sample.X.index)
    pd.testing.assert_series_equal(sample.y, small_test.y.loc[sample.X.index])


def test_sample_is_stratified(split):
    _, test = split
    sample = xper_sample(test, n=400)

    assert sample.base_rate == pytest.approx(test.base_rate, abs=0.005)


def test_sample_is_seeded(small_test):
    first = xper_sample(small_test, n=50, seed=1)

    assert first.X.index.equals(xper_sample(small_test, n=50, seed=1).X.index)
    assert not first.X.index.equals(xper_sample(small_test, n=50, seed=2).X.index)


def test_sample_rejects_n_at_least_test_size(small_test):
    with pytest.raises(ValueError):
        xper_sample(small_test, n=len(small_test))


# --- calculate_xper() wiring (fast, fake XPER) ---------------------------------------------


def test_xper_uses_every_row_it_is_given(fake_xper, small_model, small_train, small_test):
    """XPER silently subsamples to 500 rows unless sample_size covers the whole input."""
    calculate_xper(small_model, small_train, small_test)
    call = fake_xper.calls[-1]

    assert call["sample_size"] == len(small_test)
    assert len(call["y_test"]) == len(small_test)


def test_xper_arguments(fake_xper, small_model, small_train, small_test):
    calculate_xper(small_model, small_train, small_test, seed=7, n_coalitions=25)
    call = fake_xper.calls[-1]

    assert call["model"] is small_model
    assert call["Eval_Metric"] == ["AUC"]
    assert call["kernel"] is True
    assert call["seed"] == 7
    assert call["N_coalition_sampled"] == 25
    assert isinstance(call["X_test"], np.ndarray)


def test_xper_default_coalitions_left_to_xper(fake_xper, small_model, small_train, small_test):
    calculate_xper(small_model, small_train, small_test)

    assert fake_xper.calls[-1]["N_coalition_sampled"] is None


def test_xper_runs_model_single_threaded(fake_xper, small_train, small_test):
    """XPER already runs 60 threads; a multi-threaded model on top oversubscribes the CPU."""
    from xgboost import XGBClassifier

    model = XGBClassifier(n_estimators=5, max_depth=2, n_jobs=4).fit(small_train.X, small_train.y)
    calculate_xper(model, small_train, small_test)

    assert fake_xper.calls[-1]["n_jobs"] == 1


def test_xper_output_labels(fake_xper, small_model, small_train, small_test):
    phi, phi_i = calculate_xper(small_model, small_train, small_test)
    columns = ["benchmark", *small_test.X.columns]

    assert list(phi.index) == columns
    assert phi.name == "xper_auc"
    assert list(phi_i.columns) == columns
    assert phi_i.index.equals(small_test.X.index)


def test_xper_on_subsample_keeps_subsample_index(fake_xper, small_model, small_train, small_test):
    sample = xper_sample(small_test, n=50)
    _, phi_i = calculate_xper(small_model, small_train, sample)

    assert phi_i.index.equals(sample.X.index)


# --- save / load ---------------------------------------------------------------------------


def test_save_load_round_trip(model_dir, xper_frames):
    phi, phi_i = xper_frames
    model_dir.mkdir()
    save_xper(phi, phi_i, "race_aware")
    phi_loaded, phi_i_loaded = load_xper("race_aware")

    pd.testing.assert_series_equal(phi_loaded, phi)
    pd.testing.assert_frame_equal(phi_i_loaded, phi_i)


def test_save_creates_missing_folder(model_dir, xper_frames):
    """save_model creates models/ when it is missing; save_xper should too."""
    phi, phi_i = xper_frames
    save_xper(phi, phi_i, "race_aware")

    assert all(path.is_file() for path in xper_paths("race_aware"))


def test_xper_paths_naming(model_dir):
    global_path, individual_path = xper_paths("selected")

    assert global_path == model_dir / "xper_selected.csv"
    assert individual_path == model_dir / "xper_selected_individual.csv"


# --- real XPER on a tiny sample (slow) -----------------------------------------------------


@pytest.fixture(scope="module")
def xper_subsample(small_test):
    return xper_sample(small_test, n=XPER_ROWS)


@pytest.fixture(scope="module")
def real_xper(small_model, small_train, xper_subsample):
    return calculate_xper(
        small_model, small_train, xper_subsample, seed=42, n_coalitions=XPER_COALITIONS
    )


@pytest.mark.slow
def test_xper_approximately_sums_to_auc(real_xper, small_model, xper_subsample):
    """Kernel XPER fits an unconstrained regression on sampled coalitions, so benchmark +
    contributions only approximate the AUC of the rows it was computed on."""
    phi, _ = real_xper
    auc = roc_auc_score(xper_subsample.y, scores(small_model, xper_subsample))

    assert phi.sum() == pytest.approx(auc, abs=0.03)


@pytest.mark.slow
def test_xper_individual_values_average_to_global(real_xper):
    """Exact, not approximate: the regression is linear in the per-individual metric."""
    phi, phi_i = real_xper

    pd.testing.assert_series_equal(phi_i.mean(), phi, check_names=False, rtol=0, atol=1e-10)


@pytest.mark.slow
def test_xper_is_reproducible(real_xper, small_model, small_train, xper_subsample):
    phi, _ = real_xper
    phi_again, _ = calculate_xper(
        small_model, small_train, xper_subsample, seed=42, n_coalitions=XPER_COALITIONS
    )

    pd.testing.assert_series_equal(phi_again, phi)


# --- main() --------------------------------------------------------------------------------


def test_main_computes_saves_and_reports(model_dir, fake_xper, small_model, monkeypatch, capsys):
    """Smoke test of the script: the small model stands in for the saved one and a fake for
    XPER, so the sampling, wiring and saving still run for real.

    ``performance`` does ``from xgb_model import load_model``, so it is patched there.
    """
    monkeypatch.setattr(performance, "load_model", lambda feature_set: small_model)

    performance.main("race_aware")

    assert all(path.is_file() for path in xper_paths("race_aware"))
    call = fake_xper.calls[-1]
    assert call["N_coalition_sampled"] == 300
    assert call["sample_size"] == 400
    out = capsys.readouterr().out
    assert "COMPAS (incumbent)" in out
    assert "Confusion matrix" in out
