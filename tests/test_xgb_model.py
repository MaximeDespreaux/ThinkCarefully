"""Training, saving and loading in xgboost/xgb_model.py, and the split it trains on."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import xgb_model
from compas_scoring.config import CONFIG
from compas_scoring.data import build_dataset
from xgb_model import (
    SEARCH_SPACE,
    load_data,
    load_metadata,
    load_model,
    model_paths,
    save_model,
    tune_xgboost,
)

META_KEYS = {
    "feature_set",
    "features",
    "best_params",
    "cv_auc",
    "cv_folds",
    "test_size",
    "random_state",
}


@pytest.fixture
def fake_search(small_model):
    """Stands in for a fitted RandomizedSearchCV: save_model only reads these attributes."""
    return SimpleNamespace(
        best_estimator_=small_model,
        best_params_={"max_depth": 2, "learning_rate": 0.1},
        best_score_=np.float64(0.71),
    )


# --- save / load ---------------------------------------------------------------------------


def test_save_load_round_trip(model_dir, fake_search, small_model, small_test):
    save_model(fake_search, "race_aware")
    reloaded = load_model("race_aware")

    np.testing.assert_array_equal(
        reloaded.predict_proba(small_test.X), small_model.predict_proba(small_test.X)
    )


def test_save_creates_missing_folder(model_dir, fake_search):
    assert not model_dir.exists()
    path = save_model(fake_search, "race_aware")

    assert path.is_file()
    assert path.parent == model_dir


def test_metadata_contents(model_dir, fake_search):
    save_model(fake_search, "race_aware")
    _, meta_path = model_paths("race_aware")
    meta = load_metadata("race_aware")

    json.loads(meta_path.read_text())  # valid JSON on disk
    assert set(meta) == META_KEYS
    assert meta["feature_set"] == "race_aware"
    assert meta["features"] == CONFIG.features("race_aware")
    assert meta["best_params"] == fake_search.best_params_
    assert meta["cv_auc"] == pytest.approx(0.71)


def test_load_model_missing_file(model_dir):
    with pytest.raises(FileNotFoundError, match="xgb_model.py"):
        load_model("selected")


def test_model_paths_naming(model_dir):
    model_path, meta_path = model_paths("selected")

    assert model_path == model_dir / "xgb_selected.json"
    assert meta_path == model_dir / "xgb_selected.meta.json"


# --- the shared split (leakage guard) ------------------------------------------------------


def test_split_is_disjoint_and_complete(split):
    train, test = split
    full = build_dataset("race_aware")

    assert train.X.index.intersection(test.X.index).empty
    assert train.X.index.union(test.X.index).equals(full.X.index.sort_values())


def test_split_test_share(split):
    train, test = split

    assert len(test) / (len(train) + len(test)) == pytest.approx(CONFIG.test_size, abs=0.001)


def test_split_is_stratified(split):
    train, test = split

    assert train.base_rate == pytest.approx(test.base_rate, abs=0.01)


def test_no_target_or_incumbent_in_features(split):
    train, test = split

    for data in (train, test):
        assert CONFIG.target not in data.X.columns
        assert CONFIG.incumbent not in data.X.columns


def test_split_is_deterministic(split):
    train, test = split
    train_again, test_again = load_data("race_aware")

    assert train.X.index.equals(train_again.X.index)
    assert test.X.index.equals(test_again.X.index)


# --- tuning --------------------------------------------------------------------------------


@pytest.mark.slow
def test_tuned_params_come_from_search_space(small_train):
    search = tune_xgboost(small_train.X, small_train.y, n_iter=2)

    for name, value in search.best_params_.items():
        assert value in SEARCH_SPACE[name]


@pytest.mark.slow
def test_tuning_is_reproducible(small_train):
    first = tune_xgboost(small_train.X, small_train.y, n_iter=2)
    second = tune_xgboost(small_train.X, small_train.y, n_iter=2)

    assert first.best_params_ == second.best_params_
    assert first.best_score_ == pytest.approx(second.best_score_)


# --- the saved artifact in xgboost/models/ -------------------------------------------------


@pytest.fixture
def saved_model():
    model_path, _ = model_paths("race_aware")
    if not model_path.is_file():
        pytest.skip("no saved model - run `python xgboost/xgb_model.py` first")
    return load_model("race_aware"), load_metadata("race_aware")


def test_saved_model_features_match_metadata(saved_model):
    model, meta = saved_model

    assert model.get_booster().feature_names == meta["features"]
    assert meta["features"] == CONFIG.features(meta["feature_set"])


def test_saved_model_beats_floor_and_compas(saved_model, split):
    model, _ = saved_model
    _, test = split
    auc = roc_auc_score(test.y, model.predict_proba(test.X)[:, 1])

    assert auc > 0.70
    assert auc > roc_auc_score(test.y, test.incumbent)


def test_xgb_model_dir_is_next_to_script():
    assert xgb_model.MODEL_DIR.name == "models"
    assert xgb_model.MODEL_DIR.parent.name == "xgboost"


# --- main() --------------------------------------------------------------------------------


def test_main_tunes_saves_and_reports(model_dir, fake_search, monkeypatch, capsys):
    """Smoke test of the script: the slow randomized search is replaced by a fake result."""
    monkeypatch.setattr(xgb_model, "tune_xgboost", lambda X, y: fake_search)

    xgb_model.main("race_aware")

    model_path, meta_path = model_paths("race_aware")
    assert model_path.is_file()
    assert meta_path.is_file()
    out = capsys.readouterr().out
    assert "Best CV AUC: 0.7100" in out
    assert f"Saved model to {model_path}" in out
