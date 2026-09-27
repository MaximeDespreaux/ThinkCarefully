"""Shared fixtures for the XGBoost scripts.

``xgboost/`` is a folder of scripts, not a package, so it is put on ``sys.path`` here to make
``import xgb_model`` / ``import performance`` work the same way they do in the notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "xgboost"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from xgboost import XGBClassifier  # noqa: E402

import performance  # noqa: E402
import xgb_model  # noqa: E402

N_SMALL_TRAIN = 400
N_SMALL_TEST = 200


@pytest.fixture(scope="session")
def split():
    """The project's real shared train/test split (race-aware features)."""
    return xgb_model.load_data("race_aware")


@pytest.fixture(scope="session")
def small_train(split):
    train, _ = split
    return train.subset(train.X.index[:N_SMALL_TRAIN])


@pytest.fixture(scope="session")
def small_test(split):
    _, test = split
    return test.subset(test.X.index[:N_SMALL_TEST])


@pytest.fixture(scope="session")
def small_model(small_train):
    """A quick, untuned model: enough to exercise the code paths, not to judge performance."""
    model = XGBClassifier(n_estimators=20, max_depth=2, random_state=0, n_jobs=1)
    return model.fit(small_train.X, small_train.y)


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Redirect every save/load to a temporary folder so tests never touch xgboost/models/.

    ``performance`` does ``from xgb_model import MODEL_DIR``, so it holds its own reference
    and has to be patched separately.
    """
    target = tmp_path / "models"
    monkeypatch.setattr(xgb_model, "MODEL_DIR", target)
    monkeypatch.setattr(performance, "MODEL_DIR", target)
    return target
