"""Import paths for every model folder, and shared fixtures for the XGBoost scripts.

* the repo root, so ``import logreg`` (a package) and ``scripts/`` work;
* ``tabpfn/`` and ``xgboost/``, folders of scripts rather than packages, so their modules import
  by name (``import pfn_metrics``, ``import xgb_model``) the same way they do when run directly.
  Neither folder has an ``__init__.py``, so the installed ``tabpfn`` and ``xgboost`` libraries
  still win over the folders of the same name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "tabpfn", ROOT / "xgboost"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import performance  # noqa: E402
import xgb_model  # noqa: E402

from xgboost import XGBClassifier  # noqa: E402

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
