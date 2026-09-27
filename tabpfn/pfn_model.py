"""TabPFN v2 (Hollmann et al., Nature 2025), wrapped as an ordinary sklearn classifier.

TabPFN is a transformer pre-trained on millions of synthetic tables. It has no weights to fit
on COMPAS: ``fit`` stores the training rows, and ``predict_proba`` runs one forward pass that
reads them as context (in-context learning). Two consequences worth keeping in mind:

* there are **no estimated parameters** to compare between two fits -- the pre-trained weights
  are frozen -- so "distance between estimated params" must be measured on outputs instead
  (predictions, explanations, metrics); see README.md, stability;
* every prediction re-reads the whole training set, so anything that queries the model many
  times (SHAP, LIME, PDP/ICE, permutation importance, XPER) is slow.

Speed, measured on this project (8-core CPU, 4,320 training rows, 300 rows predicted):

    threads  fit_mode             fit      predict 300 rows
    1        fit_preprocessors      4.6 s  396 s
    8        fit_preprocessors      4.2 s  169 s
    8        fit_with_cache       150.5 s   10.7 s     <- default here

``fit_with_cache`` encodes the training context once at fit time, so every later
``predict_proba`` call is cheap -- exactly what the explanation methods need. Predictions are
identical across all four settings.

Threads: ``import compas_scoring`` pins every numeric library to one thread, to avoid a known
deadlock when XGBoost and PyTorch share a process. This adapter raises torch's thread count
back up for its own calls; do not load XGBoost in the same process as a TabPFN run.

No hyperparameter tuning: TabPFN is used with its defaults, which is the model's claim.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from compas_scoring.config import CONFIG


class TabPFNModel(ClassifierMixin, BaseEstimator):
    """DataFrame-friendly adapter around ``tabpfn.TabPFNClassifier``.

    ClassifierMixin must precede BaseEstimator: sklearn resolves ``__sklearn_tags__`` by MRO,
    and the other order makes ``is_classifier`` false, which breaks permutation_importance
    and the sklearn scorers with a confusing error.
    """

    def __init__(
        self,
        random_state: int = CONFIG.random_state,
        n_estimators: int = 8,
        fit_mode: str = "fit_with_cache",
        n_threads: int | None = None,
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.fit_mode = fit_mode
        self.n_threads = n_threads

    def _use_threads(self) -> None:
        import torch

        torch.set_num_threads(self.n_threads or os.cpu_count() or 1)

    @staticmethod
    def _as_array(X) -> np.ndarray:
        return X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier

        self._use_threads()
        self.feature_names_in_ = np.asarray(X.columns) if isinstance(X, pd.DataFrame) else None
        self.classes_ = np.unique(np.asarray(y))
        # ignore_pretraining_limits: tabpfn refuses >1,000 rows on CPU by default, purely as a
        # speed guard. Our <=4,937 training rows x 10 features are well inside the sizes the
        # model was pre-trained on, so lifting the guard costs time, not validity.
        self.estimator_ = TabPFNClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            device="cpu",
            fit_mode=self.fit_mode,
            ignore_pretraining_limits=True,
        )
        self.estimator_.fit(self._as_array(X), np.asarray(y))
        return self

    def predict_proba(self, X):
        self._use_threads()
        return self.estimator_.predict_proba(self._as_array(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_model(random_state: int = CONFIG.random_state) -> TabPFNModel:
    return TabPFNModel(random_state=random_state)
