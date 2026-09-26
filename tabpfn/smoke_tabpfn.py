"""Gate: prove TabPFN installs, downloads its weights and predicts. Run via `make tabpfn-smoke`.

TabPFN downloads pretrained weights from HuggingFace on first fit. If that fails -- no
network, changed API, a new licence gate -- we need to know before building an analysis
that depends on it. pyproject pins tabpfn 2.2.1 because tabpfn >=9 refuses to download
weights without a Prior Labs account.
"""

from __future__ import annotations

# isort: split
import compas_scoring  # noqa: F401  (pins thread pools before numpy and torch load)

# isort: split
import sys
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

MIN_AUC = 0.80  # easy synthetic data; anything lower means the model is not really working


def main() -> int:
    try:
        import torch

        import tabpfn
        from tabpfn import TabPFNClassifier
    except Exception as exc:  # pragma: no cover - environment probe
        print(f"FAIL: import error: {type(exc).__name__}: {exc}")
        return 1
    print(f"tabpfn {tabpfn.__version__}   torch {torch.__version__}")

    X, y = make_classification(n_samples=700, n_features=10, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print("Fitting on 490 synthetic rows (the first run also downloads the weights)...")
    start = time.perf_counter()
    try:
        proba = TabPFNClassifier(random_state=42).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.perf_counter() - start

    auc = roc_auc_score(y_test, proba)
    print(f"fit + predict: {elapsed:.1f}s   AUC: {auc:.4f}")
    if not np.isfinite(proba).all() or auc < MIN_AUC:
        print("FAIL: non-finite probabilities or AUC below the sanity floor")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
