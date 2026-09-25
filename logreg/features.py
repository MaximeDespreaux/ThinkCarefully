"""Feature engineering specific to the logistic regression (white-box) model."""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineered(X: pd.DataFrame) -> pd.DataFrame:
    """Extra terms for the white-box model.

    Priors is the only genuinely continuous predictor and its effect on recidivism is
    strongly concave, so a raw linear term underfits the low end and overstates the tail.
    The log term and the age interaction give logistic regression a fair shot at the
    non-linearity the tree model gets for free.
    """
    out = X.copy()
    priors = out["Number_of_Priors"]
    out["log_priors"] = np.log1p(priors)
    out["priors_capped"] = priors.clip(upper=10)
    out["no_priors"] = (priors == 0).astype(float)
    if "Age_Below_TwentyFive" in out:
        out["young_x_log_priors"] = out["Age_Below_TwentyFive"] * out["log_priors"]
    return out
