"""Test logreg/features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compas_scoring.data import build_dataset
from logreg.features import engineered


class TestEngineered:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "Number_of_Priors": [0, 1, 4, 25],
                "Age_Below_TwentyFive": [1, 0, 1, 0],
                "Female": [0, 1, 0, 1],
            }
        )

    def test_adds_the_four_terms_after_the_original_columns(self, frame):
        out = engineered(frame)
        assert list(out.columns) == [
            *frame.columns,
            "log_priors",
            "priors_capped",
            "no_priors",
            "young_x_log_priors",
        ]

    def test_log_priors_is_log1p(self, frame):
        np.testing.assert_allclose(engineered(frame)["log_priors"], np.log1p([0, 1, 4, 25]))

    def test_priors_capped_clips_at_ten(self, frame):
        assert engineered(frame)["priors_capped"].tolist() == [0, 1, 4, 10]

    def test_no_priors_flags_exactly_zero(self, frame):
        assert engineered(frame)["no_priors"].tolist() == [1.0, 0.0, 0.0, 0.0]

    def test_interaction_is_zero_for_defendants_over_twenty_five(self, frame):
        out = engineered(frame)
        expected = frame["Age_Below_TwentyFive"] * np.log1p(frame["Number_of_Priors"])
        np.testing.assert_allclose(out["young_x_log_priors"], expected)
        assert (out.loc[frame["Age_Below_TwentyFive"] == 0, "young_x_log_priors"] == 0).all()

    def test_interaction_is_skipped_without_the_age_dummy(self, frame):
        out = engineered(frame.drop(columns="Age_Below_TwentyFive"))
        assert "young_x_log_priors" not in out.columns
        assert {"log_priors", "priors_capped", "no_priors"} <= set(out.columns)

    def test_does_not_mutate_its_input(self, frame):
        before = frame.copy()
        engineered(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_is_additive_and_finite_on_the_real_race_aware_set(self):
        data = build_dataset("race_aware")
        out = engineered(data.X)
        assert set(data.X.columns) <= set(out.columns)
        assert out.notna().all().all()
        assert (out["log_priors"] >= 0).all()
        assert out["priors_capped"].max() <= 10

    def test_preserves_the_row_index(self, frame):
        frame.index = [10, 20, 30, 40]
        assert list(engineered(frame).index) == [10, 20, 30, 40]
