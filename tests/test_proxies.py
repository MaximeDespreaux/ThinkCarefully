"""Race leakage: the race-blind set still carries race; the proxy-free set much less."""

from __future__ import annotations

from compas_scoring.proxies import race_leakage


def test_dropping_the_race_columns_does_not_remove_race():
    assert race_leakage("race_blind") > 0.62


def test_dropping_race_and_its_proxies_removes_most_of_it():
    assert race_leakage("race_proxy_blind") < race_leakage("race_priors_blind")
    assert race_leakage("race_priors_blind") < race_leakage("race_blind")
