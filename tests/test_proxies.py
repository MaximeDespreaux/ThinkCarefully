"""The race-and-proxy feature sets in pyproject.toml must match what the data implies."""

from __future__ import annotations

from compas_scoring.config import CONFIG
from compas_scoring.proxies import race_leakage, race_proxies


def _race_aware_without(*removed: str) -> list[str]:
    drop = set(CONFIG.race_dummies) | set(removed)
    return [f for f in CONFIG.features("race_aware") if f not in drop]


def test_priors_is_the_dominant_race_proxy():
    assert "Number_of_Priors" in race_proxies()


def test_race_proxy_blind_drops_exactly_race_and_every_proxy():
    """Fails if the committed list drifts from the proxies the data flags."""
    assert CONFIG.features("race_proxy_blind") == _race_aware_without(*race_proxies())


def test_race_priors_blind_drops_race_and_priors_only():
    assert CONFIG.features("race_priors_blind") == _race_aware_without("Number_of_Priors")


def test_removing_the_proxies_removes_most_race_information():
    """Race is predictable from the race-blind set, but barely from the proxy-free one."""
    assert race_leakage("race_blind") > 0.62
    assert race_leakage("race_proxy_blind") < 0.56
