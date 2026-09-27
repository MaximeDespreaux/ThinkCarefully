"""T10. Marginal effects: TabPFN's stand-in for coefficients.

TabPFN has no coefficients. The marginal effect of a feature is the average change in
predicted risk when it moves by one unit, everything else held as observed:
    dummies   everybody at 1 minus everybody at 0 (for race and age: against the reference,
              Caucasian / 25-45, the other dummies of the group cleared)
    priors    one more prior than observed

Refits TabPFN once per run in _common.INTERPRET_RUNS and explains 500 test rows per run, in one
predict_proba call. Output: marginal_effects.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import pandas as pd
from pfn_interpret import marginal_effects


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "marginal_effects.csv"
    if c.cached(out, args.force):
        return

    frames = []
    for run, feature_set in c.INTERPRET_RUNS:
        model, spec = c.fit(run, feature_set)
        effects = marginal_effects(model, c.explain_sample(spec.test.X))
        effects.insert(0, "feature_set", feature_set)
        effects.insert(0, "run", run)
        frames.append(effects)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(out, index=False)

    wide = table.pivot_table(
        index="feature", columns=["run", "feature_set"], values="marginal_effect"
    )
    print(c.fmt(wide.reset_index(), 3))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
