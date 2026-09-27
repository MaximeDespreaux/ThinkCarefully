"""T16. XPER on cost: which feature earns (or loses) the money, at the 0.252 threshold.

XPER decomposes a performance metric exactly into a benchmark (no feature informative) plus
one Shapley contribution per feature: PM = phi_0 + sum_j phi_j. The earlier study ran it on
cost at its 0.21 threshold; this reruns it at the break-even threshold, fixed in advance.
The AUC decomposition does not depend on the threshold and is imported in artifacts/legacy/.

All 2^10 = 1,024 coalitions on 100 holdout defendants, each averaged over 10 background rows
from the training set (~1 M rows scored, ~1 h). The efficiency property is asserted to 1e-8.
Refits TabPFN once (holdout x race_aware). Output: xper_cost.csv.
"""

from __future__ import annotations

# isort: split
import _common as c

# isort: split
import time

from pfn_metrics import break_even_threshold
from pfn_xper import as_table, check_efficiency, cost_metric, decompose, sample

RUN, FEATURE_SET = "holdout", "race_aware"
N_EXPLAIN, N_BACKGROUND = 100, 10


def main() -> None:
    args = c.parse_args(__doc__.splitlines()[0])
    out = c.output_dir(__file__) / "xper_cost.csv"
    if c.cached(out, args.force):
        return

    model, spec = c.fit(RUN, FEATURE_SET)
    X_explain, y_explain = sample(spec.test.X, spec.test.y, N_EXPLAIN)
    X_background, _ = sample(spec.train.X, spec.train.y, N_BACKGROUND)

    done, total, start = [0], 2 ** X_explain.shape[1], time.perf_counter()

    def progress(n: int) -> None:
        done[0] += n
        elapsed = time.perf_counter() - start
        eta = elapsed / done[0] * (total - done[0])
        print(f"  {done[0]:4d}/{total} coalitions  {elapsed:5.0f}s  eta {eta:5.0f}s", flush=True)

    threshold = break_even_threshold()
    decomposition = decompose(
        lambda frame: model.predict_proba(frame)[:, 1],
        X_explain,
        y_explain,
        X_background,
        metric=cost_metric(threshold),
        progress=progress,
    )
    error = check_efficiency(decomposition)

    table = as_table(decomposition).assign(
        run=RUN,
        feature_set=FEATURE_SET,
        metric="cost_per_defendant",
        threshold=threshold,
        n_explain=N_EXPLAIN,
        n_background=N_BACKGROUND,
        efficiency_error=error,
        seconds=time.perf_counter() - start,
    )
    table.to_csv(out, index=False)

    print(c.fmt(table[["feature", "contribution", "share_of_explained"]], 2))
    print(
        f"\ncost achieved {decomposition.attrs['metric_value']:,.0f} $/defendant, benchmark "
        f"{decomposition['benchmark']:,.0f}; efficiency error {error:.1e}\n-> {out}"
    )


if __name__ == "__main__":
    main()
