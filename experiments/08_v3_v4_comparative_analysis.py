
"""
EXPERIMENT 08 - V3 VS V4 COMPARATIVE ANALYSIS

Purpose:
Compare the static baseline and CAN-context refinement methods
within V3 and V4, then compare the observed method improvements
between the two benchmark versions.

The original datasets, static priorities, context rules and
context weights remain unchanged.

This is a descriptive comparison of controlled benchmark results.
It does not establish real-vehicle diagnostic generalization.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

INPUTS = {
    "V3": {
        "static": RESULTS_DIR / "static_baseline_results.csv",
        "context": RESULTS_DIR / "context_refinement_results.csv",
        "expected_cases": 48,
    },
    "V4": {
        "static": RESULTS_DIR / "static_baseline_results_v4.csv",
        "context": RESULTS_DIR / "context_refinement_results_v4.csv",
        "expected_cases": 24,
    },
}

CASE_OUTPUT = RESULTS_DIR / "v3_v4_comparative_case_results.csv"
METRICS_OUTPUT = RESULTS_DIR / "v3_v4_comparative_metrics.csv"
OUTCOME_OUTPUT = RESULTS_DIR / "v3_v4_comparative_outcomes.csv"
REPORT_OUTPUT = RESULTS_DIR / "v3_v4_comparative_report.txt"
RUN_OUTPUT = RESULTS_DIR / "v3_v4_comparative_run.txt"


# ============================================================
# HELPERS
# ============================================================

def parse_ranking(value):
    """Parse a pipe-separated DTC ranking."""

    if pd.isna(value):
        return []

    return [
        item.strip().upper()
        for item in str(value).split("|")
        if item.strip()
    ]


def normalize_dtc(value):
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def validate_ranking(ranking, candidates, case_id, method, version):
    """Check that a ranking contains every candidate exactly once."""

    if not ranking:
        raise ValueError(
            f"{version} {method}: empty ranking in {case_id}"
        )

    if len(ranking) != len(set(ranking)):
        raise ValueError(
            f"{version} {method}: duplicate DTC in ranking "
            f"for {case_id}"
        )

    if set(ranking) != set(candidates):
        raise ValueError(
            f"{version} {method}: ranking does not match "
            f"candidate set in {case_id}. "
            f"Ranking={ranking}; candidates={candidates}"
        )


def calculate_metrics(records):
    """Calculate metrics from validated primary-DTC ranks."""

    ranks = pd.to_numeric(
        records["primary_rank"],
        errors="raise",
    )

    candidate_counts = pd.to_numeric(
        records["candidate_count"],
        errors="raise",
    )

    return {
        "case_count": len(records),
        "top1_accuracy": float((ranks == 1).mean()),
        "top3_accuracy": float((ranks <= 3).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "mean_primary_rank": float(ranks.mean()),
        "median_primary_rank": float(ranks.median()),
        "mean_candidate_count": float(candidate_counts.mean()),
    }


# ============================================================
# INPUT VALIDATION
# ============================================================

def load_benchmark(version, paths):

    static_file = paths["static"]
    context_file = paths["context"]
    expected_cases = paths["expected_cases"]

    for file_path in [static_file, context_file]:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required {version} input not found: {file_path}"
            )

    static = pd.read_csv(static_file)
    context = pd.read_csv(context_file)

    print(f"\n{version} INPUT VALIDATION")
    print("-" * 65)
    print(f"Static rows: {len(static)}")
    print(f"Context rows: {len(context)}")

    if len(static) != expected_cases:
        raise ValueError(
            f"{version}: expected {expected_cases} static rows; "
            f"found {len(static)}"
        )

    if len(context) != expected_cases:
        raise ValueError(
            f"{version}: expected {expected_cases} context rows; "
            f"found {len(context)}"
        )

    static_required = {
        "case_id",
        "session_id",
        "candidate_dtcs",
        "candidate_count",
        "static_ranking",
        "primary_dtc",
        "primary_rank_static",
        "reciprocal_rank_static",
    }

    context_required = {
        "case_id",
        "session_id",
        "candidate_dtcs",
        "candidate_count",
        "refined_ranking",
        "primary_dtc",
        "primary_rank",
        "reciprocal_rank",
    }

    missing_static = static_required - set(static.columns)
    missing_context = context_required - set(context.columns)

    if missing_static:
        raise ValueError(
            f"{version}: missing static columns: "
            f"{sorted(missing_static)}"
        )

    if missing_context:
        raise ValueError(
            f"{version}: missing context columns: "
            f"{sorted(missing_context)}"
        )

    static["case_id"] = static["case_id"].astype(str)
    context["case_id"] = context["case_id"].astype(str)

    if static["case_id"].duplicated().any():
        raise ValueError(
            f"{version}: duplicate case IDs in static results."
        )

    if context["case_id"].duplicated().any():
        raise ValueError(
            f"{version}: duplicate case IDs in context results."
        )

    if set(static["case_id"]) != set(context["case_id"]):
        raise ValueError(
            f"{version}: static and context case IDs do not match."
        )

    return static, context


# ============================================================
# BUILD CASE-LEVEL COMPARISON
# ============================================================

def build_case_comparison(version, static, context):

    paired = static.merge(
        context,
        on="case_id",
        how="inner",
        suffixes=("_static", "_context"),
        validate="one_to_one",
    )

    paired = paired.sort_values(
        "case_id"
    ).reset_index(drop=True)

    rows = []

    for _, record in paired.iterrows():

        case_id = str(record["case_id"])

        static_candidates = parse_ranking(
            record["candidate_dtcs_static"]
        )

        context_candidates = parse_ranking(
            record["candidate_dtcs_context"]
        )

        if set(static_candidates) != set(context_candidates):
            raise ValueError(
                f"{version}: candidate sets differ in {case_id}"
            )

        if len(static_candidates) != int(
            record["candidate_count_static"]
        ):
            raise ValueError(
                f"{version}: static candidate count mismatch "
                f"in {case_id}"
            )

        if len(context_candidates) != int(
            record["candidate_count_context"]
        ):
            raise ValueError(
                f"{version}: context candidate count mismatch "
                f"in {case_id}"
            )

        static_primary = normalize_dtc(
            record["primary_dtc_static"]
        )

        context_primary = normalize_dtc(
            record["primary_dtc_context"]
        )

        if static_primary != context_primary:
            raise ValueError(
                f"{version}: primary DTC mismatch in {case_id}"
            )

        if static_primary not in static_candidates:
            raise ValueError(
                f"{version}: primary DTC is not a candidate "
                f"in {case_id}"
            )

        static_ranking = parse_ranking(
            record["static_ranking"]
        )

        context_ranking = parse_ranking(
            record["refined_ranking"]
        )

        validate_ranking(
            static_ranking,
            static_candidates,
            case_id,
            "Static",
            version,
        )

        validate_ranking(
            context_ranking,
            context_candidates,
            case_id,
            "Context",
            version,
        )

        # Recalculate ranks from the actual ranking strings.
        static_rank = static_ranking.index(static_primary) + 1
        context_rank = context_ranking.index(context_primary) + 1

        # Verify the recalculated ranks against stored values.
        if static_rank != int(record["primary_rank_static"]):
            raise ValueError(
                f"{version}: static primary rank mismatch "
                f"in {case_id}"
            )

        if context_rank != int(record["primary_rank"]):
            raise ValueError(
                f"{version}: context primary rank mismatch "
                f"in {case_id}"
            )

        static_top1 = static_ranking[0]
        context_top1 = context_ranking[0]

        static_correct = int(static_top1 == static_primary)
        context_correct = int(context_top1 == context_primary)

        rank_improvement = static_rank - context_rank

        if rank_improvement > 0:
            rank_outcome = "Improved"
        elif rank_improvement < 0:
            rank_outcome = "Worsened"
        else:
            rank_outcome = "Unchanged"

        if static_correct == 0 and context_correct == 1:
            top1_outcome = "Corrected"
        elif static_correct == 1 and context_correct == 0:
            top1_outcome = "Regressed"
        elif static_correct == 1 and context_correct == 1:
            top1_outcome = "Both correct"
        else:
            top1_outcome = "Both incorrect"

        rows.append({
            "benchmark": version,
            "case_id": case_id,
            "session_id": str(record["session_id_static"]),
            "candidate_dtcs": "|".join(
                sorted(static_candidates)
            ),
            "candidate_count": len(static_candidates),
            "primary_dtc": static_primary,
            "static_ranking": "|".join(static_ranking),
            "context_ranking": "|".join(context_ranking),
            "static_top1_dtc": static_top1,
            "context_top1_dtc": context_top1,
            "static_primary_rank": static_rank,
            "context_primary_rank": context_rank,
            "static_reciprocal_rank": 1.0 / static_rank,
            "context_reciprocal_rank": 1.0 / context_rank,
            "static_top1_correct": static_correct,
            "context_top1_correct": context_correct,
            "rank_improvement": rank_improvement,
            "rank_outcome": rank_outcome,
            "top1_outcome": top1_outcome,
        })

    return pd.DataFrame(rows)


# ============================================================
# METRICS BY BENCHMARK AND METHOD
# ============================================================

def build_metrics(case_results):

    metric_rows = []

    for version in ["V3", "V4"]:

        version_data = case_results[
            case_results["benchmark"] == version
        ].copy()

        static_records = version_data[
            [
                "candidate_count",
                "static_primary_rank",
            ]
        ].rename(
            columns={
                "static_primary_rank": "primary_rank"
            }
        )

        context_records = version_data[
            [
                "candidate_count",
                "context_primary_rank",
            ]
        ].rename(
            columns={
                "context_primary_rank": "primary_rank"
            }
        )

        static_metrics = calculate_metrics(static_records)
        context_metrics = calculate_metrics(context_records)

        for method, metrics in [
            ("Static baseline", static_metrics),
            ("CAN-context refinement", context_metrics),
        ]:

            metric_rows.append({
                "benchmark": version,
                "method": method,
                **metrics,
            })

        # Context improvement relative to static baseline.
        delta = {
            key: context_metrics[key] - static_metrics[key]
            for key in [
                "top1_accuracy",
                "top3_accuracy",
                "mrr",
                "mean_primary_rank",
                "median_primary_rank",
                "mean_candidate_count",
            ]
        }

        metric_rows.append({
            "benchmark": version,
            "method": "Context minus static",
            "case_count": len(version_data),
            **delta,
        })

    return pd.DataFrame(metric_rows)


# ============================================================
# PAIRED OUTCOME SUMMARY
# ============================================================

def build_outcome_summary(case_results):

    rows = []

    for version in ["V3", "V4"]:

        group = case_results[
            case_results["benchmark"] == version
        ]

        rows.append({
            "benchmark": version,
            "case_count": len(group),
            "rank_improved": int(
                (group["rank_outcome"] == "Improved").sum()
            ),
            "rank_worsened": int(
                (group["rank_outcome"] == "Worsened").sum()
            ),
            "rank_unchanged": int(
                (group["rank_outcome"] == "Unchanged").sum()
            ),
            "top1_corrected": int(
                (group["top1_outcome"] == "Corrected").sum()
            ),
            "top1_regressed": int(
                (group["top1_outcome"] == "Regressed").sum()
            ),
            "both_correct": int(
                (group["top1_outcome"] == "Both correct").sum()
            ),
            "both_incorrect": int(
                (group["top1_outcome"] == "Both incorrect").sum()
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# REPORT
# ============================================================

def write_report(
    case_results,
    metrics,
    outcomes,
):

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "EXPERIMENT 08 - V3 VS V4 COMPARATIVE ANALYSIS\n"
        )

        report.write("=" * 78 + "\n\n")

        report.write("1. BENCHMARK STRUCTURE\n")
        report.write("-" * 78 + "\n")

        for version in ["V3", "V4"]:

            group = case_results[
                case_results["benchmark"] == version
            ]

            report.write(
                f"{version}: {len(group)} cases; "
                f"mean candidate count = "
                f"{group['candidate_count'].mean():.4f}\n"
            )

        report.write(
            "\nV3 contains 48 cases. V4 contains 24 cases "
            "from 12 scenario templates repeated twice.\n"
        )

        report.write(
            "The V4 repetitions are not 24 independent "
            "scenario families.\n"
        )

        report.write("\n2. METHOD METRICS\n")
        report.write("-" * 78 + "\n")
        report.write(metrics.to_string(index=False))

        report.write("\n\n3. PAIRED CASE OUTCOMES\n")
        report.write("-" * 78 + "\n")
        report.write(outcomes.to_string(index=False))

        report.write("\n\n4. TOP-1 IMPROVEMENT\n")
        report.write("-" * 78 + "\n")

        for version in ["V3", "V4"]:

            subset = metrics[
                metrics["benchmark"] == version
            ]

            static_row = subset[
                subset["method"] == "Static baseline"
            ].iloc[0]

            context_row = subset[
                subset["method"] == "CAN-context refinement"
            ].iloc[0]

            delta = (
                context_row["top1_accuracy"]
                - static_row["top1_accuracy"]
            )

            report.write(
                f"{version}: static = "
                f"{static_row['top1_accuracy']:.4f}; "
                f"context = {context_row['top1_accuracy']:.4f}; "
                f"change = {delta * 100:+.2f} percentage points\n"
            )

        report.write(
            "\n5. INTERPRETATION AND LIMITATIONS\n"
        )
        report.write("-" * 78 + "\n")

        report.write(
            "1. V3 and V4 are benchmark versions with different "
            "case counts and benchmark designs.\n"
        )

        report.write(
            "2. V4 uses 12 scenario templates repeated twice; "
            "the repeated cases should not be treated as "
            "independent scenario families.\n"
        )

        report.write(
            "3. V4 is a controlled synthetic benchmark and "
            "does not represent recordings from physical vehicles.\n"
        )

        report.write(
            "4. The comparison measures ranking performance "
            "conditional on the supplied candidate DTC set. "
            "It does not evaluate DTC detection.\n"
        )

        report.write(
            "5. Top-3 accuracy is not suitable as a primary "
            "cross-version comparison because V3 includes "
            "three-candidate cases while V4 has two candidates "
            "per case.\n"
        )

        report.write(
            "6. Differences between V3 and V4 performance cannot "
            "be attributed solely to benchmark size. Scenario "
            "composition and candidate-pair composition also differ.\n"
        )

        report.write(
            "7. These results do not establish generalization "
            "to real vehicles or unseen diagnostic conditions.\n"
        )

        report.write(
            "8. No ranking rules, context thresholds, static "
            "priorities or context weights were changed.\n"
        )

        report.write("\n6. OUTPUT FILES\n")
        report.write("-" * 78 + "\n")

        for path in [
            CASE_OUTPUT,
            METRICS_OUTPUT,
            OUTCOME_OUTPUT,
            REPORT_OUTPUT,
            RUN_OUTPUT,
        ]:
            report.write(str(path) + "\n")

        report.write("\nSTATUS: PASS\n")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("EXPERIMENT 08 - V3 VS V4 COMPARATIVE ANALYSIS")
    print("=" * 78)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_case_results = []

    # Load and evaluate both benchmark versions.
    for version, paths in INPUTS.items():

        static, context = load_benchmark(
            version,
            paths,
        )

        case_results = build_case_comparison(
            version,
            static,
            context,
        )

        if len(case_results) != paths["expected_cases"]:
            raise ValueError(
                f"{version}: unexpected comparison row count."
            )

        all_case_results.append(case_results)

    case_results = pd.concat(
        all_case_results,
        ignore_index=True,
    )

    metrics = build_metrics(case_results)

    outcomes = build_outcome_summary(case_results)

    # Confirm expected case totals.
    expected_total = sum(
        item["expected_cases"]
        for item in INPUTS.values()
    )

    if len(case_results) != expected_total:
        raise ValueError(
            f"Expected {expected_total} combined case rows; "
            f"found {len(case_results)}."
        )

    # Save outputs.
    case_results.to_csv(
        CASE_OUTPUT,
        index=False,
    )

    metrics.to_csv(
        METRICS_OUTPUT,
        index=False,
    )

    outcomes.to_csv(
        OUTCOME_OUTPUT,
        index=False,
    )

    write_report(
        case_results,
        metrics,
        outcomes,
    )

    # Display metrics.
    print("\nCOMPARATIVE METRICS")
    print("-" * 78)

    print(
        metrics[
            [
                "benchmark",
                "method",
                "case_count",
                "top1_accuracy",
                "top3_accuracy",
                "mrr",
                "mean_primary_rank",
                "median_primary_rank",
            ]
        ].to_string(index=False)
    )

    print("\nPAIRED OUTCOMES")
    print("-" * 78)

    print(
        outcomes.to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print("-" * 78)

    for path in [
        CASE_OUTPUT,
        METRICS_OUTPUT,
        OUTCOME_OUTPUT,
        REPORT_OUTPUT,
    ]:
        print(path)

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
