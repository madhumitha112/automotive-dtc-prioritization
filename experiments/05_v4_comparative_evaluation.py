
"""
EXPERIMENT 05 - V4 Comparative Evaluation

Purpose:
Compare the frozen static DTC baseline and frozen CAN-context
refinement method on the same V4 diagnostic benchmark cases.

No ranking rules, static priorities, context rules or weights
are modified in this experiment.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

STATIC_FILE = ROOT / "results" / "static_baseline_results_v4.csv"
CONTEXT_FILE = ROOT / "results" / "context_refinement_results_v4.csv"

OUTPUT_DIR = ROOT / "results"

CASE_OUTPUT = OUTPUT_DIR / "v4_comparative_case_results.csv"
METRICS_OUTPUT = OUTPUT_DIR / "v4_comparative_metrics.csv"

EXPECTED_CASES = 24


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


def validate_ranking(ranking, candidates, case_id, method):
    """Validate that a ranking contains every candidate exactly once."""

    if not ranking:
        raise ValueError(
            f"{method}: empty ranking in {case_id}"
        )

    if len(ranking) != len(set(ranking)):
        raise ValueError(
            f"{method}: duplicate DTC in ranking for {case_id}"
        )

    if set(ranking) != set(candidates):
        raise ValueError(
            f"{method}: ranking does not match candidates "
            f"for {case_id}"
        )


def calculate_metrics(df, rank_column):
    """Calculate ranking performance metrics."""

    ranks = pd.to_numeric(
        df[rank_column],
        errors="raise",
    )

    candidate_counts = pd.to_numeric(
        df["candidate_count"],
        errors="raise",
    )

    return {
        "case_count": len(df),
        "top1_accuracy": float((ranks == 1).mean()),
        "top3_accuracy": float((ranks <= 3).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "mean_primary_rank": float(ranks.mean()),
        "median_primary_rank": float(ranks.median()),
        "mean_candidate_count": float(candidate_counts.mean()),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("EXPERIMENT 05 - V4 COMPARATIVE EVALUATION")
    print("=" * 72)

    # --------------------------------------------------------
    # INPUT FILE VALIDATION
    # --------------------------------------------------------

    for file_path in [STATIC_FILE, CONTEXT_FILE]:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required input file not found: {file_path}"
            )

    static = pd.read_csv(STATIC_FILE)
    context = pd.read_csv(CONTEXT_FILE)

    print(f"\nStatic baseline rows: {len(static)}")
    print(f"Context refinement rows: {len(context)}")

    if len(static) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_CASES} static rows; "
            f"found {len(static)}"
        )

    if len(context) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_CASES} context rows; "
            f"found {len(context)}"
        )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    static_required = {
        "case_id",
        "session_id",
        "candidate_dtcs",
        "candidate_count",
        "static_ranking",
        "primary_dtc",
        "primary_rank_static",
        "reciprocal_rank_static",
        "top_1_correct_static",
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
            f"Missing static columns: {sorted(missing_static)}"
        )

    if missing_context:
        raise ValueError(
            f"Missing context columns: {sorted(missing_context)}"
        )

    # --------------------------------------------------------
    # CASE ID VALIDATION
    # --------------------------------------------------------

    if static["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs in static results.")

    if context["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs in context results.")

    static["case_id"] = static["case_id"].astype(str)
    context["case_id"] = context["case_id"].astype(str)

    static_ids = set(static["case_id"])
    context_ids = set(context["case_id"])

    if static_ids != context_ids:
        raise ValueError(
            "Static and context case IDs do not match. "
            f"Static-only: {sorted(static_ids - context_ids)}. "
            f"Context-only: {sorted(context_ids - static_ids)}."
        )

    # --------------------------------------------------------
    # MERGE PAIRED RESULTS
    # --------------------------------------------------------

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

    if len(paired) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_CASES} paired cases; "
            f"found {len(paired)}."
        )

    # --------------------------------------------------------
    # CASE-LEVEL VALIDATION AND COMPARISON
    # --------------------------------------------------------

    comparison_rows = []

    for _, row in paired.iterrows():

        case_id = str(row["case_id"])

        static_candidates = parse_ranking(
            row["candidate_dtcs_static"]
        )

        context_candidates = parse_ranking(
            row["candidate_dtcs_context"]
        )

        if set(static_candidates) != set(context_candidates):
            raise ValueError(
                f"Candidate mismatch between methods in {case_id}"
            )

        if len(static_candidates) != len(set(static_candidates)):
            raise ValueError(
                f"Duplicate candidates in {case_id}"
            )

        static_primary = str(
            row["primary_dtc_static"]
        ).strip().upper()

        context_primary = str(
            row["primary_dtc_context"]
        ).strip().upper()

        if static_primary != context_primary:
            raise ValueError(
                f"Primary DTC mismatch between methods in {case_id}"
            )

        if static_primary not in static_candidates:
            raise ValueError(
                f"Primary DTC is not a candidate in {case_id}"
            )

        static_ranking = parse_ranking(
            row["static_ranking"]
        )

        context_ranking = parse_ranking(
            row["refined_ranking"]
        )

        validate_ranking(
            static_ranking,
            static_candidates,
            case_id,
            "Static",
        )

        validate_ranking(
            context_ranking,
            context_candidates,
            case_id,
            "Context",
        )

        static_rank_calculated = (
            static_ranking.index(static_primary) + 1
        )

        context_rank_calculated = (
            context_ranking.index(context_primary) + 1
        )

        static_rank_recorded = int(
            row["primary_rank_static"]
        )

        context_rank_recorded = int(
            row["primary_rank"]
        )

        if static_rank_calculated != static_rank_recorded:
            raise ValueError(
                f"Static rank inconsistency in {case_id}"
            )

        if context_rank_calculated != context_rank_recorded:
            raise ValueError(
                f"Context rank inconsistency in {case_id}"
            )

        if len(static_candidates) != int(
            row["candidate_count_static"]
        ):
            raise ValueError(
                f"Static candidate count mismatch in {case_id}"
            )

        if len(context_candidates) != int(
            row["candidate_count_context"]
        ):
            raise ValueError(
                f"Context candidate count mismatch in {case_id}"
            )

        static_top1 = static_ranking[0]
        context_top1 = context_ranking[0]

        static_correct = int(
            static_top1 == static_primary
        )

        context_correct = int(
            context_top1 == context_primary
        )

        rank_change = (
            static_rank_calculated - context_rank_calculated
        )

        if rank_change > 0:
            outcome = "Improved"
        elif rank_change < 0:
            outcome = "Worsened"
        else:
            outcome = "Unchanged"

        comparison_rows.append({
            "case_id": case_id,
            "session_id": str(row["session_id_static"]),
            "candidate_dtcs": "|".join(static_candidates),
            "candidate_count": len(static_candidates),
            "primary_dtc": static_primary,
            "static_ranking": "|".join(static_ranking),
            "context_ranking": "|".join(context_ranking),
            "static_top1_dtc": static_top1,
            "context_top1_dtc": context_top1,
            "static_primary_rank": static_rank_calculated,
            "context_primary_rank": context_rank_calculated,
            "static_reciprocal_rank": 1.0 / static_rank_calculated,
            "context_reciprocal_rank": 1.0 / context_rank_calculated,
            "static_top1_correct": static_correct,
            "context_top1_correct": context_correct,
            "rank_improvement": rank_change,
            "rank_outcome": outcome,
            "top1_outcome": (
                "Corrected"
                if static_correct == 0 and context_correct == 1
                else "Regressed"
                if static_correct == 1 and context_correct == 0
                else "Both correct"
                if static_correct == 1 and context_correct == 1
                else "Both incorrect"
            ),
        })

    comparison = pd.DataFrame(comparison_rows)

    # --------------------------------------------------------
    # METHOD METRICS
    # --------------------------------------------------------

    static_metrics = calculate_metrics(
        comparison,
        "static_primary_rank",
    )

    context_metrics = calculate_metrics(
        comparison,
        "context_primary_rank",
    )

    metric_rows = [
        {
            "method": "Static baseline",
            **static_metrics,
        },
        {
            "method": "CAN-context refinement",
            **context_metrics,
        },
    ]

    delta_metrics = {}

    for metric in [
        "top1_accuracy",
        "top3_accuracy",
        "mrr",
        "mean_primary_rank",
        "median_primary_rank",
    ]:
        delta_metrics[metric] = (
            context_metrics[metric] - static_metrics[metric]
        )

    metric_rows.append({
        "method": "Context minus static",
        "case_count": EXPECTED_CASES,
        **delta_metrics,
        "mean_candidate_count": context_metrics[
            "mean_candidate_count"
        ] - static_metrics["mean_candidate_count"],
    })

    metrics = pd.DataFrame(metric_rows)

    # --------------------------------------------------------
    # PAIRED OUTCOME COUNTS
    # --------------------------------------------------------

    improved_cases = int(
        (comparison["rank_outcome"] == "Improved").sum()
    )

    worsened_cases = int(
        (comparison["rank_outcome"] == "Worsened").sum()
    )

    unchanged_cases = int(
        (comparison["rank_outcome"] == "Unchanged").sum()
    )

    corrected_cases = int(
        (comparison["top1_outcome"] == "Corrected").sum()
    )

    regressed_cases = int(
        (comparison["top1_outcome"] == "Regressed").sum()
    )

    both_correct_cases = int(
        (comparison["top1_outcome"] == "Both correct").sum()
    )

    both_incorrect_cases = int(
        (comparison["top1_outcome"] == "Both incorrect").sum()
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        CASE_OUTPUT,
        index=False,
    )

    metrics.to_csv(
        METRICS_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("V4 COMPARATIVE EVALUATION RESULTS")
    print("=" * 72)

    print("\nMETHOD PERFORMANCE")
    print("-" * 72)

    print(
        metrics[
            [
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

    print("\nPAIRED CASE OUTCOMES")
    print("-" * 50)

    print(f"Cases with improved rank:       {improved_cases}")
    print(f"Cases with worsened rank:       {worsened_cases}")
    print(f"Cases with unchanged rank:      {unchanged_cases}")

    print(f"\nTop-1 cases corrected:          {corrected_cases}")
    print(f"Top-1 cases regressed:          {regressed_cases}")
    print(f"Both methods correct:           {both_correct_cases}")
    print(f"Both methods incorrect:         {both_incorrect_cases}")

    print("\nTOP-1 ACCURACY CHANGE")
    print("-" * 50)

    print(
        f"{(context_metrics['top1_accuracy'] - static_metrics['top1_accuracy']) * 100:+.2f} "
        "percentage points"
    )

    print("\nCASE-LEVEL COMPARISON")
    print("-" * 72)

    print(
        comparison[
            [
                "case_id",
                "primary_dtc",
                "static_primary_rank",
                "context_primary_rank",
                "rank_outcome",
                "top1_outcome",
            ]
        ].to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print("-" * 50)
    print(CASE_OUTPUT)
    print(METRICS_OUTPUT)

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
