
"""
EXPERIMENT 06 - V4 CAN-Context Rule Audit

Purpose:
Audit the activation, candidate-level discrimination and ranking
contribution of the frozen CAN-context rules on the V4 benchmark.

This is a post-hoc evaluation. It does not modify thresholds,
static priorities, context weights or ranking rules.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

CONTEXT_FILE = (
    RESULTS_DIR / "context_refinement_results_v4.csv"
)

COMPARISON_FILE = (
    RESULTS_DIR / "v4_comparative_case_results.csv"
)

EXPECTED_CASES = 24
CONTEXT_WEIGHT = 1.00

STATIC_PRIORITY = {
    "P0335": 1.00,
    "P0300": 0.95,
    "C0035": 0.95,
    "P0121": 0.80,
    "P0130": 0.70,
    "P0171": 0.65,
    "P0172": 0.65,
    "P0128": 0.55,
}

DTC_RULES = list(STATIC_PRIORITY.keys())

CONTEXT_COLUMNS = {
    dtc: f"context_{dtc}"
    for dtc in DTC_RULES
}

CASE_OUTPUT = (
    RESULTS_DIR / "v4_context_rule_case_scores.csv"
)

ACTIVATION_OUTPUT = (
    RESULTS_DIR / "v4_context_rule_activation_summary.csv"
)

ABLATION_OUTPUT = (
    RESULTS_DIR / "v4_context_rule_ablation_summary.csv"
)

FAILURE_OUTPUT = (
    RESULTS_DIR / "v4_context_rule_failure_cases.csv"
)

REPORT_OUTPUT = (
    RESULTS_DIR / "v4_context_rule_audit_report.txt"
)


# ============================================================
# HELPERS
# ============================================================

def parse_candidates(value):
    """Parse pipe-separated DTC candidate values."""

    if pd.isna(value):
        return []

    return [
        item.strip().upper()
        for item in str(value).split("|")
        if item.strip()
    ]


def calculate_metrics(ranks):
    """Calculate ranking metrics from primary DTC ranks."""

    ranks = pd.Series(ranks, dtype=float)

    return {
        "case_count": int(len(ranks)),
        "top1_accuracy": float((ranks == 1).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "mean_primary_rank": float(ranks.mean()),
        "median_primary_rank": float(ranks.median()),
    }


def rank_candidates(candidates, context_scores, removed_rule=None):
    """
    Reconstruct the frozen context ranking.

    Ablation sets the selected DTC's context contribution to zero.
    Its static priority and candidate status remain unchanged.
    """

    ranking = []

    for dtc in candidates:

        static_score = STATIC_PRIORITY[dtc]
        context_score = context_scores[dtc]

        if dtc == removed_rule:
            context_score = 0.0

        refined_score = (
            static_score
            + CONTEXT_WEIGHT * context_score
        )

        ranking.append({
            "dtc": dtc,
            "static_score": static_score,
            "context_score": context_score,
            "refined_score": refined_score,
        })

    ranking.sort(
        key=lambda item: (
            -item["refined_score"],
            -item["static_score"],
            item["dtc"],
        )
    )

    return ranking


# ============================================================
# INPUT VALIDATION
# ============================================================

def load_inputs():

    for path in [CONTEXT_FILE, COMPARISON_FILE]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    context = pd.read_csv(CONTEXT_FILE)
    comparison = pd.read_csv(COMPARISON_FILE)

    required_context = {
        "case_id",
        "session_id",
        "candidate_dtcs",
        "candidate_count",
        "refined_ranking",
        "primary_dtc",
        "primary_rank",
        "reciprocal_rank",
        *CONTEXT_COLUMNS.values(),
    }

    required_comparison = {
        "case_id",
        "primary_dtc",
        "context_primary_rank",
    }

    missing_context = required_context - set(context.columns)
    missing_comparison = required_comparison - set(
        comparison.columns
    )

    if missing_context:
        raise ValueError(
            f"Missing context columns: {sorted(missing_context)}"
        )

    if missing_comparison:
        raise ValueError(
            f"Missing comparison columns: {sorted(missing_comparison)}"
        )

    if len(context) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_CASES} context cases; "
            f"found {len(context)}."
        )

    if len(comparison) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_CASES} comparison cases; "
            f"found {len(comparison)}."
        )

    if context["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs in context results.")

    if comparison["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs in comparison results.")

    context["case_id"] = context["case_id"].astype(str)
    comparison["case_id"] = comparison["case_id"].astype(str)

    if set(context["case_id"]) != set(comparison["case_id"]):
        raise ValueError(
            "Case IDs do not match between context and comparison files."
        )

    # Missing values are permitted for non-candidate DTCs.
    # Populated context scores must remain within [0, 1].
    for column in CONTEXT_COLUMNS.values():

        context[column] = pd.to_numeric(
            context[column],
            errors="coerce",
        )

        invalid = (
            context[column].notna()
            & ~context[column].between(0.0, 1.0)
        )

        if invalid.any():
            raise ValueError(
                f"Context scores outside [0, 1]: {column}"
            )

    return context, comparison


# ============================================================
# CANDIDATE-LEVEL AUDIT
# ============================================================

def build_candidate_records(context):

    case_records = []
    candidate_records = []

    for _, row in context.iterrows():

        case_id = str(row["case_id"])
        session_id = str(row["session_id"])

        candidates = parse_candidates(
            row["candidate_dtcs"]
        )

        primary = str(
            row["primary_dtc"]
        ).strip().upper()

        if len(candidates) != int(row["candidate_count"]):
            raise ValueError(
                f"{case_id}: candidate count mismatch."
            )

        if len(candidates) != len(set(candidates)):
            raise ValueError(
                f"{case_id}: duplicate candidate DTCs."
            )

        if primary not in candidates:
            raise ValueError(
                f"{case_id}: primary DTC not in candidates."
            )

        unknown = set(candidates) - set(DTC_RULES)

        if unknown:
            raise ValueError(
                f"{case_id}: unknown DTCs: {sorted(unknown)}"
            )

        context_scores = {}

        for dtc in candidates:

            score = row[CONTEXT_COLUMNS[dtc]]

            if pd.isna(score):
                raise ValueError(
                    f"{case_id}: missing context score for "
                    f"candidate DTC {dtc}."
                )

            score = float(score)

            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"{case_id}: invalid context score for "
                    f"{dtc}: {score}"
                )

            context_scores[dtc] = score

            candidate_records.append({
                "case_id": case_id,
                "session_id": session_id,
                "dtc": dtc,
                "candidate_role": (
                    "PRIMARY"
                    if dtc == primary
                    else "COMPETING_CANDIDATE"
                ),
                "primary_dtc": primary,
                "static_priority": STATIC_PRIORITY[dtc],
                "context_score": score,
                "context_activation": int(score > 0.0),
                "strong_activation": int(score >= 0.5),
            })

        ranking = rank_candidates(
            candidates,
            context_scores,
        )

        ranked_dtcs = [
            item["dtc"] for item in ranking
        ]

        reconstructed_rank = ranked_dtcs.index(primary) + 1
        stored_rank = int(row["primary_rank"])

        if reconstructed_rank != stored_rank:
            raise ValueError(
                f"{case_id}: reconstructed rank "
                f"{reconstructed_rank} differs from stored rank "
                f"{stored_rank}."
            )

        stored_ranking = parse_candidates(
            row["refined_ranking"]
        )

        if ranked_dtcs != stored_ranking:
            raise ValueError(
                f"{case_id}: reconstructed ranking differs "
                "from stored ranking."
            )

        case_records.append({
            "case_id": case_id,
            "session_id": session_id,
            "candidate_dtcs": "|".join(candidates),
            "candidate_count": len(candidates),
            "primary_dtc": primary,
            "primary_rank": reconstructed_rank,
            "top1_dtc": ranked_dtcs[0],
            "full_ranking": "|".join(ranked_dtcs),
            "candidates": candidates,
            "context_scores": context_scores,
        })

    return (
        pd.DataFrame(candidate_records),
        case_records,
    )


# ============================================================
# RULE ACTIVATION SUMMARY
# ============================================================

def build_activation_summary(candidate_df):

    records = []

    for dtc in DTC_RULES:

        subset = candidate_df[
            candidate_df["dtc"] == dtc
        ]

        primary = subset[
            subset["candidate_role"] == "PRIMARY"
        ]

        competing = subset[
            subset["candidate_role"] == "COMPETING_CANDIDATE"
        ]

        primary_count = len(primary)
        competing_count = len(competing)

        def mean_or_nan(series):
            return (
                float(series.mean())
                if len(series)
                else np.nan
            )

        if primary_count == 0:
            coverage = "NO_PRIMARY_CASES"
        elif competing_count == 0:
            coverage = "NO_COMPETING_CASES"
        elif primary_count < 4 or competing_count < 4:
            coverage = "LIMITED_COVERAGE"
        else:
            coverage = "BASIC_COMPARATIVE_COVERAGE"

        primary_mean = mean_or_nan(
            primary["context_score"]
        )

        competing_mean = mean_or_nan(
            competing["context_score"]
        )

        if (
            np.isnan(primary_mean)
            or np.isnan(competing_mean)
        ):
            score_difference = np.nan
        else:
            score_difference = (
                primary_mean - competing_mean
            )

        records.append({
            "dtc": dtc,
            "candidate_appearances": len(subset),
            "primary_appearances": primary_count,
            "competing_appearances": competing_count,
            "mean_context_score_all": mean_or_nan(
                subset["context_score"]
            ),
            "mean_context_score_primary": primary_mean,
            "mean_context_score_competing": competing_mean,
            "primary_minus_competing_mean_score": score_difference,
            "activation_pct_all": (
                100 * mean_or_nan(
                    subset["context_activation"]
                )
                if len(subset) else np.nan
            ),
            "activation_pct_primary": (
                100 * mean_or_nan(
                    primary["context_activation"]
                )
                if primary_count else np.nan
            ),
            "activation_pct_competing": (
                100 * mean_or_nan(
                    competing["context_activation"]
                )
                if competing_count else np.nan
            ),
            "strong_activation_pct_primary": (
                100 * mean_or_nan(
                    primary["strong_activation"]
                )
                if primary_count else np.nan
            ),
            "strong_activation_pct_competing": (
                100 * mean_or_nan(
                    competing["strong_activation"]
                )
                if competing_count else np.nan
            ),
            "coverage_assessment": coverage,
        })

    return pd.DataFrame(records)


# ============================================================
# ABLATION AUDIT
# ============================================================

def build_ablation_summary(case_records):

    static_ranks = []
    full_ranks = []

    for record in case_records:

        candidates = record["candidates"]
        primary = record["primary_dtc"]

        static_ranking = sorted(
            candidates,
            key=lambda dtc: (
                -STATIC_PRIORITY[dtc],
                dtc,
            ),
        )

        static_ranks.append(
            static_ranking.index(primary) + 1
        )

        full_ranks.append(
            record["primary_rank"]
        )

    static_metrics = calculate_metrics(static_ranks)
    full_metrics = calculate_metrics(full_ranks)

    records = []

    records.append({
        "configuration": "STATIC_ONLY",
        "removed_rule": "NONE",
        **static_metrics,
    })

    records.append({
        "configuration": "FULL_CONTEXT_MODEL",
        "removed_rule": "NONE",
        **full_metrics,
    })

    for removed_dtc in DTC_RULES:

        ranks = []

        for record in case_records:

            ranking = rank_candidates(
                record["candidates"],
                record["context_scores"],
                removed_rule=removed_dtc,
            )

            ranked_dtcs = [
                item["dtc"] for item in ranking
            ]

            ranks.append(
                ranked_dtcs.index(record["primary_dtc"]) + 1
            )

        records.append({
            "configuration": "RULE_REMOVED",
            "removed_rule": removed_dtc,
            **calculate_metrics(ranks),
        })

    ablation = pd.DataFrame(records)

    full = ablation[
        ablation["configuration"] == "FULL_CONTEXT_MODEL"
    ].iloc[0]

    ablation["top1_delta_vs_full"] = (
        ablation["top1_accuracy"] - full["top1_accuracy"]
    )

    ablation["mrr_delta_vs_full"] = (
        ablation["mrr"] - full["mrr"]
    )

    ablation["mean_rank_delta_vs_full"] = (
        ablation["mean_primary_rank"]
        - full["mean_primary_rank"]
    )

    return ablation


# ============================================================
# FAILURE CASE ANALYSIS
# ============================================================

def build_failure_cases(case_records):

    records = []

    for record in case_records:

        if record["primary_rank"] == 1:
            continue

        ranking = rank_candidates(
            record["candidates"],
            record["context_scores"],
        )

        top = ranking[0]

        primary = next(
            item for item in ranking
            if item["dtc"] == record["primary_dtc"]
        )

        records.append({
            "case_id": record["case_id"],
            "candidate_dtcs": record["candidate_dtcs"],
            "primary_dtc": record["primary_dtc"],
            "primary_rank": record["primary_rank"],
            "top1_dtc": top["dtc"],
            "top1_static_score": top["static_score"],
            "top1_context_score": top["context_score"],
            "primary_static_score": primary["static_score"],
            "primary_context_score": primary["context_score"],
            "context_score_difference_top1_minus_primary": (
                top["context_score"] - primary["context_score"]
            ),
            "full_ranking": record["full_ranking"],
        })

    return pd.DataFrame(records)


# ============================================================
# REPORT
# ============================================================

def write_report(
    context,
    candidate_df,
    activation,
    ablation,
    failures,
):

    full_metrics = ablation[
        ablation["configuration"] == "FULL_CONTEXT_MODEL"
    ].iloc[0]

    static_metrics = ablation[
        ablation["configuration"] == "STATIC_ONLY"
    ].iloc[0]

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "EXPERIMENT 06 - V4 CAN-CONTEXT RULE AUDIT\n"
        )

        report.write("=" * 78 + "\n\n")

        report.write("BENCHMARK SUMMARY\n")
        report.write("-" * 78 + "\n")

        report.write(f"Cases audited: {len(context)}\n")
        report.write(
            f"Candidate-level records: {len(candidate_df)}\n"
        )
        report.write(f"Context weight: {CONTEXT_WEIGHT:.2f}\n")

        report.write(
            "Static priorities and context rules were not modified.\n"
        )

        report.write("\nFULL MODEL VS STATIC BASELINE\n")
        report.write("-" * 78 + "\n")

        report.write(
            f"Static Top-1 accuracy: "
            f"{static_metrics['top1_accuracy']:.4f}\n"
        )

        report.write(
            f"Context Top-1 accuracy: "
            f"{full_metrics['top1_accuracy']:.4f}\n"
        )

        report.write(
            f"Static MRR: {static_metrics['mrr']:.4f}\n"
        )

        report.write(
            f"Context MRR: {full_metrics['mrr']:.4f}\n"
        )

        report.write("\nRULE ACTIVATION AND DISCRIMINATION\n")
        report.write("-" * 78 + "\n")

        report.write(
            activation.round(4).to_string(index=False)
        )

        report.write("\n\nLEAVE-ONE-RULE-OUT ABLATION\n")
        report.write("-" * 78 + "\n")

        report.write(
            ablation.round(4).to_string(index=False)
        )

        report.write("\n\nFAILURE CASES\n")
        report.write("-" * 78 + "\n")

        if failures.empty:
            report.write("No incorrectly ranked cases.\n")
        else:
            report.write(
                failures.round(4).to_string(index=False)
            )

        report.write("\n\nINTERPRETATION LIMITATIONS\n")
        report.write("-" * 78 + "\n")

        report.write(
            "1. Context scores represent threshold-based evidence "
            "fractions, not calibrated fault probabilities.\n"
        )

        report.write(
            "2. Primary-versus-competing score comparisons are "
            "descriptive and do not establish statistical significance.\n"
        )

        report.write(
            "3. Ablation measures the effect of zeroing one DTC's "
            "context contribution while retaining its static priority.\n"
        )

        report.write(
            "4. A rule's ranking contribution is specific to this "
            "V4 benchmark and its candidate combinations.\n"
        )

        report.write(
            "5. The V4 benchmark contains 12 scenario templates "
            "with two repetitions each; repetitions are not "
            "independent real-world vehicle trials.\n"
        )

        report.write(
            "6. P0335 evidence uses RPM instability as an indirect "
            "proxy, not a direct crankshaft-position waveform.\n"
        )

        report.write(
            "7. No thresholds, weights, decoding definitions or "
            "ranking rules were tuned in this audit.\n"
        )

        report.write("\nSTATUS: PASS\n")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 72)
    print("EXPERIMENT 06 - V4 CAN-CONTEXT RULE AUDIT")
    print("=" * 72)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    context, comparison = load_inputs()

    print(f"\nContext cases loaded: {len(context)}")
    print(f"Comparison cases loaded: {len(comparison)}")

    # Cross-check context ranks against comparative results.
    merged = context[
        ["case_id", "primary_dtc", "primary_rank"]
    ].merge(
        comparison[
            [
                "case_id",
                "primary_dtc",
                "context_primary_rank",
            ]
        ],
        on="case_id",
        suffixes=("_context", "_comparison"),
        validate="one_to_one",
    )

    if not (
        merged["primary_dtc_context"].astype(str)
        == merged["primary_dtc_comparison"].astype(str)
    ).all():
        raise ValueError(
            "Primary DTC mismatch with comparative results."
        )

    if not (
        pd.to_numeric(
            merged["primary_rank"],
            errors="raise",
        )
        == pd.to_numeric(
            merged["context_primary_rank"],
            errors="raise",
        )
    ).all():
        raise ValueError(
            "Context ranks do not match comparative results."
        )

    candidate_df, case_records = build_candidate_records(
        context
    )

    activation = build_activation_summary(
        candidate_df
    )

    ablation = build_ablation_summary(
        case_records
    )

    failures = build_failure_cases(
        case_records
    )

    # Save CSV outputs.
    candidate_df.to_csv(
        CASE_OUTPUT,
        index=False,
    )

    activation.to_csv(
        ACTIVATION_OUTPUT,
        index=False,
    )

    ablation.to_csv(
        ABLATION_OUTPUT,
        index=False,
    )

    failures.to_csv(
        FAILURE_OUTPUT,
        index=False,
    )

    write_report(
        context,
        candidate_df,
        activation,
        ablation,
        failures,
    )

    # Console summary.
    full = ablation[
        ablation["configuration"] == "FULL_CONTEXT_MODEL"
    ].iloc[0]

    static = ablation[
        ablation["configuration"] == "STATIC_ONLY"
    ].iloc[0]

    print("\nRULE ACTIVATION SUMMARY")
    print("-" * 72)

    print(
        activation[
            [
                "dtc",
                "primary_appearances",
                "competing_appearances",
                "primary_minus_competing_mean_score",
                "coverage_assessment",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nABLATION SUMMARY")
    print("-" * 72)

    print(
        ablation[
            [
                "configuration",
                "removed_rule",
                "top1_accuracy",
                "mrr",
                "mean_primary_rank",
                "top1_delta_vs_full",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nFAILURE CASES")
    print("-" * 72)

    if failures.empty:
        print("No failure cases.")
    else:
        print(
            failures[
                [
                    "case_id",
                    "primary_dtc",
                    "top1_dtc",
                    "primary_rank",
                ]
            ].to_string(index=False)
        )

    print("\nOUTPUT FILES")
    print("-" * 72)

    for path in [
        CASE_OUTPUT,
        ACTIVATION_OUTPUT,
        ABLATION_OUTPUT,
        FAILURE_OUTPUT,
        REPORT_OUTPUT,
    ]:
        print(path)

    print(
        "\nStatic Top-1 accuracy:",
        f"{static['top1_accuracy']:.4f}",
    )

    print(
        "Context Top-1 accuracy:",
        f"{full['top1_accuracy']:.4f}",
    )

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
