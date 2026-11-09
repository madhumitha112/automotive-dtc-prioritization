
"""
Experiment 04A
--------------
Post-hoc audit of CAN-context rules and ablation analysis.

Purpose
-------
1. Measure activation of each DTC context rule.
2. Compare rule scores when the corresponding DTC is primary
   versus when it is a competing candidate.
3. Perform leave-one-rule-out ablation.
4. Identify cases where the primary DTC is not ranked first.

Important
---------
This script reads the existing Experiment 04 output.
It does NOT:
- Modify the generated dataset.
- Recalculate or tune context rules.
- Change context thresholds or weights.
- Use primary DTC labels to calculate context scores.

Primary DTC labels are used only for evaluation and grouping.

This is a post-hoc diagnostic audit, not a new training
or optimization experiment.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# PATHS AND CONFIGURATION
# ================================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "context_refinement_results.csv"
)

OUTPUT_DIR = ROOT / "results"

EXPECTED_CASES = 48
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

DTC_RULES = [
    "P0171",
    "P0172",
    "P0128",
    "P0300",
    "P0130",
    "P0121",
    "P0335",
    "C0035",
]

CONTEXT_COLUMNS = {
    dtc: f"context_{dtc}"
    for dtc in DTC_RULES
}


# ================================================================
# HELPERS
# ================================================================

def parse_candidates(value):
    """Parse pipe-separated candidate DTC values."""

    if pd.isna(value):
        return []

    return [
        item.strip().upper()
        for item in str(value).split("|")
        if item.strip()
    ]


def calculate_metrics(frame):
    """Calculate ranking metrics from primary ranks."""

    ranks = frame["primary_rank"].astype(float)

    return {
        "cases": int(len(frame)),
        "top1_accuracy": float((ranks == 1).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "mean_primary_rank": float(ranks.mean()),
        "median_primary_rank": float(ranks.median()),
    }


def rank_candidates(
    candidates,
    context_scores,
    removed_rule=None,
):
    """
    Rank candidates using static + context scores.

    If removed_rule is supplied, only that DTC's context
    contribution is set to zero.

    Candidate DTCs remain in the ranking.
    """

    scored = []

    for dtc in candidates:

        static_score = STATIC_PRIORITY[dtc]

        context_score = context_scores[dtc]

        if dtc == removed_rule:
            context_score = 0.0

        refined_score = (
            static_score
            + CONTEXT_WEIGHT * context_score
        )

        scored.append({
            "dtc": dtc,
            "static_score": static_score,
            "context_score": context_score,
            "refined_score": refined_score,
        })

    scored.sort(
        key=lambda item: (
            -item["refined_score"],
            -item["static_score"],
            item["dtc"],
        )
    )

    return scored


# ================================================================
# LOAD INPUT
# ================================================================

print("=" * 72)
print("EXPERIMENT 04A - CONTEXT RULE AUDIT")
print("=" * 72)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Experiment 04 output not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nInput file: {INPUT_FILE}")
print(f"Rows loaded: {len(df)}")


# ================================================================
# INPUT VALIDATION
# ================================================================

required_columns = {
    "case_id",
    "session_id",
    "candidate_dtcs",
    "primary_dtc",
    "primary_rank",
    "reciprocal_rank",
}

required_columns.update(
    CONTEXT_COLUMNS.values()
)

missing_columns = sorted(
    required_columns - set(df.columns)
)

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

if len(df) != EXPECTED_CASES:
    raise ValueError(
        f"Expected {EXPECTED_CASES} cases; found {len(df)}."
    )

if df["case_id"].duplicated().any():
    raise ValueError("Duplicate case IDs found.")

df["case_id"] = df["case_id"].astype(str)
df["session_id"] = df["session_id"].astype(str)

df["primary_dtc"] = (
    df["primary_dtc"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["primary_rank"] = pd.to_numeric(
    df["primary_rank"],
    errors="raise",
)

for column in CONTEXT_COLUMNS.values():
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ================================================================
# BUILD CANDIDATE-LEVEL SCORE RECORDS
# ================================================================

case_records = []
candidate_records = []

for _, row in df.iterrows():

    case_id = row["case_id"]
    session_id = row["session_id"]

    candidates = parse_candidates(
        row["candidate_dtcs"]
    )

    primary_dtc = row["primary_dtc"]

    if len(candidates) < 2:
        raise ValueError(
            f"{case_id}: fewer than two candidates."
        )

    if primary_dtc not in candidates:
        raise ValueError(
            f"{case_id}: primary DTC not in candidate set."
        )

    unknown = [
        dtc for dtc in candidates
        if dtc not in STATIC_PRIORITY
    ]

    if unknown:
        raise ValueError(
            f"{case_id}: unknown candidates {unknown}"
        )

    context_scores = {}

    for dtc in candidates:

        column = CONTEXT_COLUMNS[dtc]

        score = row[column]

        if pd.isna(score):
            raise ValueError(
                f"{case_id}: missing context score for {dtc}."
            )

        score = float(score)

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"{case_id}: invalid context score "
                f"for {dtc}: {score}"
            )

        context_scores[dtc] = score

        candidate_records.append({
            "case_id": case_id,
            "session_id": session_id,
            "dtc": dtc,
            "primary_dtc": primary_dtc,
            "is_primary_candidate": int(dtc == primary_dtc),
            "static_score": STATIC_PRIORITY[dtc],
            "context_score": score,
            "any_activation": int(score > 0.0),
            "strong_activation": int(score >= 0.5),
        })

    # Reconstruct the full ranking from stored context scores.
    full_ranking = rank_candidates(
        candidates,
        context_scores,
    )

    full_ranked_dtcs = [
        item["dtc"] for item in full_ranking
    ]

    calculated_primary_rank = (
        full_ranked_dtcs.index(primary_dtc) + 1
    )

    # Check that the stored Experiment 04 result is reproducible.
    stored_rank = int(row["primary_rank"])

    if calculated_primary_rank != stored_rank:
        raise ValueError(
            f"{case_id}: reconstructed rank "
            f"{calculated_primary_rank} does not match "
            f"stored Experiment 04 rank {stored_rank}. "
            "Check the source output and tie-breaking logic."
        )

    case_records.append({
        "case_id": case_id,
        "session_id": session_id,
        "candidate_dtcs": "|".join(candidates),
        "candidate_count": len(candidates),
        "primary_dtc": primary_dtc,
        "primary_rank": calculated_primary_rank,
        "top1_dtc": full_ranked_dtcs[0],
        "full_ranking": "|".join(full_ranked_dtcs),
        "context_scores": context_scores,
        "candidates": candidates,
    })


candidate_df = pd.DataFrame(candidate_records)
case_df = pd.DataFrame(case_records)

print(
    f"\nCases audited: {len(case_df)}"
)

print(
    f"Candidate-level records: {len(candidate_df)}"
)


# ================================================================
# OUTPUT 1 - CASE-LEVEL CONTEXT SCORES
# ================================================================

case_score_output = candidate_df.copy()

case_score_output["candidate_role"] = np.where(
    case_score_output["is_primary_candidate"] == 1,
    "PRIMARY",
    "COMPETING_CANDIDATE",
)

case_score_output = case_score_output[
    [
        "case_id",
        "session_id",
        "dtc",
        "candidate_role",
        "primary_dtc",
        "static_score",
        "context_score",
        "any_activation",
        "strong_activation",
    ]
]

case_score_file = (
    OUTPUT_DIR / "context_rule_case_scores.csv"
)

case_score_output.to_csv(
    case_score_file,
    index=False,
)


# ================================================================
# OUTPUT 2 - RULE ACTIVATION SUMMARY
# ================================================================

activation_records = []

for dtc in DTC_RULES:

    rule_data = candidate_df[
        candidate_df["dtc"] == dtc
    ].copy()

    primary_data = rule_data[
        rule_data["is_primary_candidate"] == 1
    ]

    competing_data = rule_data[
        rule_data["is_primary_candidate"] == 0
    ]

    activation_records.append({

        "dtc": dtc,

        "candidate_appearances": len(rule_data),

        "primary_appearances": len(primary_data),

        "competing_appearances": len(competing_data),

        "mean_score_all": (
            rule_data["context_score"].mean()
            if len(rule_data) else np.nan
        ),

        "median_score_all": (
            rule_data["context_score"].median()
            if len(rule_data) else np.nan
        ),

        "max_score_all": (
            rule_data["context_score"].max()
            if len(rule_data) else np.nan
        ),

        "any_activation_pct_all": (
            100 * rule_data["any_activation"].mean()
            if len(rule_data) else np.nan
        ),

        "strong_activation_pct_all": (
            100 * rule_data["strong_activation"].mean()
            if len(rule_data) else np.nan
        ),

        "mean_score_primary": (
            primary_data["context_score"].mean()
            if len(primary_data) else np.nan
        ),

        "mean_score_competing": (
            competing_data["context_score"].mean()
            if len(competing_data) else np.nan
        ),

        "any_activation_pct_primary": (
            100 * primary_data["any_activation"].mean()
            if len(primary_data) else np.nan
        ),

        "any_activation_pct_competing": (
            100 * competing_data["any_activation"].mean()
            if len(competing_data) else np.nan
        ),

        "strong_activation_pct_primary": (
            100 * primary_data["strong_activation"].mean()
            if len(primary_data) else np.nan
        ),

        "strong_activation_pct_competing": (
            100 * competing_data["strong_activation"].mean()
            if len(competing_data) else np.nan
        ),
    })


activation_df = pd.DataFrame(
    activation_records
)

activation_file = (
    OUTPUT_DIR / "context_rule_activation_summary.csv"
)

activation_df.to_csv(
    activation_file,
    index=False,
)


# ================================================================
# OUTPUT 3 - FULL MODEL AND LEAVE-ONE-RULE-OUT ABLATION
# ================================================================

ablation_records = []

# Static-only baseline
static_ranks = []

for record in case_records:

    candidates = record["candidates"]
    primary_dtc = record["primary_dtc"]

    static_ranking = sorted(
        candidates,
        key=lambda dtc: (
            -STATIC_PRIORITY[dtc],
            dtc,
        )
    )

    static_ranks.append(
        static_ranking.index(primary_dtc) + 1
    )

static_frame = pd.DataFrame({
    "primary_rank": static_ranks
})

static_metrics = calculate_metrics(
    static_frame
)

ablation_records.append({
    "configuration": "STATIC_ONLY",
    "removed_rule": "NONE",
    **static_metrics,
})


# Full context-refined method
full_metrics = calculate_metrics(
    case_df[["primary_rank"]]
)

ablation_records.append({
    "configuration": "FULL_CONTEXT_MODEL",
    "removed_rule": "NONE",
    **full_metrics,
})


# Remove one DTC rule at a time
for removed_dtc in DTC_RULES:

    ranks = []

    for record in case_records:

        candidates = record["candidates"]
        primary_dtc = record["primary_dtc"]
        context_scores = record["context_scores"]

        ranking = rank_candidates(
            candidates,
            context_scores,
            removed_rule=removed_dtc,
        )

        ranked_dtcs = [
            item["dtc"] for item in ranking
        ]

        ranks.append(
            ranked_dtcs.index(primary_dtc) + 1
        )

    ablation_frame = pd.DataFrame({
        "primary_rank": ranks
    })

    metrics = calculate_metrics(
        ablation_frame
    )

    ablation_records.append({
        "configuration": "RULE_REMOVED",
        "removed_rule": removed_dtc,
        **metrics,
    })


ablation_df = pd.DataFrame(
    ablation_records
)

full_top1 = full_metrics["top1_accuracy"]
full_mrr = full_metrics["mrr"]
full_mean_rank = full_metrics["mean_primary_rank"]

ablation_df["top1_delta_vs_full"] = (
    ablation_df["top1_accuracy"] - full_top1
)

ablation_df["mrr_delta_vs_full"] = (
    ablation_df["mrr"] - full_mrr
)

ablation_df["mean_rank_delta_vs_full"] = (
    ablation_df["mean_primary_rank"] - full_mean_rank
)

ablation_file = (
    OUTPUT_DIR / "context_rule_ablation_summary.csv"
)

ablation_df.to_csv(
    ablation_file,
    index=False,
)


# ================================================================
# OUTPUT 4 - FAILURE CASES
# ================================================================

failure_records = []

for record in case_records:

    if record["primary_rank"] == 1:
        continue

    candidates = record["candidates"]
    context_scores = record["context_scores"]

    ranking = rank_candidates(
        candidates,
        context_scores,
    )

    primary_dtc = record["primary_dtc"]

    primary_item = next(
        item for item in ranking
        if item["dtc"] == primary_dtc
    )

    top_item = ranking[0]

    failure_records.append({

        "case_id": record["case_id"],

        "session_id": record["session_id"],

        "candidate_dtcs": record["candidate_dtcs"],

        "primary_dtc": primary_dtc,

        "primary_rank": record["primary_rank"],

        "top1_dtc": top_item["dtc"],

        "top1_static_score": top_item["static_score"],

        "top1_context_score": top_item["context_score"],

        "primary_static_score": primary_item["static_score"],

        "primary_context_score": primary_item["context_score"],

        "top1_context_advantage": (
            top_item["context_score"]
            - primary_item["context_score"]
        ),

        "full_ranking": record["full_ranking"],

    })


failure_df = pd.DataFrame(
    failure_records
)

failure_file = (
    OUTPUT_DIR / "context_rule_failure_cases.csv"
)

failure_df.to_csv(
    failure_file,
    index=False,
)


# ================================================================
# REPORT
# ================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("\n" + "=" * 72)
print("EXPERIMENT 04A - AUDIT SUMMARY")
print("=" * 72)

print("\n1. RULE ACTIVATION AND DISCRIMINATION")
print("-" * 72)

print(
    activation_df.round(4).to_string(index=False)
)

print("\n2. ABLATION RESULTS")
print("-" * 72)

display_columns = [
    "configuration",
    "removed_rule",
    "top1_accuracy",
    "mrr",
    "mean_primary_rank",
    "median_primary_rank",
    "top1_delta_vs_full",
    "mrr_delta_vs_full",
]

print(
    ablation_df[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\n3. FAILURE CASES")
print("-" * 72)

if failure_df.empty:
    print("No incorrectly ranked cases.")
else:
    print(
        failure_df.to_string(index=False)
    )

print("\n4. INTERPRETATION NOTES")
print("-" * 72)

print(
    "Activation percentages are calculated only over cases "
    "where the DTC was present as a candidate."
)

print(
    "Primary-versus-competing comparisons are descriptive. "
    "They do not establish statistical significance."
)

print(
    "Ablation removes one DTC's context contribution while "
    "retaining its candidate status and static priority."
)

print(
    "This audit does not tune thresholds or weights."
)

print("\n5. OUTPUT FILES")
print("-" * 72)

for path in [
    activation_file,
    case_score_file,
    ablation_file,
    failure_file,
]:
    print(path)

print("\nSTATUS: PASS")
