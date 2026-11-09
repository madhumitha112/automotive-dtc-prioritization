"""
Experiment 03V4
-------------
Static DTC prioritization baseline.

Purpose:
    Establish a fixed, conventional DTC-priority baseline against which
    the CAN-context refinement method will later be compared.

IMPORTANT:
    This experiment must NOT use:
        - primary_dtc
        - secondary_dtcs
        - context_condition
        - CAN signal values

The baseline uses only:
    - candidate DTCs
    - a fixed, pre-defined static priority table

The priority table is frozen before evaluating results.
No scores are learned or tuned from the benchmark data.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "expanded_v4" / "diagnostic_cases_v4.csv"
OUTPUT_DIR = ROOT / "results"
OUTPUT_FILE = OUTPUT_DIR / "expanded_v4" / "static_baseline_results_v4.csv"


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

EXPECTED_CASES = 24


# ---------------------------------------------------------------------
# STATIC DTC PRIORITY TABLE
# ---------------------------------------------------------------------
#
# Fixed engineering-priority baseline.
#
# Higher score = higher diagnostic priority.
#
# These values are deliberately defined independently of the
# benchmark's primary-DTC labels.
#
# They are NOT learned from the generated cases.
# ---------------------------------------------------------------------

STATIC_PRIORITY = {
    # High diagnostic priority
    "P0335": 1.00,   # Crankshaft position / engine-speed related
    "P0300": 0.95,   # Random/multiple cylinder misfire
    "C0035": 0.95,   # Wheel-speed sensor related

    # Medium-high priority
    "P0121": 0.80,   # Throttle/pedal position performance

    # Medium priority
    "P0130": 0.70,   # Oxygen sensor circuit
    "P0171": 0.65,   # System too lean
    "P0172": 0.65,   # System too rich

    # Lower priority
    "P0128": 0.55,   # Coolant thermostat / temperature behavior
}


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Diagnostic case file not found:\n{INPUT_FILE}"
    )

cases = pd.read_csv(INPUT_FILE)

required_columns = {
    "case_id",
    "session_id",
    "candidate_dtcs",
    "candidate_count",
    "primary_dtc",
}

missing_columns = sorted(
    required_columns - set(cases.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if len(cases) != EXPECTED_CASES:
    raise ValueError(
        f"Expected {EXPECTED_CASES} cases, found {len(cases)}."
    )

if set(STATIC_PRIORITY.keys()) != {
    "P0335",
    "P0300",
    "C0035",
    "P0121",
    "P0130",
    "P0171",
    "P0172",
    "P0128",
}:
    raise ValueError(
        "Static priority table does not contain the expected DTC set."
    )


# ---------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------

def parse_candidates(value):
    """Convert pipe-separated candidate DTCs into a list."""

    if pd.isna(value):
        return []

    return [
        dtc.strip().upper()
        for dtc in str(value).split("|")
        if dtc.strip()
    ]


# ---------------------------------------------------------------------
# RANK EACH CASE
# ---------------------------------------------------------------------

results = []

for _, case in cases.iterrows():

    case_id = case["case_id"]

    candidates = parse_candidates(
        case["candidate_dtcs"]
    )

    if len(candidates) < 2:
        raise ValueError(
            f"{case_id} has fewer than two candidates."
        )

    unknown_dtcs = [
        dtc
        for dtc in candidates
        if dtc not in STATIC_PRIORITY
    ]

    if unknown_dtcs:
        raise ValueError(
            f"{case_id} contains DTCs missing from static priority "
            f"table: {unknown_dtcs}"
        )

    primary_dtc = str(
        case["primary_dtc"]
    ).strip().upper()

    if primary_dtc not in candidates:
        raise ValueError(
            f"Primary DTC {primary_dtc} is not in candidate set "
            f"for {case_id}."
        )

    # -------------------------------------------------------------
    # Static ranking
    # -------------------------------------------------------------

    ranked_candidates = sorted(
        candidates,
        key=lambda dtc: (
            -STATIC_PRIORITY[dtc],
            dtc,
        ),
    )

    # -------------------------------------------------------------
    # Rank of the evaluation target
    #
    # This uses primary_dtc ONLY after ranking has been generated.
    # It is an evaluation operation, not an algorithm input.
    # -------------------------------------------------------------

    primary_rank = (
        ranked_candidates.index(primary_dtc) + 1
    )

    top_1_dtc = ranked_candidates[0]

    top_3_dtc = ranked_candidates[:3]

    top_1_correct = (
        top_1_dtc == primary_dtc
    )

    top_3_correct = (
        primary_dtc in top_3_dtc
    )

    reciprocal_rank = 1.0 / primary_rank

    results.append(
        {
            "case_id": case_id,
            "session_id": case["session_id"],

            "candidate_dtcs": "|".join(candidates),
            "candidate_count": len(candidates),

            "static_ranking": "|".join(
                ranked_candidates
            ),

            "static_top_1_dtc": top_1_dtc,

            "primary_dtc": primary_dtc,

            "primary_rank_static": primary_rank,

            "reciprocal_rank_static": reciprocal_rank,

            "top_1_correct_static": int(
                top_1_correct
            ),

            "top_3_correct_static": int(
                top_3_correct
            ),

            "static_score_top_1": STATIC_PRIORITY[
                top_1_dtc
            ],
        }
    )


# ---------------------------------------------------------------------
# RESULTS DATAFRAME
# ---------------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="case_id"
).reset_index(drop=True)


# ---------------------------------------------------------------------
# RESULT VALIDATION
# ---------------------------------------------------------------------

if len(results_df) != EXPECTED_CASES:
    raise ValueError(
        f"Expected {EXPECTED_CASES} results, "
        f"created {len(results_df)}."
    )

if results_df["case_id"].duplicated().any():
    raise ValueError(
        "Duplicate case IDs detected in static baseline results."
    )

if results_df["primary_rank_static"].lt(1).any():
    raise ValueError(
        "Invalid primary DTC rank detected."
    )

if results_df["primary_rank_static"].gt(
    results_df["candidate_count"]
).any():
    raise ValueError(
        "Primary DTC rank exceeds candidate count."
    )

if not results_df["reciprocal_rank_static"].between(
    0, 1
).all():
    raise ValueError(
        "Invalid reciprocal-rank values detected."
    )


# ---------------------------------------------------------------------
# AGGREGATE METRICS
# ---------------------------------------------------------------------

top_1_accuracy = (
    results_df["top_1_correct_static"].mean()
)

top_3_accuracy = (
    results_df["top_3_correct_static"].mean()
)

mrr = (
    results_df["reciprocal_rank_static"].mean()
)

mean_rank = (
    results_df["primary_rank_static"].mean()
)

median_rank = (
    results_df["primary_rank_static"].median()
)


# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

print("=" * 70)
print("EXPERIMENT 03 - STATIC DTC PRIORITIZATION BASELINE")
print("=" * 70)

print("\nStatic priority table:")

for dtc, score in sorted(
    STATIC_PRIORITY.items(),
    key=lambda item: -item[1],
):
    print(
        f"  {dtc}: {score:.2f}"
    )

print("\nCases evaluated:")
print(f"  {len(results_df)}")

print("\nBaseline metrics:")
print(
    f"  Top-1 accuracy: {top_1_accuracy:.4f}"
)
print(
    f"  Top-3 accuracy: {top_3_accuracy:.4f}"
)
print(
    f"  Mean Reciprocal Rank: {mrr:.4f}"
)
print(
    f"  Mean primary rank: {mean_rank:.4f}"
)
print(
    f"  Median primary rank: {median_rank:.4f}"
)

print("\nFirst 10 rankings:")

print(
    results_df[
        [
            "case_id",
            "candidate_dtcs",
            "static_ranking",
            "primary_dtc",
            "primary_rank_static",
            "reciprocal_rank_static",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nSTATUS: PASS")