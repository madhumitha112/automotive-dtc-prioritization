
"""
Experiment 03V4
---------------
Static DTC prioritization baseline on the expanded V4 benchmark.

Uses the exact frozen static priority table from Experiment 03.
No priorities are learned or tuned using V4 evaluation labels.
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "diagnostic_cases_v4.csv"
OUTPUT_DIR = ROOT / "results"
OUTPUT_FILE = OUTPUT_DIR / "static_baseline_results_v4.csv"


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

EXPECTED_CASES = 24


# ---------------------------------------------------------------------
# FROZEN V3 STATIC PRIORITY TABLE
# DO NOT MODIFY THESE VALUES FOR V4.
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"V4 diagnostic case file not found:\n{INPUT_FILE}"
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
        f"Expected {EXPECTED_CASES} V4 cases, found {len(cases)}."
    )

if cases["case_id"].duplicated().any():
    raise ValueError("Duplicate V4 case IDs detected.")


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

    if len(candidates) != len(set(candidates)):
        raise ValueError(
            f"{case_id} contains duplicate candidates."
        )

    if int(case["candidate_count"]) != len(candidates):
        raise ValueError(
            f"Candidate count mismatch in {case_id}."
        )

    unknown_dtcs = [
        dtc for dtc in candidates
        if dtc not in STATIC_PRIORITY
    ]

    if unknown_dtcs:
        raise ValueError(
            f"{case_id} contains DTCs missing from static "
            f"priority table: {unknown_dtcs}"
        )

    primary_dtc = str(
        case["primary_dtc"]
    ).strip().upper()

    if primary_dtc not in candidates:
        raise ValueError(
            f"Primary DTC {primary_dtc} is not in candidates "
            f"for {case_id}."
        )

    # Static ranking: independent of primary DTC labels.
    ranked_candidates = sorted(
        candidates,
        key=lambda dtc: (
            -STATIC_PRIORITY[dtc],
            dtc,
        ),
    )

    # Evaluation only, after ranking is complete.
    primary_rank = (
        ranked_candidates.index(primary_dtc) + 1
    )

    top_1_dtc = ranked_candidates[0]
    top_3_dtcs = ranked_candidates[:3]

    results.append({
        "case_id": case_id,
        "session_id": case["session_id"],
        "candidate_dtcs": "|".join(candidates),
        "candidate_count": len(candidates),
        "static_ranking": "|".join(ranked_candidates),
        "static_top_1_dtc": top_1_dtc,
        "primary_dtc": primary_dtc,
        "primary_rank_static": primary_rank,
        "reciprocal_rank_static": 1.0 / primary_rank,
        "top_1_correct_static": int(
            top_1_dtc == primary_dtc
        ),
        "top_3_correct_static": int(
            primary_dtc in top_3_dtcs
        ),
        "static_score_top_1": STATIC_PRIORITY[top_1_dtc],
    })


# ---------------------------------------------------------------------
# RESULTS VALIDATION
# ---------------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="case_id"
).reset_index(drop=True)

if len(results_df) != EXPECTED_CASES:
    raise ValueError("Incorrect number of V4 result rows.")

if results_df["case_id"].duplicated().any():
    raise ValueError("Duplicate case IDs in V4 results.")

if results_df["primary_rank_static"].lt(1).any():
    raise ValueError("Invalid primary DTC rank.")

if results_df["primary_rank_static"].gt(
    results_df["candidate_count"]
).any():
    raise ValueError("Primary rank exceeds candidate count.")


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

top_1_accuracy = results_df["top_1_correct_static"].mean()
top_3_accuracy = results_df["top_3_correct_static"].mean()
mrr = results_df["reciprocal_rank_static"].mean()
mean_rank = results_df["primary_rank_static"].mean()
median_rank = results_df["primary_rank_static"].median()


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

OUTPUT_DIR.mkdir(
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
print("EXPERIMENT 03V4 - FROZEN STATIC DTC BASELINE")
print("=" * 70)

print(f"\nCases evaluated: {len(results_df)}")

print("\nFrozen V3 priority table:")
for dtc, score in sorted(
    STATIC_PRIORITY.items(),
    key=lambda item: -item[1],
):
    print(f"  {dtc}: {score:.2f}")

print("\nV4 baseline metrics:")
print(f"  Top-1 accuracy:       {top_1_accuracy:.4f}")
print(f"  Top-3 accuracy:       {top_3_accuracy:.4f}")
print(f"  Mean Reciprocal Rank: {mrr:.4f}")
print(f"  Mean primary rank:    {mean_rank:.4f}")
print(f"  Median primary rank:  {median_rank:.4f}")

print("\nFirst 10 rankings:")
print(
    results_df[
        [
            "case_id",
            "candidate_dtcs",
            "static_ranking",
            "primary_dtc",
            "primary_rank_static",
        ]
    ].head(10).to_string(index=False)
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nSTATUS: PASS")
