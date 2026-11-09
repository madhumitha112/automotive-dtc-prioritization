
"""
Experiment 04B
--------------
Rule Validity and Candidate Coverage Audit

Purpose
-------
Determine whether the current benchmark provides sufficient candidate
coverage to assess each CAN-context rule.

This experiment:
1. Counts primary and competing appearances for each DTC.
2. Measures pairwise candidate co-occurrence.
3. Identifies unique candidate-set patterns and repetitions.
4. Assesses rule testability using primary/competing appearances.
5. Identifies coverage gaps.

IMPORTANT
---------
This is a coverage audit only.

It does NOT:
- Modify the dataset.
- Modify Experiment 04 scores.
- Change thresholds or context weights.
- Tune or select rules.
- Use evaluation labels as algorithm inputs.

Inputs:
    results/diagnostic_cases.csv
    results/context_rule_case_scores.csv

Outputs:
    results/04b_rule_coverage_summary.csv
    results/04b_pairwise_candidate_coverage.csv
    results/04b_scenario_coverage.csv
    results/04b_coverage_gaps.csv
    results/04b_rule_validity_report.txt
"""

from pathlib import Path
from itertools import combinations

import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

CASES_FILE = RESULTS_DIR / "diagnostic_cases.csv"
SCORES_FILE = RESULTS_DIR / "context_rule_case_scores.csv"

EXPECTED_CASES = 48

DTC_LIST = [
    "P0171",
    "P0172",
    "P0128",
    "P0300",
    "P0130",
    "P0121",
    "P0335",
    "C0035",
]

OUTPUT_RULES = RESULTS_DIR / "04b_rule_coverage_summary.csv"
OUTPUT_PAIRS = RESULTS_DIR / "04b_pairwise_candidate_coverage.csv"
OUTPUT_SCENARIOS = RESULTS_DIR / "04b_scenario_coverage.csv"
OUTPUT_GAPS = RESULTS_DIR / "04b_coverage_gaps.csv"
OUTPUT_REPORT = RESULTS_DIR / "04b_rule_validity_report.txt"


# ============================================================
# HELPERS
# ============================================================

def parse_candidates(value):
    """Parse pipe-separated candidate DTC values."""

    return sorted(
        set(
            str(value).strip().upper().split("|")
        )
    )


def write_section(handle, title):
    handle.write("\n")
    handle.write("=" * 78 + "\n")
    handle.write(title + "\n")
    handle.write("=" * 78 + "\n")


# ============================================================
# LOAD INPUTS
# ============================================================

print("=" * 78)
print("EXPERIMENT 04B - RULE VALIDITY AND CANDIDATE COVERAGE")
print("=" * 78)

if not CASES_FILE.exists():
    raise FileNotFoundError(
        f"Missing input file: {CASES_FILE}"
    )

if not SCORES_FILE.exists():
    raise FileNotFoundError(
        f"Missing input file: {SCORES_FILE}"
    )

cases = pd.read_csv(CASES_FILE)
scores = pd.read_csv(SCORES_FILE)

print(f"\nDiagnostic cases loaded: {len(cases)}")
print(f"Candidate score records:  {len(scores)}")


# ============================================================
# VALIDATE INPUT STRUCTURE
# ============================================================

required_case_columns = {
    "case_id",
    "candidate_dtcs",
    "primary_dtc",
}

required_score_columns = {
    "case_id",
    "dtc",
    "candidate_role",
    "primary_dtc",
    "context_score",
}

missing_cases = (
    required_case_columns - set(cases.columns)
)

missing_scores = (
    required_score_columns - set(scores.columns)
)

if missing_cases:
    raise ValueError(
        f"Diagnostic cases missing columns: {sorted(missing_cases)}"
    )

if missing_scores:
    raise ValueError(
        f"Candidate score file missing columns: {sorted(missing_scores)}"
    )

if len(cases) != EXPECTED_CASES:
    raise ValueError(
        f"Expected {EXPECTED_CASES} cases, found {len(cases)}."
    )

if cases["case_id"].duplicated().any():
    raise ValueError("Duplicate case IDs in diagnostic cases.")

if scores.duplicated(["case_id", "dtc"]).any():
    raise ValueError(
        "Duplicate case_id + DTC combinations in score file."
    )

cases["case_id"] = cases["case_id"].astype(str)
scores["case_id"] = scores["case_id"].astype(str)

cases["primary_dtc"] = (
    cases["primary_dtc"].astype(str).str.strip().str.upper()
)

scores["dtc"] = (
    scores["dtc"].astype(str).str.strip().str.upper()
)

scores["candidate_role"] = (
    scores["candidate_role"].astype(str).str.strip().str.upper()
)

scores["context_score"] = pd.to_numeric(
    scores["context_score"],
    errors="raise",
)

if not scores["context_score"].between(0, 1).all():
    raise ValueError("Context scores outside the range [0, 1].")

if set(scores["candidate_role"].unique()) - {
    "PRIMARY",
    "COMPETING_CANDIDATE",
}:
    raise ValueError("Unexpected candidate_role value.")

if set(scores["dtc"].unique()) - set(DTC_LIST):
    raise ValueError("Unexpected DTC found in score records.")

if set(cases["case_id"]) != set(scores["case_id"]):
    raise ValueError(
        "Case IDs differ between diagnostic cases and score records."
    )


# ============================================================
# VALIDATE CANDIDATE RECORDS AGAINST CASE DEFINITIONS
# ============================================================

case_lookup = cases.set_index("case_id")

for _, row in cases.iterrows():

    case_id = str(row["case_id"])
    candidates = parse_candidates(row["candidate_dtcs"])
    primary = str(row["primary_dtc"]).strip().upper()

    if primary not in candidates:
        raise ValueError(
            f"{case_id}: primary DTC missing from candidate set."
        )

    observed = sorted(
        scores.loc[
            scores["case_id"] == case_id,
            "dtc"
        ].tolist()
    )

    if observed != candidates:
        raise ValueError(
            f"{case_id}: candidate mismatch. "
            f"Cases={candidates}, scores={observed}"
        )

    primary_records = scores[
        (scores["case_id"] == case_id)
        & (scores["candidate_role"] == "PRIMARY")
    ]

    if len(primary_records) != 1:
        raise ValueError(
            f"{case_id}: expected exactly one PRIMARY score record."
        )

    if primary_records.iloc[0]["dtc"] != primary:
        raise ValueError(
            f"{case_id}: primary DTC mismatch between input files."
        )


# ============================================================
# 1. RULE COVERAGE SUMMARY
# ============================================================

rule_rows = []

for dtc in DTC_LIST:

    dtc_scores = scores[scores["dtc"] == dtc]

    primary = dtc_scores[
        dtc_scores["candidate_role"] == "PRIMARY"
    ]

    competing = dtc_scores[
        dtc_scores["candidate_role"] == "COMPETING_CANDIDATE"
    ]

    primary_count = len(primary)
    competing_count = len(competing)
    total_count = len(dtc_scores)

    primary_activation = (
        primary["context_score"].gt(0).mean()
        if primary_count else 0.0
    )

    competing_activation = (
        competing["context_score"].gt(0).mean()
        if competing_count else 0.0
    )

    primary_mean = (
        primary["context_score"].mean()
        if primary_count else None
    )

    competing_mean = (
        competing["context_score"].mean()
        if competing_count else None
    )

    primary_strong = (
        primary["context_score"].ge(0.5).mean()
        if primary_count else 0.0
    )

    competing_strong = (
        competing["context_score"].ge(0.5).mean()
        if competing_count else 0.0
    )

    if primary_count == 0:
        assessment = "NO_PRIMARY_CASES"

    elif competing_count == 0:
        assessment = "NO_COMPETING_CASES"

    elif primary_count < 4 or competing_count < 4:
        assessment = "LIMITED_COVERAGE"

    else:
        assessment = "BASIC_COMPARATIVE_COVERAGE"

    rule_rows.append({
        "dtc": dtc,
        "total_candidate_appearances": total_count,
        "primary_appearances": primary_count,
        "competing_appearances": competing_count,
        "primary_activation_rate": primary_activation,
        "competing_activation_rate": competing_activation,
        "primary_mean_context_score": primary_mean,
        "competing_mean_context_score": competing_mean,
        "primary_strong_activation_rate": primary_strong,
        "competing_strong_activation_rate": competing_strong,
        "coverage_assessment": assessment,
    })

rule_summary = pd.DataFrame(rule_rows)

rule_summary.to_csv(
    OUTPUT_RULES,
    index=False,
)


# ============================================================
# 2. PAIRWISE CANDIDATE COVERAGE
# ============================================================

pair_rows = []

for dtc_a, dtc_b in combinations(DTC_LIST, 2):

    cooccurring_cases = []

    for _, row in cases.iterrows():

        candidates = parse_candidates(
            row["candidate_dtcs"]
        )

        if dtc_a in candidates and dtc_b in candidates:
            cooccurring_cases.append(row)

    total_pair_cases = len(cooccurring_cases)

    a_primary = sum(
        str(row["primary_dtc"]).upper() == dtc_a
        for row in cooccurring_cases
    )

    b_primary = sum(
        str(row["primary_dtc"]).upper() == dtc_b
        for row in cooccurring_cases
    )

    pair_rows.append({
        "dtc_a": dtc_a,
        "dtc_b": dtc_b,
        "cooccurrence_cases": total_pair_cases,
        "dtc_a_primary_cases": a_primary,
        "dtc_b_primary_cases": b_primary,
        "both_primary_labels_observed": int(
            a_primary > 0 and b_primary > 0
        ),
        "pair_coverage": (
            "NO_CO_OCCURRENCE"
            if total_pair_cases == 0
            else
            "ONE_DIRECTION_ONLY"
            if a_primary == 0 or b_primary == 0
            else
            "BOTH_DIRECTIONS_OBSERVED"
        ),
    })

pair_summary = pd.DataFrame(pair_rows)

pair_summary.to_csv(
    OUTPUT_PAIRS,
    index=False,
)


# ============================================================
# 3. UNIQUE SCENARIO / CANDIDATE-SET COVERAGE
# ============================================================

scenario_rows = []

for _, row in cases.iterrows():

    candidates = parse_candidates(row["candidate_dtcs"])
    primary = str(row["primary_dtc"]).strip().upper()

    scenario_rows.append({
        "case_id": str(row["case_id"]),
        "primary_dtc": primary,
        "candidate_set": "|".join(candidates),
        "candidate_count": len(candidates),
        "scenario_signature": (
            primary + "::" + "|".join(candidates)
        ),
    })

scenario_df = pd.DataFrame(scenario_rows)

scenario_summary = (
    scenario_df
    .groupby(
        [
            "scenario_signature",
            "primary_dtc",
            "candidate_set",
            "candidate_count",
        ],
        as_index=False
    )
    .agg(
        repetition_count=("case_id", "count"),
        case_ids=("case_id", lambda x: "|".join(sorted(x))),
    )
    .sort_values(
        ["primary_dtc", "candidate_set"]
    )
)

scenario_summary.to_csv(
    OUTPUT_SCENARIOS,
    index=False,
)


# ============================================================
# 4. COVERAGE GAPS
# ============================================================

gap_rows = []

for _, row in rule_summary.iterrows():

    dtc = row["dtc"]
    primary_count = int(row["primary_appearances"])
    competing_count = int(row["competing_appearances"])

    if primary_count == 0:
        gap_rows.append({
            "dtc": dtc,
            "gap_type": "NO_PRIMARY_CASES",
            "evidence": "Rule has no primary-labelled cases.",
            "recommended_action": "Add a suitable primary scenario."
        })

    if competing_count == 0:
        gap_rows.append({
            "dtc": dtc,
            "gap_type": "NO_COMPETING_CASES",
            "evidence": (
                "Rule is observed only when this DTC is primary."
            ),
            "recommended_action": (
                "Add cases where this DTC competes with other candidates."
            )
        })

    if primary_count < 4 or competing_count < 4:
        gap_rows.append({
            "dtc": dtc,
            "gap_type": "LOW_SAMPLE_COVERAGE",
            "evidence": (
                f"Primary={primary_count}; competing={competing_count}."
            ),
            "recommended_action": (
                "Increase scenario diversity before drawing strong conclusions."
            )
        })

    if (
        primary_count > 0
        and competing_count > 0
        and row["primary_mean_context_score"] is not None
        and row["competing_mean_context_score"] is not None
    ):

        difference = (
            row["primary_mean_context_score"]
            - row["competing_mean_context_score"]
        )

        if difference <= 0:
            gap_rows.append({
                "dtc": dtc,
                "gap_type": "NO_POSITIVE_PRIMARY_SCORE_SEPARATION",
                "evidence": (
                    f"Primary-minus-competing mean score={difference:.4f}."
                ),
                "recommended_action": (
                    "Inspect overlapping evidence and rule specificity."
                )
            })

gap_summary = pd.DataFrame(
    gap_rows,
    columns=[
        "dtc",
        "gap_type",
        "evidence",
        "recommended_action",
    ]
)

gap_summary.to_csv(
    OUTPUT_GAPS,
    index=False,
)


# ============================================================
# 5. REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as report:

    write_section(
        report,
        "EXPERIMENT 04B - RULE COVERAGE SUMMARY"
    )

    report.write(
        f"Diagnostic cases: {len(cases)}\n"
    )

    report.write(
        f"Candidate score records: {len(scores)}\n"
    )

    report.write(
        f"Unique DTCs: {scores['dtc'].nunique()}\n"
    )

    report.write(
        f"Unique candidate signatures: "
        f"{scenario_summary['scenario_signature'].nunique()}\n"
    )

    report.write(
        f"Scenario signature repetitions: "
        f"{scenario_summary['repetition_count'].describe().to_dict()}\n"
    )

    write_section(
        report,
        "1. RULE COVERAGE"
    )

    report.write(
        rule_summary.to_string(index=False)
    )

    write_section(
        report,
        "2. PAIRWISE CANDIDATE COVERAGE"
    )

    report.write(
        pair_summary[
            pair_summary["cooccurrence_cases"] > 0
        ].to_string(index=False)
    )

    write_section(
        report,
        "3. UNIQUE SCENARIO COVERAGE"
    )

    report.write(
        scenario_summary.to_string(index=False)
    )

    write_section(
        report,
        "4. COVERAGE GAPS"
    )

    if gap_summary.empty:
        report.write("No coverage gaps identified.\n")
    else:
        report.write(
            gap_summary.to_string(index=False)
        )

    write_section(
        report,
        "5. INTERPRETATION LIMITATIONS"
    )

    report.write(
        "1. Candidate appearances are not independent experiments.\n"
        "2. Repeated scenario signatures do not add scenario diversity.\n"
        "3. Primary-versus-competing score differences are descriptive.\n"
        "4. Activation does not establish diagnostic specificity.\n"
        "5. A rule with no competing appearances cannot be assessed "
        "for discrimination.\n"
        "6. This audit does not establish statistical significance.\n"
        "7. No thresholds, weights, or ranking rules were changed.\n"
    )

    write_section(
        report,
        "6. OUTPUT FILES"
    )

    for path in [
        OUTPUT_RULES,
        OUTPUT_PAIRS,
        OUTPUT_SCENARIOS,
        OUTPUT_GAPS,
        OUTPUT_REPORT,
    ]:
        report.write(str(path) + "\n")

    report.write("\nSTATUS: PASS\n")


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("RULE COVERAGE SUMMARY")
print("=" * 78)

print(
    rule_summary[
        [
            "dtc",
            "primary_appearances",
            "competing_appearances",
            "coverage_assessment",
        ]
    ].to_string(index=False)
)

print("\nUnique candidate signatures:")
print(
    scenario_summary[
        [
            "scenario_signature",
            "repetition_count",
        ]
    ].to_string(index=False)
)

print("\nCoverage gaps identified:", len(gap_summary))

print("\nOutput files:")
print(OUTPUT_RULES)
print(OUTPUT_PAIRS)
print(OUTPUT_SCENARIOS)
print(OUTPUT_GAPS)
print(OUTPUT_REPORT)

print("\nSTATUS: PASS")
