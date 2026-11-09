
"""
EXPERIMENT 07 - V4 Benchmark Coverage and Validity Analysis

Purpose:
Evaluate DTC coverage, candidate-pair balance, scenario-template
diversity, repetition structure and case-level ranking outcomes.

This is a descriptive benchmark audit.
It does not modify the dataset, ranking methods, thresholds or weights.
"""

from pathlib import Path
from itertools import combinations

import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

CASES_FILE = RESULTS_DIR / "diagnostic_cases_v4.csv"
SCENARIO_FILE = RESULTS_DIR / "scenario_manifest_v4.csv"
COMPARISON_FILE = RESULTS_DIR / "v4_comparative_case_results.csv"

EXPECTED_CASES = 24
EXPECTED_TEMPLATES = 12
EXPECTED_REPETITIONS = 2

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

OUTPUT_DTC = RESULTS_DIR / "v4_validity_dtc_coverage.csv"
OUTPUT_PAIRS = RESULTS_DIR / "v4_validity_pair_coverage.csv"
OUTPUT_TEMPLATES = RESULTS_DIR / "v4_validity_template_coverage.csv"
OUTPUT_CASES = RESULTS_DIR / "v4_validity_case_checks.csv"
OUTPUT_REPORT = RESULTS_DIR / "v4_coverage_validity_report.txt"
OUTPUT_RUN = RESULTS_DIR / "v4_coverage_validity_run.txt"


# ============================================================
# HELPERS
# ============================================================

def parse_candidates(value):
    """Parse and normalize pipe-separated candidate DTCs."""

    if pd.isna(value):
        return []

    return sorted({
        item.strip().upper()
        for item in str(value).split("|")
        if item.strip()
    })


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def write_section(handle, title):
    handle.write("\n")
    handle.write("=" * 78 + "\n")
    handle.write(title + "\n")
    handle.write("=" * 78 + "\n")


# ============================================================
# LOAD AND VALIDATE INPUTS
# ============================================================

def load_inputs():

    for path in [
        CASES_FILE,
        SCENARIO_FILE,
        COMPARISON_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    cases = pd.read_csv(CASES_FILE)
    scenarios = pd.read_csv(SCENARIO_FILE)
    comparison = pd.read_csv(COMPARISON_FILE)

    required_cases = {
        "case_id",
        "session_id",
        "candidate_dtcs",
        "candidate_count",
        "observation_start_s",
        "observation_end_s",
        "observation_duration_s",
        "primary_dtc",
        "secondary_dtcs",
        "context_condition",
        "template_id",
        "repetition",
    }

    required_scenarios = {
        "session_id",
        "template_id",
        "repetition",
        "primary_dtc",
        "candidate_dtcs",
        "context_condition",
        "coverage_role",
        "context_design",
    }

    required_comparison = {
        "case_id",
        "primary_dtc",
        "static_primary_rank",
        "context_primary_rank",
        "static_top1_correct",
        "context_top1_correct",
        "rank_outcome",
        "top1_outcome",
    }

    for name, frame, required in [
        ("diagnostic cases", cases, required_cases),
        ("scenario manifest", scenarios, required_scenarios),
        ("comparative results", comparison, required_comparison),
    ]:
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

    if len(cases) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_CASES} cases; found {len(cases)}."
        )

    if len(scenarios) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_CASES} scenario rows; "
            f"found {len(scenarios)}."
        )

    if len(comparison) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_CASES} comparison rows; "
            f"found {len(comparison)}."
        )

    for name, frame, column in [
        ("cases", cases, "case_id"),
        ("scenarios", scenarios, "session_id"),
        ("comparison", comparison, "case_id"),
    ]:
        if frame[column].duplicated().any():
            raise ValueError(
                f"Duplicate {column} values in {name}."
            )

    cases["case_id"] = cases["case_id"].astype(str)
    cases["session_id"] = cases["session_id"].astype(str)
    scenarios["session_id"] = scenarios["session_id"].astype(str)
    comparison["case_id"] = comparison["case_id"].astype(str)

    if set(cases["case_id"]) != set(comparison["case_id"]):
        raise ValueError(
            "Case IDs differ between cases and comparative results."
        )

    if set(cases["session_id"]) != set(scenarios["session_id"]):
        raise ValueError(
            "Session IDs differ between cases and scenario manifest."
        )

    # Validate comparison primary labels against case definitions.
    check = cases[
        ["case_id", "primary_dtc"]
    ].merge(
        comparison[
            ["case_id", "primary_dtc"]
        ],
        on="case_id",
        suffixes=("_case", "_comparison"),
        validate="one_to_one",
    )

    if not (
        check["primary_dtc_case"].astype(str).str.upper()
        == check["primary_dtc_comparison"].astype(str).str.upper()
    ).all():
        raise ValueError(
            "Primary DTC labels differ between cases and comparison."
        )

    # Validate scenario manifest against case definitions.
    scenario_check = cases[
        [
            "session_id",
            "template_id",
            "repetition",
            "primary_dtc",
            "candidate_dtcs",
            "context_condition",
        ]
    ].merge(
        scenarios[
            [
                "session_id",
                "template_id",
                "repetition",
                "primary_dtc",
                "candidate_dtcs",
                "context_condition",
            ]
        ],
        on="session_id",
        suffixes=("_case", "_scenario"),
        validate="one_to_one",
    )

    fields_to_compare = [
        "template_id",
        "repetition",
        "primary_dtc",
        "candidate_dtcs",
        "context_condition",
    ]

    for field in fields_to_compare:

        left = (
            scenario_check[f"{field}_case"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        right = (
            scenario_check[f"{field}_scenario"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        if not (left == right).all():
            raise ValueError(
                f"Scenario manifest mismatch in field: {field}"
            )

    return cases, scenarios, comparison


# ============================================================
# CASE-LEVEL INTEGRITY CHECKS
# ============================================================

def build_case_checks(cases, scenarios, comparison):

    scenario_lookup = scenarios.set_index("session_id")
    comparison_lookup = comparison.set_index("case_id")

    rows = []

    for _, case in cases.iterrows():

        case_id = str(case["case_id"])
        session_id = str(case["session_id"])

        candidates = parse_candidates(
            case["candidate_dtcs"]
        )

        primary = normalize_text(
            case["primary_dtc"]
        ).upper()

        secondary = normalize_text(
            case["secondary_dtcs"]
        ).upper()

        template_id = normalize_text(
            case["template_id"]
        )

        repetition = int(case["repetition"])

        start = int(case["observation_start_s"])
        end = int(case["observation_end_s"])
        duration = int(case["observation_duration_s"])

        scenario = scenario_lookup.loc[session_id]
        result = comparison_lookup.loc[case_id]

        candidate_count_valid = (
            len(candidates) == int(case["candidate_count"])
        )

        candidate_uniqueness_valid = (
            len(candidates) == len(set(candidates))
        )

        primary_in_candidates = primary in candidates

        secondary_in_candidates = (
            secondary in candidates
        )

        observation_interval_valid = (
            start < end and duration == end - start
        )

        repetition_valid = (
            repetition in [1, 2]
        )

        template_match = (
            template_id == str(scenario["template_id"])
        )

        context_match = (
            normalize_text(case["context_condition"])
            == normalize_text(scenario["context_condition"])
        )

        static_rank = int(result["static_primary_rank"])
        context_rank = int(result["context_primary_rank"])

        ranks_valid = (
            static_rank in [1, 2]
            and context_rank in [1, 2]
        )

        all_valid = all([
            candidate_count_valid,
            candidate_uniqueness_valid,
            primary_in_candidates,
            secondary_in_candidates,
            observation_interval_valid,
            repetition_valid,
            template_match,
            context_match,
            ranks_valid,
        ])

        rows.append({
            "case_id": case_id,
            "session_id": session_id,
            "template_id": template_id,
            "repetition": repetition,
            "primary_dtc": primary,
            "candidate_dtcs": "|".join(candidates),
            "candidate_count": len(candidates),
            "context_condition": normalize_text(
                case["context_condition"]
            ),
            "candidate_count_valid": int(candidate_count_valid),
            "candidate_uniqueness_valid": int(
                candidate_uniqueness_valid
            ),
            "primary_in_candidates": int(
                primary_in_candidates
            ),
            "secondary_in_candidates": int(
                secondary_in_candidates
            ),
            "observation_interval_valid": int(
                observation_interval_valid
            ),
            "repetition_valid": int(repetition_valid),
            "template_manifest_match": int(template_match),
            "context_manifest_match": int(context_match),
            "ranking_positions_valid": int(ranks_valid),
            "static_primary_rank": static_rank,
            "context_primary_rank": context_rank,
            "static_top1_correct": int(
                result["static_top1_correct"]
            ),
            "context_top1_correct": int(
                result["context_top1_correct"]
            ),
            "case_integrity_status": (
                "PASS" if all_valid else "FAIL"
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# DTC COVERAGE
# ============================================================

def build_dtc_coverage(cases):

    rows = []

    for dtc in DTC_LIST:

        primary_cases = cases[
            cases["primary_dtc"].astype(str).str.upper() == dtc
        ]

        candidate_mask = cases["candidate_dtcs"].apply(
            lambda value: dtc in parse_candidates(value)
        )

        candidate_cases = cases[candidate_mask]

        competing_cases = candidate_cases[
            candidate_cases["primary_dtc"].astype(str).str.upper()
            != dtc
        ]

        templates_primary = primary_cases[
            "template_id"
        ].nunique()

        templates_competing = competing_cases[
            "template_id"
        ].nunique()

        if len(primary_cases) == 0:
            assessment = "NO_PRIMARY_COVERAGE"
        elif len(competing_cases) == 0:
            assessment = "NO_COMPETING_COVERAGE"
        elif templates_primary < 2 or templates_competing < 2:
            assessment = "LIMITED_TEMPLATE_COVERAGE"
        else:
            assessment = "OBSERVED_BOTH_ROLES"

        rows.append({
            "dtc": dtc,
            "primary_case_count": len(primary_cases),
            "competing_case_count": len(competing_cases),
            "total_candidate_appearances": len(candidate_cases),
            "primary_template_count": templates_primary,
            "competing_template_count": templates_competing,
            "primary_case_share": (
                len(primary_cases) / len(cases)
            ),
            "candidate_case_share": (
                len(candidate_cases) / len(cases)
            ),
            "coverage_assessment": assessment,
        })

    return pd.DataFrame(rows)


# ============================================================
# PAIRWISE CANDIDATE COVERAGE
# ============================================================

def build_pair_coverage(cases):

    pair_counts = {}

    for _, case in cases.iterrows():

        candidates = parse_candidates(
            case["candidate_dtcs"]
        )

        if len(candidates) < 2:
            continue

        primary = normalize_text(
            case["primary_dtc"]
        ).upper()

        for dtc_a, dtc_b in combinations(candidates, 2):

            # Canonical ordering prevents pair-order mismatches.
            key = tuple(sorted((dtc_a, dtc_b)))

            if key not in pair_counts:
                pair_counts[key] = {
                    "cooccurrence_cases": 0,
                    "a_primary_cases": 0,
                    "b_primary_cases": 0,
                    "template_ids": set(),
                    "context_conditions": set(),
                }

            item = pair_counts[key]

            item["cooccurrence_cases"] += 1

            if primary == key[0]:
                item["a_primary_cases"] += 1
            elif primary == key[1]:
                item["b_primary_cases"] += 1

            item["template_ids"].add(
                str(case["template_id"])
            )

            item["context_conditions"].add(
                str(case["context_condition"])
            )

    rows = []

    for dtc_a, dtc_b in combinations(
        sorted(DTC_LIST), 2
    ):

        key = (dtc_a, dtc_b)

        item = pair_counts.get(
            key,
            {
                "cooccurrence_cases": 0,
                "a_primary_cases": 0,
                "b_primary_cases": 0,
                "template_ids": set(),
                "context_conditions": set(),
            },
        )

        a_count = item["a_primary_cases"]
        b_count = item["b_primary_cases"]
        total = item["cooccurrence_cases"]

        if total == 0:
            assessment = "NO_CO_OCCURRENCE"
        elif a_count == 0 or b_count == 0:
            assessment = "ONE_PRIMARY_DIRECTION_ONLY"
        else:
            assessment = "BOTH_PRIMARY_DIRECTIONS_OBSERVED"

        rows.append({
            "dtc_a": dtc_a,
            "dtc_b": dtc_b,
            "cooccurrence_cases": total,
            "dtc_a_primary_cases": a_count,
            "dtc_b_primary_cases": b_count,
            "unique_template_count": len(
                item["template_ids"]
            ),
            "template_ids": "|".join(
                sorted(item["template_ids"])
            ),
            "context_conditions": "|".join(
                sorted(item["context_conditions"])
            ),
            "pair_coverage_assessment": assessment,
        })

    return pd.DataFrame(rows)


# ============================================================
# TEMPLATE COVERAGE AND PERFORMANCE
# ============================================================

def build_template_coverage(cases, comparison):

    # Select only required comparison columns.
    # This avoids duplicate candidate_dtcs and primary_dtc columns.
    comparison_fields = [
        "case_id",
        "static_top1_correct",
        "context_top1_correct",
    ]

    merged = cases.merge(
        comparison[comparison_fields],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    rows = []

    for template_id, group in merged.groupby(
        "template_id",
        sort=True,
    ):

        candidate_sets = sorted(
            set(
                "|".join(parse_candidates(value))
                for value in group["candidate_dtcs"]
            )
        )

        primary_labels = sorted(
            group["primary_dtc"]
            .astype(str)
            .str.upper()
            .unique()
        )

        context_conditions = sorted(
            group["context_condition"]
            .astype(str)
            .unique()
        )

        static_correct = pd.to_numeric(
            group["static_top1_correct"],
            errors="raise",
        )

        context_correct = pd.to_numeric(
            group["context_top1_correct"],
            errors="raise",
        )

        rows.append({
            "template_id": template_id,
            "case_count": len(group),
            "repetition_values": "|".join(
                sorted(
                    group["repetition"].astype(str).unique()
                )
            ),
            "candidate_sets": "|".join(candidate_sets),
            "primary_dtc_labels": "|".join(primary_labels),
            "context_conditions": "|".join(
                context_conditions
            ),
            "static_top1_correct_cases": int(
                static_correct.sum()
            ),
            "context_top1_correct_cases": int(
                context_correct.sum()
            ),
            "static_top1_accuracy": float(
                static_correct.mean()
            ),
            "context_top1_accuracy": float(
                context_correct.mean()
            ),
            "template_repetition_count": len(group),
            "template_status": (
                "EXPECTED_TWO_REPETITIONS"
                if len(group) == EXPECTED_REPETITIONS
                else "REPETITION_COUNT_DIFFERENT"
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# REPORT
# ============================================================

def write_report(
    cases,
    scenarios,
    comparison,
    case_checks,
    dtc_coverage,
    pair_coverage,
    template_coverage,
):

    total_pairs = len(pair_coverage)

    observed_pairs = int(
        (pair_coverage["cooccurrence_cases"] > 0).sum()
    )

    both_direction_pairs = int(
        (
            pair_coverage["pair_coverage_assessment"]
            == "BOTH_PRIMARY_DIRECTIONS_OBSERVED"
        ).sum()
    )

    one_direction_pairs = int(
        (
            pair_coverage["pair_coverage_assessment"]
            == "ONE_PRIMARY_DIRECTION_ONLY"
        ).sum()
    )

    no_pairs = int(
        (
            pair_coverage["pair_coverage_assessment"]
            == "NO_CO_OCCURRENCE"
        ).sum()
    )

    passed_cases = int(
        (case_checks["case_integrity_status"] == "PASS").sum()
    )

    static_accuracy = float(
        pd.to_numeric(
            comparison["static_top1_correct"],
            errors="raise",
        ).mean()
    )

    context_accuracy = float(
        pd.to_numeric(
            comparison["context_top1_correct"],
            errors="raise",
        ).mean()
    )

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "EXPERIMENT 07 - V4 COVERAGE AND VALIDITY ANALYSIS\n"
        )

        report.write("=" * 78 + "\n\n")

        write_section(report, "1. BENCHMARK STRUCTURE")

        report.write(f"Diagnostic cases: {len(cases)}\n")
        report.write(
            f"Scenario manifest rows: {len(scenarios)}\n"
        )
        report.write(
            f"Unique scenario templates: "
            f"{cases['template_id'].nunique()}\n"
        )
        report.write(
            f"Unique context conditions: "
            f"{cases['context_condition'].nunique()}\n"
        )
        report.write(
            f"Expected repetitions per template: "
            f"{EXPECTED_REPETITIONS}\n"
        )
        report.write(
            f"Case integrity checks passed: "
            f"{passed_cases}/{len(case_checks)}\n"
        )

        write_section(report, "2. DTC COVERAGE")

        report.write(
            dtc_coverage.to_string(index=False)
        )

        write_section(report, "3. PAIRWISE CANDIDATE COVERAGE")

        report.write(
            f"Possible DTC pairs: {total_pairs}\n"
        )
        report.write(
            f"Observed pairs: {observed_pairs}\n"
        )
        report.write(
            f"Pairs tested in both primary directions: "
            f"{both_direction_pairs}\n"
        )
        report.write(
            f"Pairs tested in one primary direction only: "
            f"{one_direction_pairs}\n"
        )
        report.write(
            f"Pairs with no co-occurrence: {no_pairs}\n\n"
        )

        report.write(
            pair_coverage[
                pair_coverage["cooccurrence_cases"] > 0
            ].to_string(index=False)
        )

        write_section(report, "4. TEMPLATE COVERAGE")

        report.write(
            template_coverage.to_string(index=False)
        )

        write_section(report, "5. RANKING PERFORMANCE BY TEMPLATE")

        report.write(
            f"Static overall Top-1 accuracy: "
            f"{static_accuracy:.4f}\n"
        )
        report.write(
            f"Context overall Top-1 accuracy: "
            f"{context_accuracy:.4f}\n"
        )

        write_section(report, "6. VALIDITY LIMITATIONS")

        report.write(
            "1. V4 is a controlled synthetic benchmark, not data "
            "recorded from physical vehicles.\n"
        )

        report.write(
            "2. The 24 cases consist of 12 scenario templates "
            "repeated twice with different signal noise.\n"
        )

        report.write(
            "3. Repetitions within a template do not constitute "
            "independent scenario families.\n"
        )

        report.write(
            "4. Each case contains two candidates. The analysis "
            "does not establish performance with larger candidate sets.\n"
        )

        report.write(
            "5. The candidate list is assumed to be supplied by "
            "an upstream diagnostic layer. DTC detection is not evaluated.\n"
        )

        report.write(
            "6. Pairwise coverage is limited to the candidate "
            "combinations represented in V4.\n"
        )

        report.write(
            "7. P0335 uses an indirect RPM-instability proxy; "
            "a direct crankshaft-position waveform is not represented.\n"
        )

        report.write(
            "8. Ranking results are benchmark-specific and do not "
            "establish real-vehicle diagnostic performance.\n"
        )

        report.write(
            "9. No dataset, threshold, context weight or ranking "
            "rule was changed in this analysis.\n"
        )

        write_section(report, "7. OUTPUT FILES")

        for path in [
            OUTPUT_DTC,
            OUTPUT_PAIRS,
            OUTPUT_TEMPLATES,
            OUTPUT_CASES,
            OUTPUT_REPORT,
            OUTPUT_RUN,
        ]:
            report.write(str(path) + "\n")

        report.write("\nSTATUS: PASS\n")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("EXPERIMENT 07 - V4 COVERAGE AND VALIDITY ANALYSIS")
    print("=" * 78)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases, scenarios, comparison = load_inputs()

    print(f"\nCases loaded: {len(cases)}")
    print(
        f"Scenario templates: {cases['template_id'].nunique()}"
    )

    case_checks = build_case_checks(
        cases,
        scenarios,
        comparison,
    )

    dtc_coverage = build_dtc_coverage(
        cases
    )

    pair_coverage = build_pair_coverage(
        cases
    )

    template_coverage = build_template_coverage(
        cases,
        comparison,
    )

    # Structural validation.
    if len(template_coverage) != EXPECTED_TEMPLATES:
        raise ValueError(
            f"Expected {EXPECTED_TEMPLATES} templates; "
            f"found {len(template_coverage)}."
        )

    if not (
        case_checks["case_integrity_status"] == "PASS"
    ).all():

        failed = case_checks[
            case_checks["case_integrity_status"] != "PASS"
        ]

        raise ValueError(
            "Case integrity checks failed:\n"
            + failed.to_string(index=False)
        )

    if not (
        template_coverage["template_repetition_count"]
        == EXPECTED_REPETITIONS
    ).all():

        raise ValueError(
            "One or more templates do not have the expected "
            "number of repetitions."
        )

    # Save outputs.
    dtc_coverage.to_csv(
        OUTPUT_DTC,
        index=False,
    )

    pair_coverage.to_csv(
        OUTPUT_PAIRS,
        index=False,
    )

    template_coverage.to_csv(
        OUTPUT_TEMPLATES,
        index=False,
    )

    case_checks.to_csv(
        OUTPUT_CASES,
        index=False,
    )

    write_report(
        cases,
        scenarios,
        comparison,
        case_checks,
        dtc_coverage,
        pair_coverage,
        template_coverage,
    )

    print("\nDTC COVERAGE")
    print("-" * 78)

    print(
        dtc_coverage[
            [
                "dtc",
                "primary_case_count",
                "competing_case_count",
                "primary_template_count",
                "competing_template_count",
                "coverage_assessment",
            ]
        ].to_string(index=False)
    )

    print("\nPAIRWISE COVERAGE")
    print("-" * 78)

    print(
        pair_coverage[
            pair_coverage["cooccurrence_cases"] > 0
        ][
            [
                "dtc_a",
                "dtc_b",
                "cooccurrence_cases",
                "dtc_a_primary_cases",
                "dtc_b_primary_cases",
                "pair_coverage_assessment",
            ]
        ].to_string(index=False)
    )

    print("\nTEMPLATE COVERAGE")
    print("-" * 78)

    print(
        template_coverage[
            [
                "template_id",
                "case_count",
                "primary_dtc_labels",
                "context_conditions",
                "static_top1_accuracy",
                "context_top1_accuracy",
            ]
        ].to_string(index=False)
    )

    print("\nCASE INTEGRITY")
    print("-" * 78)

    print(
        case_checks[
            "case_integrity_status"
        ].value_counts().to_string()
    )

    print("\nOUTPUT FILES")
    print("-" * 78)

    for path in [
        OUTPUT_DTC,
        OUTPUT_PAIRS,
        OUTPUT_TEMPLATES,
        OUTPUT_CASES,
        OUTPUT_REPORT,
    ]:
        print(path)

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
