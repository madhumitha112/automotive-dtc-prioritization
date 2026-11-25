"""
Experiment 03E1
---------------
Evidence-informed, study-defined static DTC priority classes.

This is a separate experimental variant. It does not overwrite the frozen
V3/V4 baseline scripts or their outputs. The priority configuration is read
from experiments/static_priority_evidence_v1.json.

Ranking inputs: candidate DTCs + fixed configuration only.
Evaluation-only field: primary_dtc, used after ranking.
"""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "experiments" / "static_priority_evidence_v1.json"
BENCHMARKS = {
    "v3": (ROOT / "results" / "diagnostic_cases.csv", 48),
    "v4": (ROOT / "results" / "diagnostic_cases_v4.csv", 24),
}
OUTPUT_DIR = ROOT / "results"

with CONFIG_FILE.open(encoding="utf-8") as handle:
    config = json.load(handle)

priority_rows = config.get("dtcs", [])
priority = {row["dtc"]: float(row["score"]) for row in priority_rows}
expected_dtcs = {"P0335", "P0300", "C0035", "P0121", "P0130", "P0171", "P0172", "P0128"}

if len(priority) != len(priority_rows) or set(priority) != expected_dtcs:
    raise ValueError("Priority configuration must contain exactly one row for each expected DTC.")
if any(not 0 <= score <= 1 for score in priority.values()):
    raise ValueError("Priority scores must be in [0, 1].")

def parse_candidates(value):
    if pd.isna(value):
        return []
    return [token.strip().upper() for token in str(value).split("|") if token.strip()]

def evaluate(dataset_name, input_file, expected_cases):
    cases = pd.read_csv(input_file)
    required = {"case_id", "candidate_dtcs", "candidate_count", "primary_dtc"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"{dataset_name}: missing columns {missing}")
    if len(cases) != expected_cases:
        raise ValueError(f"{dataset_name}: expected {expected_cases} cases, got {len(cases)}")
    if cases["case_id"].duplicated().any():
        raise ValueError(f"{dataset_name}: duplicate case IDs")

    rows = []
    for _, case in cases.iterrows():
        case_id = str(case["case_id"])
        candidates = parse_candidates(case["candidate_dtcs"])
        if len(candidates) < 2:
            raise ValueError(f"{dataset_name}/{case_id}: fewer than two candidates")
        if len(candidates) != len(set(candidates)):
            raise ValueError(f"{dataset_name}/{case_id}: duplicate candidates")
        if int(case["candidate_count"]) != len(candidates):
            raise ValueError(f"{dataset_name}/{case_id}: candidate_count mismatch")
        unknown = sorted(set(candidates) - set(priority))
        if unknown:
            raise ValueError(f"{dataset_name}/{case_id}: unknown DTCs {unknown}")

        ranked = sorted(candidates, key=lambda dtc: (-priority[dtc], dtc))
        target = str(case["primary_dtc"]).strip().upper()
        if target not in candidates:
            raise ValueError(f"{dataset_name}/{case_id}: primary DTC is not a candidate")

        # Evaluation label is read only after the ranking has been produced.
        rank = ranked.index(target) + 1
        rows.append({
            "case_id": case_id,
            "candidate_dtcs": "|".join(candidates),
            "candidate_count": len(candidates),
            "priority_ranking_evidence_v1": "|".join(ranked),
            "top_1_dtc_evidence_v1": ranked[0],
            "primary_dtc": target,
            "primary_rank_evidence_v1": rank,
            "reciprocal_rank_evidence_v1": 1.0 / rank,
            "top_1_correct_evidence_v1": int(ranked[0] == target),
            "priority_score_top_1_evidence_v1": priority[ranked[0]],
        })

    frame = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    if len(frame) != expected_cases or frame["case_id"].duplicated().any():
        raise ValueError(f"{dataset_name}: result integrity check failed")
    if not frame["primary_rank_evidence_v1"].between(1, frame["candidate_count"]).all():
        raise ValueError(f"{dataset_name}: invalid target rank")

    metrics = {
        "benchmark": dataset_name,
        "cases": int(len(frame)),
        "top_1_accuracy": float(frame["top_1_correct_evidence_v1"].mean()),
        "mrr": float(frame["reciprocal_rank_evidence_v1"].mean()),
        "mean_primary_rank": float(frame["primary_rank_evidence_v1"].mean()),
        "top_3_accuracy": float(
            (frame["primary_rank_evidence_v1"] <= 3).mean()
        ),
        "priority_config_version": config["version"],
        "interpretation": "study-defined benchmark result; not real-vehicle validation",
    }
    output = OUTPUT_DIR / f"static_baseline_evidence_v1_{dataset_name}.csv"
    frame.to_csv(output, index=False)
    print(f"\n{dataset_name.upper()} metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"  output: {output}")
    return metrics

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [evaluate(name, path, count) for name, (path, count) in BENCHMARKS.items()]
    summary_file = OUTPUT_DIR / "static_baseline_evidence_v1_metrics.json"
    summary_file.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_file}")
    print("STATUS: PASS")
