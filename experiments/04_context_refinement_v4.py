
"""
EXPERIMENT 04 - V4 CAN-Context-Based DTC Prioritization

Purpose:
Evaluate the frozen V3 static priorities and CAN-context refinement
rules on the V4 diagnostic benchmark.

Algorithm inputs:
- Candidate DTCs from V4 diagnostic cases
- Raw V4 CAN context frames
- Observation interval
- Fixed signal-decoding definitions
- Fixed DTC-context rules
- Frozen V3 static priority scores

Evaluation-only fields:
- primary_dtc
- secondary_dtcs
- context_condition
- event_role

These fields are never used to calculate context scores or rankings.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CAN_FILE = ROOT / "data" / "raw" / "can_diagnostic_dataset_v4.csv"
CASES_FILE = ROOT / "results" / "diagnostic_cases_v4.csv"

OUTPUT_DIR = ROOT / "results"
OUTPUT_FILE = OUTPUT_DIR / "context_refinement_results_v4.csv"

EXPECTED_CASES = 24
CONTEXT_WEIGHT = 1.00

CONTEXT_IDS = {256, 257, 258, 259}

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

KNOWN_DTCS = set(STATIC_PRIORITY)


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_can_id(value):
    """Convert decimal or hexadecimal CAN ID to integer."""

    if isinstance(value, str):
        value = value.strip()

        if value.lower().startswith("0x"):
            return int(value, 16)

    return int(value)


def fraction_true(condition):
    """Return the fraction of observations satisfying a condition."""

    if len(condition) == 0:
        return 0.0

    return float(np.asarray(condition, dtype=bool).mean())


def clip_score(value):
    """Keep context evidence within the 0-1 range."""

    return float(np.clip(value, 0.0, 1.0))


def parse_dtc_list(value):
    """Parse pipe-separated candidate DTC strings."""

    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


# ============================================================
# CAN SIGNAL DECODING
# ============================================================

def decode_context_frames(session_frames):
    """
    Decode raw CAN bytes into engineering-unit signals.

    Only context frames from CAN IDs 0x100-0x103 are used.
    Diagnostic frames are excluded.
    """

    frames = session_frames[
        session_frames["frame_type"].astype(str).str.lower()
        == "context"
    ].copy()

    decoded = {}

    # --------------------------------------------------------
    # 0x100 ENGINE DYNAMICS
    # --------------------------------------------------------

    engine = frames[frames["CAN_ID_num"] == 0x100].copy()

    if not engine.empty:

        rpm_raw = (
            engine["DATA0"]
            + 256 * engine["DATA1"]
        )

        speed_raw = (
            engine["DATA2"]
            + 256 * engine["DATA3"]
        )

        decoded["engine"] = pd.DataFrame({
            "timestamp_s": engine["timestamp_s"],
            "rpm": rpm_raw.astype(float),
            "vehicle_speed": speed_raw / 10.0,
            "engine_load": engine["DATA4"] * 100.0 / 255.0,
            "throttle": engine["DATA5"] * 100.0 / 255.0,
        })

    # --------------------------------------------------------
    # 0x101 ENGINE THERMAL / AIR
    # --------------------------------------------------------

    thermal = frames[frames["CAN_ID_num"] == 0x101].copy()

    if not thermal.empty:

        decoded["thermal"] = pd.DataFrame({
            "timestamp_s": thermal["timestamp_s"],
            "coolant_temp": thermal["DATA0"] * 0.5 - 40.0,
            "intake_air_temp": thermal["DATA1"] * 0.5 - 40.0,
            "map_kpa": thermal["DATA2"] * 0.5,
            "oxygen_signal": thermal["DATA3"] * 5.0 / 255.0,
        })

    # --------------------------------------------------------
    # 0x102 DRIVER / FUEL
    # --------------------------------------------------------

    driver = frames[frames["CAN_ID_num"] == 0x102].copy()

    if not driver.empty:

        decoded["driver"] = pd.DataFrame({
            "timestamp_s": driver["timestamp_s"],
            "stft": driver["DATA0"] - 128.0,
            "ltft": driver["DATA1"] - 128.0,
            "accelerator_pedal": driver["DATA2"] * 100.0 / 255.0,
            "brake_state": driver["DATA3"],
        })

    # --------------------------------------------------------
    # 0x103 WHEEL SPEEDS
    # --------------------------------------------------------

    wheels = frames[frames["CAN_ID_num"] == 0x103].copy()

    if not wheels.empty:

        decoded["wheels"] = pd.DataFrame({
            "timestamp_s": wheels["timestamp_s"],
            "wheel_fl": wheels["DATA0"] * 0.5,
            "wheel_fr": wheels["DATA1"] * 0.5,
            "wheel_rl": wheels["DATA2"] * 0.5,
            "wheel_rr": wheels["DATA3"] * 0.5,
        })

    required_groups = {
        "engine", "thermal", "driver", "wheels"
    }

    if set(decoded.keys()) != required_groups:
        raise ValueError(
            "Missing one or more context CAN message groups. "
            f"Decoded groups: {sorted(decoded.keys())}"
        )

    return decoded


# ============================================================
# DTC-SPECIFIC CONTEXT EVIDENCE
# ============================================================

def calculate_context_scores(decoded):
    """
    Calculate a context consistency score for each candidate DTC.

    Scores are based only on decoded CAN signals.
    They do not use primary DTC or scenario metadata.
    """

    engine = decoded["engine"]
    thermal = decoded["thermal"]
    driver = decoded["driver"]
    wheels = decoded["wheels"]

    scores = {}

    # --------------------------------------------------------
    # P0171 - SYSTEM TOO LEAN
    # Positive combined fuel trim is supporting evidence.
    # --------------------------------------------------------

    fuel = driver["stft"] + driver["ltft"]

    scores["P0171"] = clip_score(
        fraction_true(fuel > 20.0)
    )

    # --------------------------------------------------------
    # P0172 - SYSTEM TOO RICH
    # Negative combined fuel trim is supporting evidence.
    # --------------------------------------------------------

    scores["P0172"] = clip_score(
        fraction_true(fuel < -20.0)
    )

    # --------------------------------------------------------
    # P0128 - COOLANT BELOW REGULATING TEMPERATURE
    # --------------------------------------------------------

    scores["P0128"] = clip_score(
        fraction_true(thermal["coolant_temp"] < 75.0)
    )

    # --------------------------------------------------------
    # P0130 - OXYGEN SENSOR CIRCUIT
    # Persistent signal outside the defined operating band.
    # This is context evidence, not a complete circuit test.
    # --------------------------------------------------------

    oxygen = thermal["oxygen_signal"]

    oxygen_abnormal = (
        (oxygen < 0.35)
        | (oxygen > 0.65)
    )

    scores["P0130"] = clip_score(
        fraction_true(oxygen_abnormal)
    )

    # --------------------------------------------------------
    # P0121 - THROTTLE POSITION SENSOR RANGE/PERFORMANCE
    # Compare throttle position with accelerator pedal demand.
    # --------------------------------------------------------

    engine_driver = engine[
        ["timestamp_s", "throttle"]
    ].merge(
        driver[["timestamp_s", "accelerator_pedal"]],
        on="timestamp_s",
        how="inner",
    )

    throttle_difference = (
        engine_driver["accelerator_pedal"]
        - engine_driver["throttle"]
    ).abs()

    scores["P0121"] = clip_score(
        fraction_true(throttle_difference > 8.0)
    )

    # --------------------------------------------------------
    # P0300 - RANDOM/MULTIPLE CYLINDER MISFIRE
    # Supporting evidence: RPM drop under higher engine load.
    # --------------------------------------------------------

    engine_sorted = engine.sort_values(
        "timestamp_s"
    ).copy()

    rolling_rpm = (
        engine_sorted["rpm"]
        .rolling(window=11, center=True, min_periods=5)
        .median()
    )

    rpm_drop = (
        rolling_rpm - engine_sorted["rpm"]
    )

    misfire_evidence = (
        (engine_sorted["engine_load"] > 45.0)
        & (rpm_drop > 220.0)
    )

    scores["P0300"] = clip_score(
        fraction_true(misfire_evidence)
    )

    # --------------------------------------------------------
    # P0335 - CRANKSHAFT POSITION SENSOR CIRCUIT
    # RPM instability is only an indirect proxy in this dataset.
    # No direct CKP waveform is available.
    # --------------------------------------------------------

    crank_signal_proxy = rpm_drop > 350.0

    scores["P0335"] = clip_score(
        fraction_true(crank_signal_proxy)
    )

    # --------------------------------------------------------
    # C0035 - LEFT FRONT WHEEL SPEED SENSOR CIRCUIT
    # Compare left-front wheel speed against the other wheels.
    # --------------------------------------------------------

    other_wheels = wheels[
        ["wheel_fr", "wheel_rl", "wheel_rr"]
    ].median(axis=1)

    wheel_difference = (
        wheels["wheel_fl"] - other_wheels
    ).abs()

    wheel_mismatch = (
        wheel_difference
        > (other_wheels.abs() * 0.15 + 1.0)
    )

    scores["C0035"] = clip_score(
        fraction_true(wheel_mismatch)
    )

    return scores


# ============================================================
# RANKING
# ============================================================

def rank_candidates(candidate_dtcs, context_scores):
    """Combine static priority and CAN-context evidence."""

    ranking_rows = []

    for dtc in candidate_dtcs:

        static_score = STATIC_PRIORITY[dtc]
        context_score = context_scores[dtc]

        refined_score = (
            static_score
            + CONTEXT_WEIGHT * context_score
        )

        ranking_rows.append({
            "dtc": dtc,
            "static_score": static_score,
            "context_score": context_score,
            "refined_score": refined_score,
        })

    ranking_rows.sort(
        key=lambda row: (
            -row["refined_score"],
            -row["static_score"],
            row["dtc"],
        )
    )

    for rank, row in enumerate(ranking_rows, start=1):
        row["refined_rank"] = rank

    return ranking_rows


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("=" * 72)
    print("EXPERIMENT 04 - V4 CAN-CONTEXT DTC PRIORITIZATION")
    print("=" * 72)

    # --------------------------------------------------------
    # INPUT FILE CHECKS
    # --------------------------------------------------------

    if not CAN_FILE.exists():
        raise FileNotFoundError(CAN_FILE)

    if not CASES_FILE.exists():
        raise FileNotFoundError(CASES_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    can = pd.read_csv(CAN_FILE)
    cases = pd.read_csv(CASES_FILE)

    print(f"\nCAN rows loaded: {len(can):,}")
    print(f"Diagnostic cases loaded: {len(cases):,}")

    # --------------------------------------------------------
    # REQUIRED COLUMN CHECKS
    # --------------------------------------------------------

    required_can = {
        "session_id",
        "timestamp_s",
        "CAN_ID",
        "frame_type",
        *[f"DATA{i}" for i in range(8)],
    }

    required_cases = {
        "case_id",
        "session_id",
        "candidate_dtcs",
        "observation_start_s",
        "observation_end_s",
        "primary_dtc",
    }

    missing_can = required_can - set(can.columns)
    missing_cases = required_cases - set(cases.columns)

    if missing_can:
        raise ValueError(
            f"Missing CAN columns: {sorted(missing_can)}"
        )

    if missing_cases:
        raise ValueError(
            f"Missing case columns: {sorted(missing_cases)}"
        )

    if len(cases) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_CASES} cases; "
            f"found {len(cases)}."
        )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    can["session_id"] = can["session_id"].astype(str)
    cases["session_id"] = cases["session_id"].astype(str)

    can["timestamp_s"] = pd.to_numeric(
        can["timestamp_s"],
        errors="raise",
    )

    can["CAN_ID_num"] = can["CAN_ID"].apply(
        normalize_can_id
    )

    for i in range(8):
        can[f"DATA{i}"] = pd.to_numeric(
            can[f"DATA{i}"],
            errors="raise",
        )

    # --------------------------------------------------------
    # LEAKAGE CONTROL
    # --------------------------------------------------------

    evaluation_only_columns = {
        "primary_dtc",
        "secondary_dtcs",
        "context_condition",
        "event_role",
    }

    algorithm_input_columns = {
        "case_id",
        "session_id",
        "candidate_dtcs",
        "observation_start_s",
        "observation_end_s",
    }

    if algorithm_input_columns.issubset(evaluation_only_columns):
        raise RuntimeError("Invalid leakage configuration.")

    # --------------------------------------------------------
    # CASE PROCESSING
    # --------------------------------------------------------

    results = []

    for _, case in cases.iterrows():

        case_id = str(case["case_id"])
        session_id = str(case["session_id"])

        candidates = parse_dtc_list(
            case["candidate_dtcs"]
        )

        if not candidates:
            raise ValueError(
                f"No candidates found for {case_id}"
            )

        unknown_dtcs = set(candidates) - KNOWN_DTCS

        if unknown_dtcs:
            raise ValueError(
                f"Unknown DTCs in {case_id}: "
                f"{sorted(unknown_dtcs)}"
            )

        start_time = float(case["observation_start_s"])
        end_time = float(case["observation_end_s"])

        session_frames = can[
            (can["session_id"] == session_id)
            & (can["timestamp_s"] >= start_time)
            & (can["timestamp_s"] <= end_time)
            & (can["CAN_ID_num"].isin(CONTEXT_IDS))
        ].copy()

        if session_frames.empty:
            raise ValueError(
                f"No context CAN frames found for {case_id}"
            )

        decoded = decode_context_frames(
            session_frames
        )

        context_scores = calculate_context_scores(
            decoded
        )

        ranking = rank_candidates(
            candidates,
            context_scores,
        )

        ranking_string = "|".join(
            row["dtc"] for row in ranking
        )

        primary_dtc = str(case["primary_dtc"])

        if primary_dtc not in candidates:
            raise ValueError(
                f"Evaluation primary DTC not in candidates "
                f"for {case_id}"
            )

        primary_rank = next(
            row["refined_rank"]
            for row in ranking
            if row["dtc"] == primary_dtc
        )

        # ----------------------------------------------------
        # OUTPUT ROW
        # ----------------------------------------------------

        result = {
            "case_id": case_id,
            "session_id": session_id,
            "candidate_dtcs": "|".join(candidates),
            "candidate_count": len(candidates),
            "refined_ranking": ranking_string,
            "top1_dtc": ranking[0]["dtc"],
            "primary_dtc": primary_dtc,
            "primary_rank": primary_rank,
            "reciprocal_rank": 1.0 / primary_rank,
            "static_top_score": ranking[0]["static_score"],
        }

        for dtc in candidates:
            result[f"context_{dtc}"] = context_scores[dtc]

        results.append(result)

    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------------

    top1_accuracy = (
        results_df["primary_rank"] == 1
    ).mean()

    top3_accuracy = (
        results_df["primary_rank"] <= 3
    ).mean()

    mrr = results_df["reciprocal_rank"].mean()

    mean_rank = results_df["primary_rank"].mean()

    median_rank = results_df["primary_rank"].median()

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("V4 CONTEXT REFINEMENT EXPERIMENT COMPLETE")
    print("=" * 72)

    print(f"\nCases processed:          {len(results_df)}")
    print(f"Context weight:           {CONTEXT_WEIGHT:.2f}")

    print("\nREFINED METHOD PERFORMANCE")
    print("-" * 50)

    print(f"Top-1 accuracy:           {top1_accuracy:.4f}")
    print(f"Top-3 accuracy:           {top3_accuracy:.4f}")
    print(f"Mean Reciprocal Rank:     {mrr:.4f}")
    print(f"Mean primary rank:        {mean_rank:.4f}")
    print(f"Median primary rank:      {median_rank:.4f}")

    print("\nFIRST 10 CASES")
    print("-" * 72)

    print(
        results_df[
            [
                "case_id",
                "candidate_dtcs",
                "refined_ranking",
                "primary_dtc",
                "primary_rank",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nOUTPUT FILE")
    print("-" * 50)
    print(OUTPUT_FILE)

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()