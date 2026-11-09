"""
Validate the V1 automotive DTC/CAN benchmark.

Actual V1 CAN schema:
    session_id, timestamp_s, CAN_ID, DATA0, DATA1, ..., DATA7, frame_type

Usage:
    python experiments\01_validate_dataset.py

The validator reads data/raw only and writes a report to results/.
It does not modify the raw dataset.
"""

from pathlib import Path
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CAN_FILE = RAW / "can_diagnostic_dataset.csv"
EVENT_FILE = RAW / "diagnostic_events.csv"

EXPECTED_DIAG_IDS = {0x7E8, 0x7E9}

DTC_PATTERN = re.compile(
    r"^[PBCU][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]{2}$"
)


def main():

    issues = []
    warnings = []
    checks = {}

    # ---------------------------------------------------------
    # 1. Check files exist
    # ---------------------------------------------------------

    if not CAN_FILE.exists():
        raise FileNotFoundError(f"Missing {CAN_FILE}")

    if not EVENT_FILE.exists():
        raise FileNotFoundError(f"Missing {EVENT_FILE}")

    can = pd.read_csv(CAN_FILE)
    events = pd.read_csv(EVENT_FILE)

    checks["can_rows"] = int(len(can))
    checks["event_rows"] = int(len(events))

    checks["can_columns"] = list(can.columns)
    checks["event_columns"] = list(events.columns)

    # ---------------------------------------------------------
    # 2. Check schema
    # ---------------------------------------------------------

    required_can = {
        "session_id",
        "timestamp_s",
        "CAN_ID",
        "frame_type",
    }

    data_columns = [f"DATA{i}" for i in range(8)]

    required_can.update(data_columns)

    required_event = {
        "session_id",
        "dtc",
        "ecu",
        "event_start_s",
        "event_end_s",
    }

    missing_can = sorted(
        required_can - set(can.columns)
    )

    missing_event = sorted(
        required_event - set(events.columns)
    )

    if missing_can:
        issues.append(
            f"CAN file missing columns: {missing_can}"
        )

    if missing_event:
        issues.append(
            f"Event file missing columns: {missing_event}"
        )

    if issues:

        report = {
            "status": "FAILED",
            "dataset": "automotive-dtc-benchmark-v1",
            "checks": checks,
            "issues": issues,
            "warnings": warnings,
        }

        output_file = RESULTS / "dataset_validation_report.json"

        output_file.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print(json.dumps(report, indent=2))

        raise SystemExit(1)

    # ---------------------------------------------------------
    # 3. Check null values
    # ---------------------------------------------------------

    null_can = {
        column: int(count)
        for column, count in can.isna().sum().items()
        if count
    }

    null_events = {
        column: int(count)
        for column, count in events.isna().sum().items()
        if count
    }

    checks["null_counts_can"] = null_can
    checks["null_counts_events"] = null_events

    if null_can:
        issues.append(
            f"CAN dataset contains null values: {null_can}"
        )

    if null_events:
        issues.append(
            f"Diagnostic event dataset contains null values: {null_events}"
        )

    # ---------------------------------------------------------
    # 4. Convert timestamp and CAN ID
    # ---------------------------------------------------------

    can["timestamp_s_num"] = pd.to_numeric(
        can["timestamp_s"],
        errors="coerce",
    )

    can["CAN_ID_num"] = pd.to_numeric(
        can["CAN_ID"],
        errors="coerce",
    )

    if can["timestamp_s_num"].isna().any():

        issues.append(
            "Non-numeric CAN timestamps found."
        )

    if can["CAN_ID_num"].isna().any():

        issues.append(
            "Non-numeric CAN IDs found."
        )

    # ---------------------------------------------------------
    # 5. Validate CAN data bytes
    # ---------------------------------------------------------

    data_range_violations = {}

    for column in data_columns:

        values = pd.to_numeric(
            can[column],
            errors="coerce",
        )

        if values.isna().any():

            issues.append(
                f"Non-numeric values found in {column}."
            )

            continue

        invalid = (
            (values < 0)
            | (values > 255)
        )

        if invalid.any():

            data_range_violations[column] = int(
                invalid.sum()
            )

    checks["data_byte_range_violations"] = (
        data_range_violations
    )

    if data_range_violations:

        issues.append(
            "CAN data bytes outside 0-255: "
            f"{data_range_violations}"
        )

    # ---------------------------------------------------------
    # 6. Validate CAN IDs
    # ---------------------------------------------------------

    invalid_can_ids = (
        (can["CAN_ID_num"] < 0)
        | (can["CAN_ID_num"] > 0x7FF)
    )

    checks["invalid_can_id_count"] = int(
        invalid_can_ids.sum()
    )

    if invalid_can_ids.any():

        issues.append(
            f"{int(invalid_can_ids.sum())} CAN IDs "
            "are outside the standard 11-bit range."
        )

    checks["unique_can_ids"] = int(
        can["CAN_ID_num"].nunique()
    )

    checks["can_id_counts"] = {
        f"0x{int(can_id):03X}": int(count)
        for can_id, count
        in can["CAN_ID_num"]
        .value_counts()
        .sort_index()
        .items()
    }

    # ---------------------------------------------------------
    # 7. Frame type
    # ---------------------------------------------------------

    frame_types = sorted(
        can["frame_type"]
        .astype(str)
        .unique()
    )

    checks["frame_types"] = frame_types

    if not frame_types:

        issues.append(
            "No frame_type values found."
        )

    # ---------------------------------------------------------
    # 8. Session consistency
    # ---------------------------------------------------------

    checks["unique_sessions"] = int(
        can["session_id"].nunique()
    )

    checks["event_sessions"] = int(
        events["session_id"].nunique()
    )

    if set(can["session_id"]) != set(
        events["session_id"]
    ):

        issues.append(
            "Session IDs differ between CAN data "
            "and diagnostic events."
        )

    # ---------------------------------------------------------
    # 9. Timestamp integrity per session
    # ---------------------------------------------------------

    session_timing = {}

    for session_id, group in can.groupby(
        "session_id"
    ):

        timestamps = (
            group["timestamp_s_num"]
            .sort_values()
        )

        differences = (
            timestamps.diff()
            .dropna()
        )

        session_timing[str(session_id)] = {

            "rows": int(len(group)),

            "start_s": float(
                timestamps.min()
            ),

            "end_s": float(
                timestamps.max()
            ),

            "duration_s": float(
                timestamps.max()
                - timestamps.min()
            ),

            "non_monotonic_count": int(
                (differences < 0).sum()
            ),

            "duplicate_timestamp_count": int(
                (differences == 0).sum()
            ),

            "min_delta_s": (
                float(differences.min())
                if len(differences)
                else None
            ),

            "median_delta_s": (
                float(differences.median())
                if len(differences)
                else None
            ),

            "max_delta_s": (
                float(differences.max())
                if len(differences)
                else None
            ),
        }

        if (differences < 0).any():

            issues.append(
                f"Session {session_id} "
                "has decreasing timestamps."
            )

    checks["session_timing"] = session_timing

    if can["session_id"].nunique() != 48:

        warnings.append(
            "V3 specification expected 48 sessions; "
            f"found {can['session_id'].nunique()}."
        )

    # ---------------------------------------------------------
    # 10. Diagnostic event timing
    # ---------------------------------------------------------

    events["event_start_s_num"] = pd.to_numeric(
        events["event_start_s"],
        errors="coerce",
    )

    events["event_end_s_num"] = pd.to_numeric(
        events["event_end_s"],
        errors="coerce",
    )

    if events[
        ["event_start_s_num", "event_end_s_num"]
    ].isna().any().any():

        issues.append(
            "Non-numeric diagnostic event "
            "timing values found."
        )

    invalid_event_order = (
        events["event_end_s_num"]
        <= events["event_start_s_num"]
    )

    if invalid_event_order.any():

        issues.append(
            f"{int(invalid_event_order.sum())} "
            "events have end <= start."
        )

    # ---------------------------------------------------------
    # 11. Validate DTC strings
    # ---------------------------------------------------------

    invalid_dtcs = [

        str(dtc)

        for dtc in events["dtc"]

        if not DTC_PATTERN.match(str(dtc))
    ]

    checks["unique_dtcs"] = sorted(
        events["dtc"]
        .astype(str)
        .unique()
    )

    checks["unique_ecus"] = sorted(
        events["ecu"]
        .astype(str)
        .unique()
    )

    checks["invalid_dtc_strings"] = invalid_dtcs

    if invalid_dtcs:

        warnings.append(
            "Some DTC strings do not match the "
            "generic P/B/C/U + 4-character format."
        )

    checks["events_per_session"] = {
        str(session_id): int(count)

        for session_id, count

        in events
        .groupby("session_id")
        .size()
        .items()
    }

    # ---------------------------------------------------------
    # 12. Diagnostic frame coverage
    # ---------------------------------------------------------

    diagnostic_frames = can[
        can["CAN_ID_num"].isin(
            EXPECTED_DIAG_IDS
        )
    ].copy()

    checks["diagnostic_frame_rows"] = int(
        len(diagnostic_frames)
    )

    if diagnostic_frames.empty:

        issues.append(
            "No frames found for expected "
            "diagnostic response IDs 0x7E8 / 0x7E9."
        )

    else:

        coverage = []

        for _, event in events.iterrows():

            matching_frames = diagnostic_frames[
                (
                    diagnostic_frames["session_id"]
                    == event["session_id"]
                )
                &
                (
                    diagnostic_frames["timestamp_s_num"]
                    >= event["event_start_s_num"]
                )
                &
                (
                    diagnostic_frames["timestamp_s_num"]
                    <= event["event_end_s_num"]
                )
            ]

            coverage.append({

                "session_id": event["session_id"],

                "dtc": event["dtc"],

                "ecu": event["ecu"],

                "diagnostic_frames_in_event_interval":
                    int(len(matching_frames)),
            })

        coverage_df = pd.DataFrame(
            coverage
        )

        checks[
            "event_diagnostic_frame_coverage"
        ] = coverage_df.to_dict(
            orient="records"
        )

        zero_coverage = coverage_df[
            coverage_df[
                "diagnostic_frames_in_event_interval"
            ] == 0
        ]

        if not zero_coverage.empty:

            issues.append(
                f"{len(zero_coverage)} "
                "diagnostic events have no "
                "diagnostic CAN frames within "
                "their event interval."
            )

    # ---------------------------------------------------------
    # 13. Check events against session boundaries
    # ---------------------------------------------------------

    events_outside_bounds = []

    for _, event in events.iterrows():

        session_id = event["session_id"]

        if session_id not in session_timing:
            continue

        session_start = session_timing[
            session_id
        ]["start_s"]

        session_end = session_timing[
            session_id
        ]["end_s"]

        event_start = event[
            "event_start_s_num"
        ]

        event_end = event[
            "event_end_s_num"
        ]

        if (
            event_start < session_start
            or event_end > session_end
        ):

            events_outside_bounds.append({

                "session_id": session_id,

                "dtc": event["dtc"],

                "event_start_s": float(
                    event_start
                ),

                "event_end_s": float(
                    event_end
                ),

                "session_start_s": session_start,

                "session_end_s": session_end,
            })

    checks[
        "events_outside_session_bounds"
    ] = events_outside_bounds

    if events_outside_bounds:

        issues.append(
            f"{len(events_outside_bounds)} "
            "diagnostic events extend outside "
            "their CAN session bounds."
        )

    # ---------------------------------------------------------
    # 14. Leakage check
    # ---------------------------------------------------------

    leakage_tokens = [

        "dtc_label",
        "ground_truth",
        "groundtruth",
        "target",
        "priority",
        "rank",
        "fault_type",
        "scenario",
    ]

    potential_leakage = [

        column

        for column in can.columns

        if any(
            token in column.lower()
            for token in leakage_tokens
        )
    ]

    checks[
        "potential_leakage_columns"
    ] = potential_leakage

    if potential_leakage:

        warnings.append(
            "Potential label/scenario leakage "
            "columns found: "
            + ", ".join(potential_leakage)
        )

    # ---------------------------------------------------------
    # 15. Check for DTC columns in CAN data
    # ---------------------------------------------------------

    dtc_columns = [

        column

        for column in can.columns

        if "dtc" in column.lower()
    ]

    checks[
        "dtc_columns_in_can_file"
    ] = dtc_columns

    if dtc_columns:

        warnings.append(
            "DTC-related columns exist directly "
            "in the CAN dataset. These must not "
            "be used as CAN-context features."
        )

    # ---------------------------------------------------------
    # 16. Final summary
    # ---------------------------------------------------------

    checks[
        "expected_diagnostic_ids_present"
    ] = sorted(

        f"0x{can_id:03X}"

        for can_id in
        EXPECTED_DIAG_IDS.intersection(
            set(
                can["CAN_ID_num"]
                .astype(int)
            )
        )
    )

    checks["context_frame_count"] = int(

        len(
            can[
                ~can["CAN_ID_num"].isin(
                    EXPECTED_DIAG_IDS
                )
            ]
        )
    )

    checks["diagnostic_frame_count"] = int(
        len(diagnostic_frames)
    )

    # ---------------------------------------------------------
    # 17. Create final report
    # ---------------------------------------------------------

    status = (
        "PASS"
        if not issues
        else "FAILED"
    )

    report = {

        "status": status,

        "dataset":
            "automotive-dtc-benchmark-v1",

        "checks":
            checks,

        "issues":
            issues,

        "warnings":
            warnings,
    }

    output_file = (
        RESULTS
        / "dataset_validation_report.json"
    )

    output_file.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2
        )
    )

    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()