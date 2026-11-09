"""
Experiment 02
-------------
Extract diagnostic prioritization cases from the frozen V3 benchmark.

Purpose:
    Create one diagnostic case per episode/session.

Algorithm-input information:
    - session_id
    - candidate DTCs reported by the diagnostic layer
    - candidate ECUs
    - observation interval
    - diagnostic frame count
    - context frame count

Evaluation-only information:
    - primary_dtc
    - secondary_dtcs
    - context_condition
    - event-role information

IMPORTANT:
    Evaluation-only fields must NOT be supplied to either the static
    baseline or the CAN-context prioritization method.
"""

from pathlib import Path
import ast
import json

import pandas as pd


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CAN_FILE = ROOT / "data" / "raw" / "can_diagnostic_dataset.csv"
EVENT_FILE = ROOT / "data" / "raw" / "diagnostic_events.csv"

OUTPUT_DIR = ROOT / "results"
OUTPUT_FILE = OUTPUT_DIR / "diagnostic_cases.csv"


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

EXPECTED_SESSIONS = 48

EXPECTED_DTC_COUNT = 8

CONTEXT_IDS = {
    "0x100",
    "0x101",
    "0x102",
    "0x103",
}

DIAGNOSTIC_IDS = {
    "0x7E8",
    "0x7E9",
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def normalize_can_id(value):
    """
    Normalize CAN ID values to uppercase hexadecimal form.

    Examples:
        0x100 -> 0x100
        100   -> 0x64
    """

    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value.startswith("0X"):
        return value

    try:
        return f"0x{int(value):X}"
    except ValueError:
        return value


def unique_preserve_order(values):
    """Return unique non-null values while preserving first occurrence."""

    result = []

    for value in values:
        if pd.isna(value):
            continue

        value = str(value)

        if value not in result:
            result.append(value)

    return result


def list_to_string(values):
    """
    Store list-like values in a deterministic CSV-friendly representation.
    """

    return "|".join(str(v) for v in values)


def assert_required_columns(df, required, name):
    """Check that all required columns exist."""

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

print("=" * 70)
print("EXPERIMENT 02 - DIAGNOSTIC CASE EXTRACTION")
print("=" * 70)

print("\nLoading files...")

if not CAN_FILE.exists():
    raise FileNotFoundError(
        f"CAN dataset not found:\n{CAN_FILE}"
    )

if not EVENT_FILE.exists():
    raise FileNotFoundError(
        f"Diagnostic event file not found:\n{EVENT_FILE}"
    )

can = pd.read_csv(CAN_FILE)
events = pd.read_csv(EVENT_FILE)

print(f"CAN rows loaded:    {len(can):,}")
print(f"Event rows loaded:  {len(events):,}")


# ---------------------------------------------------------------------
# VALIDATE INPUT STRUCTURE
# ---------------------------------------------------------------------

assert_required_columns(
    can,
    {
        "session_id",
        "timestamp_s",
        "CAN_ID",
        "frame_type",
    },
    "CAN dataset",
)

assert_required_columns(
    events,
    {
        "session_id",
        "dtc",
        "ecu",
        "event_start_s",
        "event_end_s",
        "event_role",
        "context_condition",
    },
    "Diagnostic event dataset",
)


# ---------------------------------------------------------------------
# BASIC NORMALIZATION
# ---------------------------------------------------------------------

can["CAN_ID"] = can["CAN_ID"].apply(normalize_can_id)

events["session_id"] = events["session_id"].astype(str)
events["dtc"] = events["dtc"].astype(str).str.upper()
events["ecu"] = events["ecu"].astype(str).str.upper()
events["event_role"] = events["event_role"].astype(str).str.lower()
events["context_condition"] = (
    events["context_condition"]
    .astype(str)
    .str.strip()
)

can["session_id"] = can["session_id"].astype(str)

events["event_start_s"] = pd.to_numeric(
    events["event_start_s"],
    errors="raise",
)

events["event_end_s"] = pd.to_numeric(
    events["event_end_s"],
    errors="raise",
)

can["timestamp_s"] = pd.to_numeric(
    can["timestamp_s"],
    errors="raise",
)


# ---------------------------------------------------------------------
# DATASET-LEVEL CHECKS
# ---------------------------------------------------------------------

sessions_can = sorted(can["session_id"].unique())
sessions_events = sorted(events["session_id"].unique())

if len(sessions_can) != EXPECTED_SESSIONS:
    raise ValueError(
        f"Expected {EXPECTED_SESSIONS} CAN sessions, "
        f"found {len(sessions_can)}."
    )

if len(sessions_events) != EXPECTED_SESSIONS:
    raise ValueError(
        f"Expected {EXPECTED_SESSIONS} event sessions, "
        f"found {len(sessions_events)}."
    )

if set(sessions_can) != set(sessions_events):
    raise ValueError(
        "CAN sessions and diagnostic-event sessions do not match."
    )

unique_dtcs = sorted(events["dtc"].unique())

if len(unique_dtcs) != EXPECTED_DTC_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_DTC_COUNT} unique DTCs, "
        f"found {len(unique_dtcs)}."
    )

if events["event_start_s"].gt(events["event_end_s"]).any():
    raise ValueError(
        "At least one diagnostic event has event_start_s > event_end_s."
    )


# ---------------------------------------------------------------------
# BUILD ONE CASE PER SESSION
# ---------------------------------------------------------------------

cases = []

for session_id in sessions_can:

    session_can = can[
        can["session_id"] == session_id
    ].copy()

    session_events = events[
        events["session_id"] == session_id
    ].copy()

    if session_events.empty:
        raise ValueError(
            f"No diagnostic events found for session {session_id}."
        )

    # -------------------------------------------------------------
    # Candidate DTCs
    # -------------------------------------------------------------
    #
    # These come from the diagnostic layer.
    #
    # They are legitimate inputs to the prioritization problem:
    #
    #     "Given multiple reported DTC candidates, which should
    #      be investigated first?"
    #
    # The primary DTC is NOT used to construct this list.
    # -------------------------------------------------------------

    candidate_dtcs = unique_preserve_order(
        session_events["dtc"].tolist()
    )

    candidate_ecus = unique_preserve_order(
        session_events["ecu"].tolist()
    )

    if len(candidate_dtcs) < 2:
        raise ValueError(
            f"Session {session_id} has fewer than two candidate DTCs."
        )

    # -------------------------------------------------------------
    # Diagnostic observation interval
    # -------------------------------------------------------------

    observation_start = float(
        session_events["event_start_s"].min()
    )

    observation_end = float(
        session_events["event_end_s"].max()
    )

    # -------------------------------------------------------------
    # Diagnostic frames inside the case interval
    # -------------------------------------------------------------

    diagnostic_mask = (
        (session_can["frame_type"].astype(str).str.lower() == "diagnostic")
        & (session_can["timestamp_s"] >= observation_start)
        & (session_can["timestamp_s"] <= observation_end)
    )

    diagnostic_frame_count = int(
        diagnostic_mask.sum()
    )

    # -------------------------------------------------------------
    # Context frames inside the same interval
    # -------------------------------------------------------------

    context_mask = (
        (session_can["frame_type"].astype(str).str.lower() == "context")
        & (session_can["timestamp_s"] >= observation_start)
        & (session_can["timestamp_s"] <= observation_end)
    )

    context_frame_count = int(
        context_mask.sum()
    )

    # -------------------------------------------------------------
    # Evaluation-only information
    # -------------------------------------------------------------
    #
    # These fields describe the synthetic scenario and are NOT
    # available to either ranking method.
    # -------------------------------------------------------------

    primary_rows = session_events[
        session_events["event_role"] == "primary"
    ]

    if len(primary_rows) != 1:
        raise ValueError(
            f"Session {session_id} must contain exactly one "
            f"primary event; found {len(primary_rows)}."
        )

    primary_dtc = str(
        primary_rows.iloc[0]["dtc"]
    )

    secondary_dtcs = unique_preserve_order(
        session_events.loc[
            session_events["event_role"] != "primary",
            "dtc"
        ].tolist()
    )

    context_conditions = unique_preserve_order(
        session_events["context_condition"].tolist()
    )

    # -------------------------------------------------------------
    # Leakage protection checks
    # -------------------------------------------------------------

    if primary_dtc not in candidate_dtcs:
        raise ValueError(
            f"Primary DTC {primary_dtc} is not in candidate set "
            f"for {session_id}."
        )

    if primary_dtc == "":
        raise ValueError(
            f"Empty primary DTC for {session_id}."
        )

    # -------------------------------------------------------------
    # Store case
    # -------------------------------------------------------------

    cases.append(
        {
            "case_id": session_id,
            "session_id": session_id,

            # Algorithm-input fields
            "candidate_dtcs": list_to_string(candidate_dtcs),
            "candidate_ecus": list_to_string(candidate_ecus),
            "candidate_count": len(candidate_dtcs),

            "observation_start_s": observation_start,
            "observation_end_s": observation_end,
            "observation_duration_s": (
                observation_end - observation_start
            ),

            "diagnostic_frame_count": diagnostic_frame_count,
            "context_frame_count": context_frame_count,

            # Evaluation-only fields
            "primary_dtc": primary_dtc,
            "secondary_dtcs": list_to_string(secondary_dtcs),
            "context_condition": list_to_string(
                context_conditions
            ),
        }
    )


# ---------------------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------------------

cases_df = pd.DataFrame(cases)

cases_df = cases_df.sort_values(
    by="case_id"
).reset_index(drop=True)


# ---------------------------------------------------------------------
# FINAL CASE-LEVEL VALIDATION
# ---------------------------------------------------------------------

if len(cases_df) != EXPECTED_SESSIONS:
    raise ValueError(
        f"Expected {EXPECTED_SESSIONS} cases, "
        f"created {len(cases_df)}."
    )

if cases_df["case_id"].duplicated().any():
    duplicates = cases_df.loc[
        cases_df["case_id"].duplicated(),
        "case_id"
    ].tolist()

    raise ValueError(
        f"Duplicate case IDs detected: {duplicates}"
    )

if cases_df["candidate_count"].lt(2).any():
    bad_cases = cases_df.loc[
        cases_df["candidate_count"] < 2,
        "case_id"
    ].tolist()

    raise ValueError(
        f"Cases with fewer than two candidates: {bad_cases}"
    )

if cases_df["primary_dtc"].isna().any():
    raise ValueError(
        "Missing primary DTC in one or more cases."
    )

if cases_df["candidate_dtcs"].isna().any():
    raise ValueError(
        "Missing candidate DTC set in one or more cases."
    )

if cases_df["context_frame_count"].le(0).any():
    bad_cases = cases_df.loc[
        cases_df["context_frame_count"] <= 0,
        "case_id"
    ].tolist()

    raise ValueError(
        f"Cases without context frames: {bad_cases}"
    )

if cases_df["diagnostic_frame_count"].le(0).any():
    bad_cases = cases_df.loc[
        cases_df["diagnostic_frame_count"] <= 0,
        "case_id"
    ].tolist()

    raise ValueError(
        f"Cases without diagnostic frames: {bad_cases}"
    )


# ---------------------------------------------------------------------
# VERIFY PRIMARY DTC IS NOT BEING STORED AS A CANDIDATE FEATURE
# ---------------------------------------------------------------------
#
# The primary DTC is retained only for evaluation.
#
# The ranking scripts must explicitly use:
#
#     candidate_dtcs
#     candidate_ecus
#     observation interval
#     CAN data
#
# and must NOT use:
#
#     primary_dtc
#     secondary_dtcs
#     context_condition
#
# This check documents the intended separation.
# ---------------------------------------------------------------------

algorithm_input_columns = {
    "case_id",
    "session_id",
    "candidate_dtcs",
    "candidate_ecus",
    "candidate_count",
    "observation_start_s",
    "observation_end_s",
    "observation_duration_s",
    "diagnostic_frame_count",
    "context_frame_count",
}

evaluation_only_columns = {
    "primary_dtc",
    "secondary_dtcs",
    "context_condition",
}

actual_columns = set(cases_df.columns)

expected_columns = (
    algorithm_input_columns
    | evaluation_only_columns
)

if actual_columns != expected_columns:
    missing = sorted(expected_columns - actual_columns)
    extra = sorted(actual_columns - expected_columns)

    raise ValueError(
        f"Unexpected case columns. Missing={missing}, Extra={extra}"
    )


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

cases_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 02 COMPLETE")
print("=" * 70)

print(f"Cases created:          {len(cases_df)}")
print(f"Unique candidate DTCs:  {len(unique_dtcs)}")
print(
    "Candidate count range: "
    f"{cases_df['candidate_count'].min()} - "
    f"{cases_df['candidate_count'].max()}"
)

print(
    "Diagnostic frames:     "
    f"{cases_df['diagnostic_frame_count'].sum():,}"
)

print(
    "Context frames:        "
    f"{cases_df['context_frame_count'].sum():,}"
)

print(f"\nOutput:")
print(OUTPUT_FILE)

print("\nFirst 5 cases:")
print(
    cases_df[
        [
            "case_id",
            "candidate_dtcs",
            "candidate_count",
            "primary_dtc",
            "secondary_dtcs",
        ]
    ].head().to_string(index=False)
)

print("\nEvaluation-only columns:")
print(
    sorted(evaluation_only_columns)
)

print("\nAlgorithm-input columns:")
print(
    sorted(algorithm_input_columns)
)

print("\nSTATUS: PASS")