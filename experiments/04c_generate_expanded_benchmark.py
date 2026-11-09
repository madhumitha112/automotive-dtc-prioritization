"""
Experiment 04C — Expanded Synthetic Benchmark V4
================================================

Creates a separate benchmark version under:
    data/expanded_v4/raw/
    results/expanded_v4/

It does NOT overwrite V3 files or prior experiment outputs.

The V4 benchmark expands candidate coverage for all eight DTCs.
Each of 12 candidate/primary templates is generated twice with
independent seeded signal noise. These are repeated template variants,
not 24 fully independent real-world scenarios.

All benchmark data are controlled synthetic data. Diagnostic event
metadata is stored separately from CAN payloads and is evaluation-only.
"""

from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_GENERATOR = ROOT / "src" / "dataset_generator" / "generate_dataset.py"

RAW_DIR = ROOT / "data" / "expanded_v4" / "raw"
RESULTS_DIR = ROOT / "results" / "expanded_v4"

CAN_FILE = RAW_DIR / "can_diagnostic_dataset_v4.csv"
EVENT_FILE = RAW_DIR / "diagnostic_events_v4.csv"
MANIFEST_FILE = RAW_DIR / "dataset_manifest_v4.json"
CASES_FILE = RESULTS_DIR / "diagnostic_cases_v4.csv"
SCENARIO_FILE = RESULTS_DIR / "scenario_manifest_v4.csv"

SEED = 20261105
REPETITIONS = 2
SESSION_DURATION_S = 1800
FAULT_START_S = 600
FAULT_END_S = 1350
DIAG_PERIOD_S = 10

# Candidate order is deliberately not treated as the correct ranking.
# The primary DTC is evaluation-only and is not passed to the ranker.
TEMPLATES = [
    {
        "template_id": "V4_01",
        "primary": "P0171",
        "candidates": ["P0171", "P0172"],
        "condition": "LEAN_FUEL",
        "coverage_role": "P0171 primary; P0172 competing",
        "context_design": "positive fuel-trim evidence",
    },
    {
        "template_id": "V4_02",
        "primary": "P0172",
        "candidates": ["P0172", "P0171"],
        "condition": "RICH_FUEL",
        "coverage_role": "P0172 primary; P0171 competing",
        "context_design": "negative fuel-trim evidence",
    },
    {
        "template_id": "V4_03",
        "primary": "P0128",
        "candidates": ["P0128", "P0171"],
        "condition": "LOW_COOLANT_TEMP",
        "coverage_role": "P0128 primary; P0171 competing",
        "context_design": "low coolant-temperature evidence",
    },
    {
        "template_id": "V4_04",
        "primary": "P0171",
        "candidates": ["P0171", "P0128"],
        "condition": "LEAN_FUEL",
        "coverage_role": "P0128 competing",
        "context_design": "fuel-trim evidence without low-coolant evidence",
    },
    {
        "template_id": "V4_05",
        "primary": "C0035",
        "candidates": ["C0035", "P0121"],
        "condition": "LEFT_WHEEL_SPEED",
        "coverage_role": "C0035 primary; P0121 competing",
        "context_design": "left-front wheel-speed disagreement",
    },
    {
        "template_id": "V4_06",
        "primary": "P0121",
        "candidates": ["P0121", "C0035"],
        "condition": "THROTTLE_MISMATCH",
        "coverage_role": "C0035 competing",
        "context_design": "pedal/throttle mismatch without wheel-speed fault",
    },
    {
        "template_id": "V4_07",
        "primary": "P0300",
        "candidates": ["P0300", "P0335"],
        "condition": "MISFIRE_LOAD",
        "coverage_role": "P0300 primary; P0335 competing",
        "context_design": "load-associated RPM drops",
    },
    {
        "template_id": "V4_08",
        "primary": "P0335",
        "candidates": ["P0335", "P0300"],
        "condition": "CRANK_SIGNAL",
        "coverage_role": "P0335 primary; P0300 competing",
        "context_design": "intermittent RPM drops; indirect CKP proxy only",
    },
    {
        "template_id": "V4_09",
        "primary": "P0130",
        "candidates": ["P0130", "P0171"],
        "condition": "O2_BIAS",
        "coverage_role": "P0130 primary; P0171 competing",
        "context_design": "oxygen-signal bias",
    },
    {
        "template_id": "V4_10",
        "primary": "P0171",
        "candidates": ["P0171", "P0130"],
        "condition": "LEAN_FUEL",
        "coverage_role": "P0130 competing",
        "context_design": "positive fuel-trim evidence",
    },
    {
        "template_id": "V4_11",
        "primary": "P0121",
        "candidates": ["P0121", "P0300"],
        "condition": "THROTTLE_MISMATCH",
        "coverage_role": "P0300 competing",
        "context_design": "throttle mismatch without imposed RPM-drop pattern",
    },
    {
        "template_id": "V4_12",
        "primary": "P0300",
        "candidates": ["P0300", "P0121"],
        "condition": "MISFIRE_LOAD",
        "coverage_role": "P0300 primary; P0121 competing",
        "context_design": "load-associated RPM drops",
    },
]


def load_base_generator():
    if not BASE_GENERATOR.exists():
        raise FileNotFoundError(
            f"Base V3 generator not found: {BASE_GENERATOR}\n"
            "Keep this V4 script inside the project's experiments folder."
        )

    spec = importlib.util.spec_from_file_location(
        "v3_dataset_generator", BASE_GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base_generator()
    rng = np.random.default_rng(SEED)

    can_rows = []
    event_rows = []
    case_rows = []
    scenario_rows = []

    episode_index = 0

    for template in TEMPLATES:
        for repetition in range(1, REPETITIONS + 1):
            episode_index += 1
            session_id = f"V4_{episode_index:03d}"

            primary = template["primary"]
            candidates = list(template["candidates"])
            condition = template["condition"]

            if primary not in candidates:
                raise ValueError(
                    f"{template['template_id']}: primary is not a candidate."
                )

            if len(candidates) < 2 or len(set(candidates)) != len(candidates):
                raise ValueError(
                    f"{template['template_id']}: invalid candidate set."
                )

            # Generate the observable CAN context.
            for timestamp_s in range(SESSION_DURATION_S):
                values = base.scenario_values(
                    timestamp_s, condition, rng
                )

                frames = base.encode_context_frames(
                    session_id, timestamp_s, values
                )

                for can_id, data, message_name in frames:
                    can_rows.append(
                        base.make_can_row(
                            session_id,
                            timestamp_s,
                            can_id,
                            data,
                            "context",
                            message_name,
                        )
                    )

            # Emit diagnostic response traffic and separate event metadata.
            diagnostic_timestamps = range(
                FAULT_START_S,
                FAULT_END_S + 1,
                DIAG_PERIOD_S,
            )

            for dtc in candidates:
                role = "primary" if dtc == primary else "secondary"

                event_rows.append(
                    [
                        session_id,
                        dtc,
                        base.DTC_ECU[dtc],
                        FAULT_START_S,
                        FAULT_END_S,
                        role,
                        condition,
                        template["template_id"],
                        repetition,
                    ]
                )

                for timestamp_s in diagnostic_timestamps:
                    can_rows.append(
                        base.make_diag_row(
                            session_id,
                            timestamp_s,
                            dtc,
                        )
                    )

            case_rows.append(
                {
                    "case_id": session_id,
                    "session_id": session_id,
                    "candidate_dtcs": "|".join(candidates),
                    "candidate_count": len(candidates),
                    "observation_start_s": FAULT_START_S,
                    "observation_end_s": FAULT_END_S,
                    "observation_duration_s": (
                        FAULT_END_S - FAULT_START_S
                    ),

                    # Evaluation-only fields below:
                    "primary_dtc": primary,
                    "secondary_dtcs": "|".join(
                        d for d in candidates if d != primary
                    ),
                    "context_condition": condition,
                    "template_id": template["template_id"],
                    "repetition": repetition,
                }
            )

            scenario_rows.append(
                {
                    "session_id": session_id,
                    "template_id": template["template_id"],
                    "repetition": repetition,
                    "primary_dtc": primary,
                    "candidate_dtcs": "|".join(candidates),
                    "context_condition": condition,
                    "coverage_role": template["coverage_role"],
                    "context_design": template["context_design"],
                }
            )

    can_columns = [
        "session_id",
        "timestamp_s",
        "CAN_ID",
        "DLC",
        *[f"DATA{i}" for i in range(8)],
        "frame_type",
        "message_name",
    ]

    event_columns = [
        "session_id",
        "dtc",
        "ecu",
        "event_start_s",
        "event_end_s",
        "event_role",
        "context_condition",
        "template_id",
        "repetition",
    ]

    can_df = pd.DataFrame(can_rows, columns=can_columns)
    events_df = pd.DataFrame(event_rows, columns=event_columns)
    cases_df = pd.DataFrame(case_rows)
    scenario_df = pd.DataFrame(scenario_rows)

    # Integrity checks: raw CAN has no evaluation labels.
    forbidden_can_columns = {
        "primary_dtc",
        "secondary_dtcs",
        "context_condition",
        "event_role",
        "template_id",
        "repetition",
    }

    leaked = forbidden_can_columns.intersection(can_df.columns)

    if leaked:
        raise RuntimeError(
            f"Evaluation metadata leaked into CAN table: {sorted(leaked)}"
        )

    data_columns = [f"DATA{i}" for i in range(8)]

    if can_df[data_columns].isna().any().any():
        raise RuntimeError("Null CAN payload values detected.")

    if not can_df[data_columns].apply(
        lambda col: col.between(0, 255).all()
    ).all():
        raise RuntimeError("CAN payload contains values outside 0..255.")

    if len(cases_df) != len(TEMPLATES) * REPETITIONS:
        raise RuntimeError("Unexpected number of diagnostic cases.")

    if cases_df["case_id"].duplicated().any():
        raise RuntimeError("Duplicate case IDs.")

    if set(cases_df["primary_dtc"]) - set(base.DTC_ECU):
        raise RuntimeError("Unknown primary DTC.")

    for _, case in cases_df.iterrows():
        candidates = case["candidate_dtcs"].split("|")

        if case["primary_dtc"] not in candidates:
            raise RuntimeError(
                f"Primary DTC missing from {case['case_id']} candidates."
            )

        if len(candidates) != len(set(candidates)):
            raise RuntimeError(
                f"Duplicate candidate DTC in {case['case_id']}."
            )

    # Validate diagnostic payload encoding against the event table.
    diagnostic = can_df[
        can_df["frame_type"] == "diagnostic"
    ]

    for row in diagnostic.itertuples(index=False):
        payload = [
            int(getattr(row, f"DATA{i}"))
            for i in range(8)
        ]

        if (
            payload[0] != 0x07
            or payload[1] != 0x59
            or payload[2] != 0x02
        ):
            raise RuntimeError(
                "Invalid UDS diagnostic response payload."
            )

        decoded = base.uds_bytes_to_dtc(
            payload[4],
            payload[5],
            payload[6],
        )

        expected = set(
            events_df.loc[
                (events_df["session_id"] == row.session_id)
                & (
                    events_df["event_start_s"]
                    <= row.timestamp_s
                )
                & (
                    events_df["event_end_s"]
                    >= row.timestamp_s
                ),
                "dtc",
            ]
        )

        if decoded not in expected:
            raise RuntimeError(
                "Diagnostic payload/event mismatch: "
                f"{row.session_id}, {decoded}"
            )

    # Save generated datasets.
    can_df.to_csv(CAN_FILE, index=False)
    events_df.to_csv(EVENT_FILE, index=False)
    cases_df.to_csv(CASES_FILE, index=False)
    scenario_df.to_csv(SCENARIO_FILE, index=False)

    manifest = {
        "dataset_name": "automotive-dtc-benchmark-v4-expanded",
        "dataset_type": (
            "controlled synthetic automotive diagnostic benchmark"
        ),
        "provenance": (
            "synthetically generated; not recorded from a physical vehicle"
        ),
        "seed": SEED,
        "episodes": len(cases_df),
        "template_count": len(TEMPLATES),
        "repetitions_per_template": REPETITIONS,
        "session_duration_s": SESSION_DURATION_S,
        "context_rate_hz": 1,
        "diagnostic_period_s": DIAG_PERIOD_S,
        "observation_interval_s": [
            FAULT_START_S,
            FAULT_END_S,
        ],
        "unique_dtcs": sorted(base.DTC_ECU.keys()),
        "dtc_ecu": base.DTC_ECU,
        "can_message_ids": [
            "0x100",
            "0x101",
            "0x102",
            "0x103",
        ],
        "diagnostic_protocol": {
            "service": "UDS ReadDTCInformation",
            "service_id": "0x19",
            "subfunction": (
                "0x02 reportDTCByStatusMask"
            ),
            "positive_response": "0x59",
            "transport": "ISO-TP single-frame CAN payload",
        },
        "evaluation_metadata_separate_from_can": True,
        "templates": TEMPLATES,
        "limitations": [
            (
                "The dataset is synthetic and does not establish "
                "real-vehicle diagnostic performance."
            ),
            (
                "Each template is repeated twice with different random "
                "signal noise; repetitions are not independent scenario families."
            ),
            (
                "The candidate list is treated as already supplied by a "
                "diagnostic layer; this experiment evaluates prioritization, "
                "not DTC detection."
            ),
            (
                "P0335 is represented only by an indirect RPM-instability "
                "proxy; no CKP waveform is simulated."
            ),
            (
                "Context conditions and primary labels are benchmark metadata "
                "and must not be used as algorithm inputs."
            ),
        ],
    }

    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    # Coverage summaries to make the expansion auditable.
    coverage = []

    for dtc in sorted(base.DTC_ECU):
        primary_count = int(
            (cases_df["primary_dtc"] == dtc).sum()
        )

        candidate_count = int(
            cases_df["candidate_dtcs"]
            .str.split("|")
            .apply(lambda xs: dtc in xs)
            .sum()
        )

        coverage.append(
            {
                "dtc": dtc,
                "primary_cases": primary_count,
                "competing_cases": candidate_count - primary_count,
                "total_candidate_cases": candidate_count,
            }
        )

    coverage_df = pd.DataFrame(coverage)

    coverage_df.to_csv(
        RESULTS_DIR / "04c_dtc_coverage_summary.csv",
        index=False,
    )

    print("=" * 72)
    print("EXPERIMENT 04C — EXPANDED BENCHMARK V4")
    print("=" * 72)

    print(f"Cases:              {len(cases_df)}")
    print(f"CAN rows:           {len(can_df):,}")
    print(f"Diagnostic events:  {len(events_df)}")

    print(
        "Context frames:     "
        f"{(can_df['frame_type'] == 'context').sum():,}"
    )

    print(
        "Diagnostic frames:  "
        f"{(can_df['frame_type'] == 'diagnostic').sum():,}"
    )

    print(f"Unique DTCs:        {events_df['dtc'].nunique()}")

    print("\nDTC candidate coverage:")
    print(coverage_df.to_string(index=False))

    print("\nFiles written:")

    for path in [
        CAN_FILE,
        EVENT_FILE,
        CASES_FILE,
        SCENARIO_FILE,
        MANIFEST_FILE,
    ]:
        print(path)

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()