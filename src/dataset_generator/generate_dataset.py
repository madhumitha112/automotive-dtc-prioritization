from pathlib import Path
import json
import math
import numpy as np
import pandas as pd


# ============================================================
# DATASET CONFIGURATION
# ============================================================

SEED = 20261103

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CAN_FILE = RAW_DIR / "can_diagnostic_dataset.csv"
EVENT_FILE = RAW_DIR / "diagnostic_events.csv"
MANIFEST_FILE = RAW_DIR / "dataset_manifest.json"

EPISODES = 48
SESSION_DURATION_S = 1800

FAULT_START_S = 600
FAULT_END_S = 1350

BASE_HZ = 1
DIAG_PERIOD_S = 10

RNG = np.random.default_rng(SEED)


# ============================================================
# DTC DEFINITIONS
# ============================================================

DTC_ECU = {
    "P0171": "ENGINE",
    "P0172": "ENGINE",
    "P0128": "ENGINE",
    "P0300": "ENGINE",
    "P0130": "ENGINE",
    "P0121": "ENGINE",
    "P0335": "ENGINE",
    "C0035": "ABS",
}


DTC_DESCRIPTION = {
    "P0171": "System too lean",
    "P0172": "System too rich",
    "P0128": "Coolant thermostat temperature below regulating temperature",
    "P0300": "Random or multiple cylinder misfire detected",
    "P0130": "Oxygen sensor circuit malfunction",
    "P0121": "Throttle position sensor range/performance",
    "P0335": "Crankshaft position sensor circuit malfunction",
    "C0035": "Left front wheel speed sensor circuit",
}


# ============================================================
# SCENARIO DEFINITIONS
# ============================================================

# Each scenario contains:
#
# primary DTC
# secondary DTC candidate(s)
# context condition used to generate the CAN behaviour
#
# These metadata fields are NOT included in the CAN payload.
# They are only used later for experimental evaluation.

SCENARIOS = [
    ("P0171", ["P0130"], "LEAN_FUEL"),
    ("P0172", ["P0130"], "RICH_FUEL"),
    ("P0128", ["P0171"], "LOW_COOLANT_TEMP"),
    ("P0300", ["P0121"], "MISFIRE_LOAD"),
    ("P0130", ["P0171"], "O2_BIAS"),
    ("P0121", ["P0171"], "THROTTLE_MISMATCH"),
    ("P0335", ["P0300"], "CRANK_SIGNAL"),
    ("C0035", ["P0121"], "LEFT_WHEEL_SPEED"),
    ("P0171", ["P0121"], "LEAN_THROTTLE"),
    ("P0172", ["P0300"], "RICH_MISFIRE"),
    ("P0128", ["P0130"], "THERMAL_O2"),
    ("P0300", ["P0335"], "MISFIRE_CRANK"),
]


# ============================================================
# DTC ENCODING / DECODING
# ============================================================

def dtc_to_uds_bytes(dtc: str, failure_type: int = 0x00):
    """
    Encode a 5-character SAE J2012-style DTC into
    a 3-byte DTC representation.

    Examples:

        P0171 -> 01 71 00
        P0172 -> 01 72 00
        P0300 -> 03 00 00
        C0035 -> 40 35 00

    The third byte is the Failure Type Byte.
    """

    if len(dtc) != 5:
        raise ValueError(f"Unsupported DTC format: {dtc}")

    prefix_char = dtc[0]

    if prefix_char not in "PCBU":
        raise ValueError(f"Unsupported DTC prefix: {dtc}")

    prefix = {
        "P": 0b00,
        "C": 0b01,
        "B": 0b10,
        "U": 0b11,
    }[prefix_char]

    second_digit = int(dtc[1], 10)
    third_digit = int(dtc[2], 16)
    remaining = int(dtc[3:5], 16)

    byte0 = (
        (prefix << 6)
        | (second_digit << 4)
        | third_digit
    )

    byte1 = remaining
    byte2 = failure_type & 0xFF

    return [byte0, byte1, byte2]


def uds_bytes_to_dtc(byte0: int, byte1: int, byte2: int):
    """
    Decode the first two bytes of the 3-byte DTC representation.

    The third byte is the Failure Type Byte and is intentionally
    not included in the 5-character base DTC string.
    """

    prefix_code = (byte0 >> 6) & 0x03

    prefix = {
        0: "P",
        1: "C",
        2: "B",
        3: "U",
    }[prefix_code]

    second_digit = (byte0 >> 4) & 0x03
    third_digit = byte0 & 0x0F

    return f"{prefix}{second_digit}{third_digit:X}{byte1:02X}"


def validate_dtc_round_trip():
    """
    Ensure every DTC used by the benchmark can be encoded
    and decoded without changing its identity.
    """

    for dtc in DTC_ECU:

        encoded = dtc_to_uds_bytes(dtc)

        decoded = uds_bytes_to_dtc(
            encoded[0],
            encoded[1],
            encoded[2],
        )

        if decoded != dtc:
            raise RuntimeError(
                f"DTC round-trip failed: "
                f"{dtc} -> {encoded} -> {decoded}"
            )


# ============================================================
# UDS DIAGNOSTIC PAYLOAD
# ============================================================

def make_diag_payload(dtc: str, status: int = 0x09):
    """
    Construct an ISO-TP single-frame representation of:

        0x59
        0x02
        0xFF
        DTC byte 1
        DTC byte 2
        DTC byte 3
        DTC status

    0x59 = positive response to UDS service 0x19

    0x02 = reportDTCByStatusMask

    0xFF = DTC status availability mask

    status 0x09:
        bit 0 = testFailed
        bit 3 = confirmedDTC

    The first CAN byte is the ISO-TP single-frame
    payload length.

    Example for P0171:

        07 59 02 FF 01 71 00 09
    """

    d0, d1, d2 = dtc_to_uds_bytes(dtc)

    uds_payload = [
        0x59,
        0x02,
        0xFF,
        d0,
        d1,
        d2,
        status,
    ]

    iso_tp_length = len(uds_payload)

    return [
        iso_tp_length,
        *uds_payload,
    ]


# ============================================================
# GENERAL HELPERS
# ============================================================

def clip(value, minimum, maximum):
    return float(
        np.clip(value, minimum, maximum)
    )


def quantize(
    value,
    scale,
    offset=0,
    minimum=0,
    maximum=255,
):
    raw = round(
        (value - offset) / scale
    )

    return int(
        np.clip(
            raw,
            minimum,
            maximum,
        )
    )


# ============================================================
# VEHICLE CONTEXT GENERATION
# ============================================================

def scenario_values(
    t,
    scenario,
    rng,
):
    """
    Generate physically plausible vehicle-context values.

    The scenario modifies observable signals.

    No DTC identifier is inserted into the signals.
    """

    phase = t / 60.0

    # --------------------------------------------------------
    # Normal operating state
    # --------------------------------------------------------

    vehicle_speed = clip(
        42
        + 18 * math.sin(phase / 3.0)
        + 8 * math.sin(phase / 7.0)
        + rng.normal(0, 1.2),
        0,
        120,
    )

    rpm = clip(
        850
        + vehicle_speed * 42
        + 180 * math.sin(phase / 2.7)
        + rng.normal(0, 55),
        650,
        6000,
    )

    throttle = clip(
        8
        + vehicle_speed * 0.18
        + 4 * math.sin(phase / 2.5)
        + rng.normal(0, 1.0),
        0,
        100,
    )

    accelerator_pedal = clip(
        throttle
        + 3 * math.sin(phase / 4.0)
        + rng.normal(0, 1.0),
        0,
        100,
    )

    engine_load = clip(
        18
        + throttle * 0.62
        + rng.normal(0, 2.0),
        0,
        100,
    )

    coolant_temp = clip(
        88
        + 2.5 * math.sin(phase / 8.0)
        + rng.normal(0, 0.6),
        20,
        115,
    )

    intake_air_temp = clip(
        30
        + 3 * math.sin(phase / 10.0)
        + rng.normal(0, 0.8),
        -10,
        70,
    )

    map_kpa = clip(
        32
        + engine_load * 0.55
        + rng.normal(0, 1.4),
        20,
        110,
    )

    oxygen_signal = clip(
        0.45
        + 0.08 * math.sin(phase / 1.7)
        + rng.normal(0, 0.025),
        0.05,
        0.95,
    )

    stft = clip(
        rng.normal(0, 1.5),
        -25,
        25,
    )

    ltft = clip(
        rng.normal(0, 1.0),
        -20,
        20,
    )

    brake_state = (
        1
        if math.sin(t / 40.0) > 0.92
        else 0
    )

    # --------------------------------------------------------
    # Scenario-specific context
    # --------------------------------------------------------

    if scenario == "LEAN_FUEL":

        stft = clip(
            18 + rng.normal(0, 2.0),
            -25,
            30,
        )

        ltft = clip(
            15 + rng.normal(0, 2.0),
            -25,
            30,
        )

        oxygen_signal = clip(
            0.25 + rng.normal(0, 0.03),
            0.05,
            0.55,
        )

    elif scenario == "RICH_FUEL":

        stft = clip(
            -18 + rng.normal(0, 2.0),
            -30,
            25,
        )

        ltft = clip(
            -15 + rng.normal(0, 2.0),
            -30,
            25,
        )

        oxygen_signal = clip(
            0.72 + rng.normal(0, 0.03),
            0.45,
            0.95,
        )

    elif scenario == "LOW_COOLANT_TEMP":

        coolant_temp = clip(
            62
            + 3 * math.sin(phase / 5.0)
            + rng.normal(0, 1.0),
            45,
            75,
        )

    elif scenario == "MISFIRE_LOAD":

        if engine_load > 45:

            rpm = clip(
                rpm
                - 280
                + rng.normal(0, 90),
                650,
                6000,
            )

    elif scenario == "O2_BIAS":

        oxygen_signal = clip(
            0.68 + rng.normal(0, 0.02),
            0.50,
            0.90,
        )

        stft = clip(
            7 + rng.normal(0, 1.5),
            -25,
            25,
        )

    elif scenario == "THROTTLE_MISMATCH":

        throttle = clip(
            throttle * 0.55
            + rng.normal(0, 1.0),
            0,
            100,
        )

        accelerator_pedal = clip(
            accelerator_pedal * 1.15
            + rng.normal(0, 1.0),
            0,
            100,
        )

        engine_load = clip(
            engine_load
            + 8
            + rng.normal(0, 1.5),
            0,
            100,
        )

    elif scenario == "CRANK_SIGNAL":

        if int(t) % 17 == 0:

            rpm = clip(
                rpm
                - 450
                + rng.normal(0, 80),
                650,
                6000,
            )

    elif scenario == "LEAN_THROTTLE":

        throttle = clip(
            throttle * 0.78
            + rng.normal(0, 1.0),
            0,
            100,
        )

        stft = clip(
            12 + rng.normal(0, 2.0),
            -25,
            25,
        )

    elif scenario == "RICH_MISFIRE":

        stft = clip(
            -13 + rng.normal(0, 2.0),
            -30,
            25,
        )

        if engine_load > 40:

            rpm = clip(
                rpm
                - 220
                + rng.normal(0, 70),
                650,
                6000,
            )

    elif scenario == "THERMAL_O2":

        coolant_temp = clip(
            68 + rng.normal(0, 1.0),
            50,
            78,
        )

        oxygen_signal = clip(
            0.58 + rng.normal(0, 0.025),
            0.40,
            0.80,
        )

    elif scenario == "MISFIRE_CRANK":

        if int(t) % 19 == 0:

            rpm = clip(
                rpm
                - 520
                + rng.normal(0, 90),
                650,
                6000,
            )

    # --------------------------------------------------------
    # Wheel speeds
    # --------------------------------------------------------

    wheel_base = vehicle_speed

    wheel_speed_fl = (
        wheel_base
        + rng.normal(0, 0.5)
    )

    wheel_speed_fr = (
        wheel_base
        + rng.normal(0, 0.5)
    )

    wheel_speed_rl = (
        wheel_base
        + rng.normal(0, 0.5)
    )

    wheel_speed_rr = (
        wheel_base
        + rng.normal(0, 0.5)
    )

    if scenario == "LEFT_WHEEL_SPEED":

        wheel_speed_fl = clip(
            wheel_base * 0.68
            + rng.normal(0, 0.8),
            0,
            130,
        )

    return {
        "rpm": rpm,
        "vehicle_speed": vehicle_speed,
        "engine_load": engine_load,
        "throttle": throttle,
        "accelerator_pedal": accelerator_pedal,
        "coolant_temp": coolant_temp,
        "intake_air_temp": intake_air_temp,
        "map_kpa": map_kpa,
        "oxygen_signal": oxygen_signal,
        "stft": stft,
        "ltft": ltft,
        "brake_state": brake_state,
        "wheel_speed_fl": wheel_speed_fl,
        "wheel_speed_fr": wheel_speed_fr,
        "wheel_speed_rl": wheel_speed_rl,
        "wheel_speed_rr": wheel_speed_rr,
    }


# ============================================================
# CAN CONTEXT MESSAGE ENCODING
# ============================================================

def encode_context_frames(
    session_id,
    timestamp_s,
    values,
):
    """
    Generate four context CAN messages per second.

    0x100 - engine dynamics
    0x101 - thermal / air
    0x102 - driver / fuel
    0x103 - wheel speeds
    """

    frames = []

    # --------------------------------------------------------
    # 0x100 ENGINE_DYNAMICS
    # --------------------------------------------------------

    rpm_raw = int(
        np.clip(
            round(values["rpm"]),
            0,
            65535,
        )
    )

    speed_raw = quantize(
        values["vehicle_speed"],
        0.1,
        0,
        0,
        65535,
    )

    load_raw = quantize(
        values["engine_load"],
        100 / 255,
        0,
        0,
        255,
    )

    throttle_raw = quantize(
        values["throttle"],
        100 / 255,
        0,
        0,
        255,
    )

    frames.append(
        (
            0x100,
            [
                rpm_raw & 0xFF,
                (rpm_raw >> 8) & 0xFF,

                speed_raw & 0xFF,
                (speed_raw >> 8) & 0xFF,

                load_raw,
                throttle_raw,

                0,
                0,
            ],
            "ENGINE_DYNAMICS",
        )
    )

    # --------------------------------------------------------
    # 0x101 ENGINE_THERMAL_AIR
    # --------------------------------------------------------

    coolant_raw = quantize(
        values["coolant_temp"],
        0.5,
        -40,
        0,
        255,
    )

    iat_raw = quantize(
        values["intake_air_temp"],
        0.5,
        -40,
        0,
        255,
    )

    map_raw = quantize(
        values["map_kpa"],
        0.5,
        0,
        0,
        255,
    )

    oxygen_raw = quantize(
        values["oxygen_signal"],
        5 / 255,
        0,
        0,
        255,
    )

    frames.append(
        (
            0x101,
            [
                coolant_raw,
                iat_raw,
                map_raw,
                oxygen_raw,

                0,
                0,
                0,
                0,
            ],
            "ENGINE_THERMAL_AIR",
        )
    )

    # --------------------------------------------------------
    # 0x102 DRIVER_FUEL
    # --------------------------------------------------------

    stft_raw = int(
        np.clip(
            round(
                values["stft"] + 128
            ),
            0,
            255,
        )
    )

    ltft_raw = int(
        np.clip(
            round(
                values["ltft"] + 128
            ),
            0,
            255,
        )
    )

    pedal_raw = quantize(
        values["accelerator_pedal"],
        100 / 255,
        0,
        0,
        255,
    )

    frames.append(
        (
            0x102,
            [
                stft_raw,
                ltft_raw,
                pedal_raw,
                int(values["brake_state"]),

                0,
                0,
                0,
                0,
            ],
            "DRIVER_FUEL",
        )
    )

    # --------------------------------------------------------
    # 0x103 WHEEL_SPEEDS
    # --------------------------------------------------------

    wheel_values = [
        quantize(
            values["wheel_speed_fl"],
            0.5,
            0,
            0,
            255,
        ),

        quantize(
            values["wheel_speed_fr"],
            0.5,
            0,
            0,
            255,
        ),

        quantize(
            values["wheel_speed_rl"],
            0.5,
            0,
            0,
            255,
        ),

        quantize(
            values["wheel_speed_rr"],
            0.5,
            0,
            0,
            255,
        ),
    ]

    frames.append(
        (
            0x103,
            wheel_values
            + [0, 0, 0, 0],
            "WHEEL_SPEEDS",
        )
    )

    return frames


# ============================================================
# RAW CAN ROW CONSTRUCTION
# ============================================================

def make_can_row(
    session_id,
    timestamp_s,
    can_id,
    data,
    frame_type,
    message_name,
):
    data = list(data)

    if len(data) != 8:
        raise ValueError(
            f"CAN payload must contain 8 bytes. "
            f"Got {len(data)}."
        )

    return [
        session_id,
        timestamp_s,
        can_id,
        8,

        data[0],
        data[1],
        data[2],
        data[3],
        data[4],
        data[5],
        data[6],
        data[7],

        frame_type,
        message_name,
    ]


def make_diag_row(
    session_id,
    timestamp_s,
    dtc,
):
    payload = make_diag_payload(
        dtc,
        status=0x09,
    )

    ecu = DTC_ECU[dtc]

    if ecu == "ENGINE":
        can_id = 0x7E8
        message_name = "ENGINE_UDS_0x19_0x02"

    else:
        can_id = 0x7E9
        message_name = "ABS_UDS_0x19_0x02"

    padded_payload = (
        payload
        + [0] * (8 - len(payload))
    )

    return make_can_row(
        session_id,
        timestamp_s,
        can_id,
        padded_payload,
        "diagnostic",
        message_name,
    )


# ============================================================
# EPISODE PLAN
# ============================================================

def build_episode_plan():

    plans = []

    for index in range(EPISODES):

        primary, secondary, condition = (
            SCENARIOS[
                index % len(SCENARIOS)
            ]
        )

        secondary = list(secondary)

        # Introduce additional candidate DTCs in some
        # episodes so that ranking is not always binary.
        if index % 4 == 2:

            additional_candidates = [
                "P0121",
                "P0171",
                "P0130",
                "P0300",
                "P0335",
                "C0035",
            ]

            for candidate in additional_candidates:

                if (
                    candidate != primary
                    and candidate not in secondary
                ):
                    secondary.append(candidate)
                    break

        plans.append(
            {
                "session_id": f"S{index + 1:02d}",
                "primary_dtc": primary,
                "secondary_dtcs": secondary,
                "context_condition": condition,
            }
        )

    return plans


# ============================================================
# MAIN GENERATION
# ============================================================

def generate_dataset():

    print("=" * 60)
    print("AUTOMOTIVE DTC BENCHMARK V3")
    print("=" * 60)

    print("Running DTC encoding round-trip tests...")

    validate_dtc_round_trip()

    print("DTC round-trip checks: PASSED")
    print()

    plans = build_episode_plan()

    can_rows = []
    event_rows = []

    # --------------------------------------------------------
    # Generate each episode
    # --------------------------------------------------------

    for plan in plans:

        session_id = plan["session_id"]

        primary_dtc = plan["primary_dtc"]

        secondary_dtcs = plan["secondary_dtcs"]

        all_dtcs = (
            [primary_dtc]
            + secondary_dtcs
        )

        # ----------------------------------------------------
        # Context CAN traffic
        # ----------------------------------------------------

        for timestamp_s in range(
            SESSION_DURATION_S
        ):

            values = scenario_values(
                timestamp_s,
                plan["context_condition"],
                RNG,
            )

            context_frames = (
                encode_context_frames(
                    session_id,
                    timestamp_s,
                    values,
                )
            )

            for (
                can_id,
                data,
                message_name,
            ) in context_frames:

                can_rows.append(
                    make_can_row(
                        session_id,
                        timestamp_s,
                        can_id,
                        data,
                        "context",
                        message_name,
                    )
                )

        # ----------------------------------------------------
        # Diagnostic events
        # ----------------------------------------------------

        diagnostic_timestamps = range(
            FAULT_START_S,
            FAULT_END_S + 1,
            DIAG_PERIOD_S,
        )

        for dtc_index, dtc in enumerate(
            all_dtcs
        ):

            event_role = (
                "primary"
                if dtc_index == 0
                else "secondary"
            )

            event_rows.append(
                [
                    session_id,
                    dtc,
                    DTC_ECU[dtc],
                    FAULT_START_S,
                    FAULT_END_S,
                    event_role,
                    plan["context_condition"],
                ]
            )

            for timestamp_s in diagnostic_timestamps:

                can_rows.append(
                    make_diag_row(
                        session_id,
                        timestamp_s,
                        dtc,
                    )
                )

    # ========================================================
    # DATAFRAME CREATION
    # ========================================================

    can_columns = [
        "session_id",
        "timestamp_s",
        "CAN_ID",
        "DLC",

        "DATA0",
        "DATA1",
        "DATA2",
        "DATA3",
        "DATA4",
        "DATA5",
        "DATA6",
        "DATA7",

        "frame_type",
        "message_name",
    ]

    can_df = pd.DataFrame(
        can_rows,
        columns=can_columns,
    )

    event_df = pd.DataFrame(
        event_rows,
        columns=[
            "session_id",
            "dtc",
            "ecu",
            "event_start_s",
            "event_end_s",
            "event_role",
            "context_condition",
        ],
    )

    # ========================================================
    # BASIC DATA INTEGRITY CHECKS
    # ========================================================

    if can_df.empty:
        raise RuntimeError(
            "CAN dataset is empty."
        )

    if event_df.empty:
        raise RuntimeError(
            "Diagnostic event dataset is empty."
        )

    if can_df[
        [
            "DATA0",
            "DATA1",
            "DATA2",
            "DATA3",
            "DATA4",
            "DATA5",
            "DATA6",
            "DATA7",
        ]
    ].isnull().any().any():

        raise RuntimeError(
            "Null CAN payload values detected."
        )

    if (
        can_df[
            [
                "DATA0",
                "DATA1",
                "DATA2",
                "DATA3",
                "DATA4",
                "DATA5",
                "DATA6",
                "DATA7",
            ]
        ]
        .lt(0)
        .any()
        .any()
    ):

        raise RuntimeError(
            "Negative CAN byte value detected."
        )

    if (
        can_df[
            [
                "DATA0",
                "DATA1",
                "DATA2",
                "DATA3",
                "DATA4",
                "DATA5",
                "DATA6",
                "DATA7",
            ]
        ]
        .gt(255)
        .any()
        .any()
    ):

        raise RuntimeError(
            "CAN byte greater than 255 detected."
        )

    # ========================================================
    # DIAGNOSTIC PAYLOAD VALIDATION
    # ========================================================

    diagnostic_df = can_df[
        can_df["frame_type"]
        == "diagnostic"
    ].copy()

    for _, row in diagnostic_df.iterrows():

        data = [
            int(row[f"DATA{i}"])
            for i in range(8)
        ]

        # Expected ISO-TP single-frame:
        #
        # 07 59 02 FF XX XX XX STATUS

        if data[0] != 0x07:
            raise RuntimeError(
                "Invalid ISO-TP diagnostic "
                f"length byte: {data}"
            )

        if data[1] != 0x59:
            raise RuntimeError(
                "Invalid UDS positive response "
                f"SID: {data}"
            )

        if data[2] != 0x02:
            raise RuntimeError(
                "Invalid UDS ReadDTCInformation "
                f"subfunction: {data}"
            )

        if data[3] != 0xFF:
            raise RuntimeError(
                "Invalid DTC status availability "
                f"mask: {data}"
            )

        decoded_dtc = uds_bytes_to_dtc(
            data[4],
            data[5],
            data[6],
        )

        if decoded_dtc not in DTC_ECU:
            raise RuntimeError(
                "Diagnostic payload contains "
                f"unexpected DTC: {decoded_dtc}"
            )

    # ========================================================
    # WRITE FILES
    # ========================================================

    can_df.to_csv(
        CAN_FILE,
        index=False,
    )

    event_df.to_csv(
        EVENT_FILE,
        index=False,
    )

    # ========================================================
    # DATASET MANIFEST
    # ========================================================

    manifest = {
        "dataset_name":
            "automotive-dtc-benchmark-v3",

        "dataset_type":
            "controlled synthetic automotive "
            "diagnostic benchmark",

        "seed":
            SEED,

        "episodes":
            EPISODES,

        "session_duration_s":
            SESSION_DURATION_S,

        "base_context_rate_hz":
            BASE_HZ,

        "diagnostic_period_s":
            DIAG_PERIOD_S,

        "fault_interval_s": [
            FAULT_START_S,
            FAULT_END_S,
        ],

        "unique_dtcs":
            sorted(DTC_ECU.keys()),

        "dtc_ecu":
            DTC_ECU,

        "dtc_descriptions":
            DTC_DESCRIPTION,

        "diagnostic_protocol": {
            "service":
                "UDS ReadDTCInformation",

            "service_id":
                "0x19",

            "subfunction":
                "0x02 reportDTCByStatusMask",

            "positive_response":
                "0x59",

            "dtc_status_availability_mask":
                "0xFF",

            "dtc_representation":
                "3-byte SAE J2012-style DTC "
                "including Failure Type Byte",

            "transport":
                "ISO-TP single-frame CAN payload",
        },

        "diagnostic_payload_example": {
            "P0171":
                "07 59 02 FF 01 71 00 09",
        },

        "can_message_definitions": {
            "0x100":
                "ENGINE_DYNAMICS",

            "0x101":
                "ENGINE_THERMAL_AIR",

            "0x102":
                "DRIVER_FUEL",

            "0x103":
                "WHEEL_SPEEDS",

            "0x7E8":
                "ENGINE diagnostic response",

            "0x7E9":
                "ABS diagnostic response",
        },

        "research_integrity_note":
            "Primary DTC, secondary DTC, and context-condition "
            "fields are event metadata for experimental evaluation. "
            "They are not present in CAN payloads and must not be "
            "used as input features by the prioritization methods.",
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 60)
    print("AUTOMOTIVE DTC BENCHMARK V3 GENERATED")
    print("=" * 60)

    print(
        f"CAN rows:          "
        f"{len(can_df):,}"
    )

    print(
        f"Diagnostic events: "
        f"{len(event_df):,}"
    )

    print(
        f"Episodes:          "
        f"{EPISODES}"
    )

    print(
        f"Unique DTCs:       "
        f"{event_df['dtc'].nunique()}"
    )

    print(
        f"Context frames:    "
        f"{(can_df['frame_type'] == 'context').sum():,}"
    )

    print(
        f"Diagnostic frames: "
        f"{(can_df['frame_type'] == 'diagnostic').sum():,}"
    )

    print()

    print(
        "DTC encoding tests:       PASSED"
    )

    print(
        "Diagnostic payload tests: PASSED"
    )

    print()

    print(
        f"CAN file:"
    )

    print(
        f"  {CAN_FILE}"
    )

    print(
        f"Events file:"
    )

    print(
        f"  {EVENT_FILE}"
    )

    print(
        f"Manifest:"
    )

    print(
        f"  {MANIFEST_FILE}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate_dataset()