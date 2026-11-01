from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw/obdiidata")
RESULTS_DIR = Path("results")

QUALITY_REPORT = RESULTS_DIR / "05_signal_validation_report.csv"
SUMMARY_FILE = RESULTS_DIR / "05_preprocessing_summary.txt"


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

COLUMN_ALIASES = {
    "ENGINE_RUN_TINE": "ENGINE_RUN_TIME",
    "FUEL_TANK": "FUEL_TANK_LEVEL_INPUT",
}


def normalize_column_name(column):
    """
    Normalize a dataset column name while retaining the
    underlying data unchanged.
    """

    import re

    name = str(column).strip().upper()

    # Remove text inside parentheses
    name = re.sub(r"\([^)]*\)", "", name)

    # Replace separators with underscores
    name = re.sub(r"[^A-Z0-9]+", "_", name)

    name = name.strip("_")

    # Correct known dataset naming variations
    if name in COLUMN_ALIASES:
        name = COLUMN_ALIASES[name]

    return name


# ============================================================
# SIGNALS USED FOR VALIDATION
# ============================================================

IMPORTANT_SIGNALS = [
    "ENGINE_RUN_TIME",
    "ENGINE_RPM",
    "VEHICLE_SPEED",
    "THROTTLE",
    "ENGINE_LOAD",
    "COOLANT_TEMPERATURE",
    "LONG_TERM_FUEL_TRIM_BANK_1",
    "SHORT_TERM_FUEL_TRIM_BANK_1",
    "INTAKE_MANIFOLD_PRESSURE",
    "INTAKE_AIR_TEMPERATURE",
    "CONTROL_MODULE_VOLTAGE",
    "TIME_RUN_WITH_MIL_ON",
    "TIME_SINCE_TROUBLE_CODES_CLEARED",
    "DISTANCE_TRAVELED_WITH_MIL_ON",
    "WARM_UPS_SINCE_CODES_CLEARED",
]


# ============================================================
# FILE ANALYSIS
# ============================================================

def analyze_file(file_path):

    try:

        # ----------------------------------------------------
        # Read raw CSV
        # ----------------------------------------------------

        df = pd.read_csv(file_path)

        rows = len(df)

        # ----------------------------------------------------
        # Normalize column names
        # ----------------------------------------------------

        rename_map = {}

        for column in df.columns:
            rename_map[column] = normalize_column_name(column)

        df = df.rename(columns=rename_map)

        # ----------------------------------------------------
        # Identify available important signals
        # ----------------------------------------------------

        available_signals = [
            signal
            for signal in IMPORTANT_SIGNALS
            if signal in df.columns
        ]

        missing_columns = [
            signal
            for signal in IMPORTANT_SIGNALS
            if signal not in df.columns
        ]

        # ----------------------------------------------------
        # Genuine missing-value analysis
        #
        # IMPORTANT:
        # We do not convert the entire dataframe.
        # Only the required numeric signals are examined.
        # ----------------------------------------------------

        missing_values = 0
        non_numeric_values = 0

        signal_missing_counts = {}

        for signal in available_signals:

            series = df[signal]

            # Genuine empty/null values already present
            null_count = int(series.isna().sum())

            # Attempt numeric conversion ONLY for this signal
            numeric_series = pd.to_numeric(
                series,
                errors="coerce"
            )

            conversion_failures = int(
                (
                    numeric_series.isna()
                    & series.notna()
                ).sum()
            )

            missing_values += null_count
            non_numeric_values += conversion_failures

            signal_missing_counts[
                f"{signal}_missing"
            ] = null_count

            signal_missing_counts[
                f"{signal}_non_numeric"
            ] = conversion_failures

        # ----------------------------------------------------
        # Duplicate analysis
        # ----------------------------------------------------

        duplicate_rows = int(
            df.duplicated().sum()
        )

        # ----------------------------------------------------
        # Range validation
        #
        # These are physical plausibility checks only.
        # They are NOT DTC thresholds.
        # ----------------------------------------------------

        range_violations = 0

        def numeric_series(signal):

            if signal not in df.columns:
                return None

            return pd.to_numeric(
                df[signal],
                errors="coerce"
            )

        # Engine RPM
        rpm = numeric_series("ENGINE_RPM")

        if rpm is not None:
            range_violations += int(
                (rpm < 0).sum()
            )

        # Vehicle speed
        speed = numeric_series("VEHICLE_SPEED")

        if speed is not None:
            range_violations += int(
                (speed < 0).sum()
            )

        # Engine load
        load = numeric_series("ENGINE_LOAD")

        if load is not None:
            range_violations += int(
                (
                    (load < 0)
                    | (load > 100)
                ).sum()
            )

        # Throttle
        throttle = numeric_series("THROTTLE")

        if throttle is not None:
            range_violations += int(
                (
                    (throttle < 0)
                    | (throttle > 100)
                ).sum()
            )

        # Coolant temperature
        coolant = numeric_series("COOLANT_TEMPERATURE")

        if coolant is not None:
            range_violations += int(
                (
                    (coolant < -80)
                    | (coolant > 200)
                ).sum()
            )

        # Control module voltage
        voltage = numeric_series("CONTROL_MODULE_VOLTAGE")

        if voltage is not None:
            range_violations += int(
                (
                    (voltage < 0)
                    | (voltage > 30)
                ).sum()
            )

        # ----------------------------------------------------
        # Dataset mode
        # ----------------------------------------------------

        filename = file_path.stem.lower()

        if filename.startswith("drive"):
            dataset_mode = "drive"
        elif filename.startswith("idle"):
            dataset_mode = "idle"
        elif filename.startswith("live"):
            dataset_mode = "live"
        elif filename.startswith("ufpe"):
            dataset_mode = "ufpe"
        elif filename.startswith("long"):
            dataset_mode = "long"
        else:
            dataset_mode = "other"

        # ----------------------------------------------------
        # Determine validation status
        # ----------------------------------------------------

        if rows == 0:

            status = "EMPTY"

        elif len(missing_columns) > 0:

            status = "MISSING_REQUIRED_COLUMNS"

        elif missing_values > 0:

            status = "HAS_MISSING_VALUES"

        elif non_numeric_values > 0:

            status = "HAS_NON_NUMERIC_VALUES"

        elif range_violations > 0:

            status = "RANGE_CHECK_REVIEW"

        else:

            status = "VALID"

        # ----------------------------------------------------
        # Build result
        # ----------------------------------------------------

        result = {
            "file": file_path.name,
            "dataset_mode": dataset_mode,
            "rows": rows,
            "columns": len(df.columns),

            "available_important_signals":
                len(available_signals),

            "missing_required_columns":
                ", ".join(missing_columns),

            "missing_values":
                missing_values,

            "non_numeric_values":
                non_numeric_values,

            "duplicate_rows":
                duplicate_rows,

            "range_violations":
                range_violations,

            "status":
                status,
        }

        # Add per-signal diagnostics
        result.update(signal_missing_counts)

        return result

    except Exception as e:

        return {
            "file": file_path.name,
            "dataset_mode": "unknown",
            "rows": 0,
            "columns": 0,
            "available_important_signals": 0,
            "missing_required_columns": "",
            "missing_values": 0,
            "non_numeric_values": 0,
            "duplicate_rows": 0,
            "range_violations": 0,
            "status": f"ERROR: {e}",
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FILE 05 - PREPROCESSING AND SIGNAL VALIDATION")
    print("=" * 70)

    if not DATA_DIR.exists():

        print("\nERROR: Data directory not found:")
        print(DATA_DIR.resolve())

        return

    csv_files = sorted(
        DATA_DIR.glob("*.csv")
    )

    print(
        f"\nInput directory: "
        f"{DATA_DIR.resolve()}"
    )

    print(
        f"CSV files found: "
        f"{len(csv_files)}"
    )

    if not csv_files:

        print("\nERROR: No CSV files found.")

        return

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    reports = []

    # --------------------------------------------------------
    # Analyze each recording
    # --------------------------------------------------------

    for index, file_path in enumerate(
        csv_files,
        start=1
    ):

        print(
            f"Processing "
            f"{index}/{len(csv_files)}: "
            f"{file_path.name}"
        )

        reports.append(
            analyze_file(file_path)
        )

    report_df = pd.DataFrame(
        reports
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_df.to_csv(
        QUALITY_REPORT,
        index=False
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    total_files = len(report_df)

    total_rows = int(
        report_df["rows"].sum()
    )

    total_missing = int(
        report_df["missing_values"].sum()
    )

    total_non_numeric = int(
        report_df["non_numeric_values"].sum()
    )

    total_duplicates = int(
        report_df["duplicate_rows"].sum()
    )

    total_range_violations = int(
        report_df["range_violations"].sum()
    )

    files_with_missing = int(
        (
            report_df["missing_values"] > 0
        ).sum()
    )

    files_with_non_numeric = int(
        (
            report_df["non_numeric_values"] > 0
        ).sum()
    )

    files_with_range_issues = int(
        (
            report_df["range_violations"] > 0
        ).sum()
    )

    files_with_errors = int(
        report_df[
            "status"
        ].astype(str)
        .str.startswith("ERROR")
        .sum()
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Total files:                     "
        f"{total_files}"
    )

    print(
        f"Total rows:                      "
        f"{total_rows:,}"
    )

    print(
        f"Files with missing values:       "
        f"{files_with_missing}"
    )

    print(
        f"Files with non-numeric values:   "
        f"{files_with_non_numeric}"
    )

    print(
        f"Files with range issues:         "
        f"{files_with_range_issues}"
    )

    print(
        f"Files with processing errors:    "
        f"{files_with_errors}"
    )

    print(
        f"\nTotal genuine missing values:    "
        f"{total_missing:,}"
    )

    print(
        f"Total non-numeric values:        "
        f"{total_non_numeric:,}"
    )

    print(
        f"Exact duplicate rows:             "
        f"{total_duplicates:,}"
    )

    print(
        f"Range violations:                 "
        f"{total_range_violations:,}"
    )

    # --------------------------------------------------------
    # Status distribution
    # --------------------------------------------------------

    print("\nStatus distribution:")

    print(
        report_df[
            "status"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Files requiring attention
    # --------------------------------------------------------

    attention = report_df[
        (
            (report_df["missing_values"] > 0)
            |
            (report_df["non_numeric_values"] > 0)
            |
            (report_df["range_violations"] > 0)
            |
            (
                report_df["missing_required_columns"]
                .astype(str)
                .str.len()
                > 0
            )
        )
    ]

    if len(attention) > 0:

        print(
            "\nFiles requiring attention:"
        )

        print(
            attention[
                [
                    "file",
                    "rows",
                    "missing_values",
                    "non_numeric_values",
                    "duplicate_rows",
                    "range_violations",
                    "status",
                ]
            ]
            .sort_values(
                "file"
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save summary text
    # --------------------------------------------------------

    summary_lines = [

        "FILE 05 - PREPROCESSING AND SIGNAL VALIDATION",

        "=" * 70,

        f"Total files: {total_files}",

        f"Total rows: {total_rows:,}",

        f"Files with missing values: "
        f"{files_with_missing}",

        f"Files with non-numeric values: "
        f"{files_with_non_numeric}",

        f"Files with range issues: "
        f"{files_with_range_issues}",

        f"Files with processing errors: "
        f"{files_with_errors}",

        f"Total genuine missing values: "
        f"{total_missing:,}",

        f"Total non-numeric values: "
        f"{total_non_numeric:,}",

        f"Exact duplicate rows: "
        f"{total_duplicates:,}",

        f"Range violations: "
        f"{total_range_violations:,}",

        "",

        "Validation policy:",

        "Raw CSV files were not modified.",

        "No rows were deleted.",

        "No values were imputed.",

        "Exact duplicate rows were retained.",

        "Range violations were flagged rather than automatically removed.",

        "Only required numeric signals were converted for validation.",

    ]

    SUMMARY_FILE.write_text(
        "\n".join(summary_lines),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        QUALITY_REPORT.resolve()
    )

    print(
        SUMMARY_FILE.resolve()
    )

    print("\nANALYSIS COMPLETE")


if __name__ == "__main__":
    main()