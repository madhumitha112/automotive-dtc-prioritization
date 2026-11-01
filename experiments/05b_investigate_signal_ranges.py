from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw/obdiidata")
RESULTS_DIR = Path("results")

OUTPUT_FILE = RESULTS_DIR / "05b_range_investigation.csv"


# ============================================================
# SIGNAL DEFINITIONS
# ============================================================

RANGE_RULES = {
    "ENGINE_RPM": (-1, None),
    "VEHICLE_SPEED": (-1, None),
    "ENGINE_LOAD": (0, 100),
    "THROTTLE": (0, 100),
    "COOLANT_TEMPERATURE": (-80, 200),
    "CONTROL_MODULE_VOLTAGE": (0, 30),
}


COLUMN_ALIASES = {
    "ENGINE_RUN_TINE": "ENGINE_RUN_TIME",
    "FUEL_TANK": "FUEL_TANK_LEVEL_INPUT",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_column_name(column):

    import re

    name = str(column).strip().upper()

    name = re.sub(r"\([^)]*\)", "", name)

    name = re.sub(r"[^A-Z0-9]+", "_", name)

    name = name.strip("_")

    if name in COLUMN_ALIASES:
        name = COLUMN_ALIASES[name]

    return name


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("FILE 05B - SIGNAL RANGE INVESTIGATION")
    print("=" * 70)

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    print(f"\nFiles found: {len(csv_files)}")

    results = []

    for file_path in csv_files:

        print(f"Processing: {file_path.name}")

        df = pd.read_csv(file_path)

        df = df.rename(
            columns={
                column: normalize_column_name(column)
                for column in df.columns
            }
        )

        for signal, (lower, upper) in RANGE_RULES.items():

            if signal not in df.columns:
                continue

            series = pd.to_numeric(
                df[signal],
                errors="coerce"
            )

            if lower is not None:
                lower_mask = series < lower
            else:
                lower_mask = pd.Series(
                    False,
                    index=series.index
                )

            if upper is not None:
                upper_mask = series > upper
            else:
                upper_mask = pd.Series(
                    False,
                    index=series.index
                )

            invalid_mask = lower_mask | upper_mask

            invalid_count = int(
                invalid_mask.sum()
            )

            if invalid_count == 0:
                continue

            invalid_values = series[
                invalid_mask
            ]

            results.append({
                "file": file_path.name,
                "signal": signal,
                "invalid_count": invalid_count,
                "minimum_invalid_value":
                    invalid_values.min(),
                "maximum_invalid_value":
                    invalid_values.max(),
                "overall_min":
                    series.min(),
                "overall_max":
                    series.max(),
                "allowed_lower":
                    lower,
                "allowed_upper":
                    upper,
            })

    report = pd.DataFrame(results)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("RANGE INVESTIGATION SUMMARY")
    print("=" * 70)

    if report.empty:

        print("\nNo range violations found.")

    else:

        print(
            f"\nSignal-level violations: "
            f"{len(report)}"
        )

        print(
            f"Total flagged observations: "
            f"{report['invalid_count'].sum():,}"
        )

        print("\nViolations by signal:")

        signal_summary = (
            report
            .groupby("signal")["invalid_count"]
            .sum()
            .sort_values(ascending=False)
        )

        print(
            signal_summary.to_string()
        )

        print("\nDetailed results:")

        print(
            report.to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)
    print("RESULT SAVED")
    print("=" * 70)

    print(
        OUTPUT_FILE.resolve()
    )

    print("\nANALYSIS COMPLETE")


if __name__ == "__main__":
    main()