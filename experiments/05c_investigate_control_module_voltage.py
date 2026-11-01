from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw/obdiidata")
RESULTS_DIR = Path("results")

OUTPUT_FILE = RESULTS_DIR / "05c_control_module_voltage_analysis.csv"
SUMMARY_FILE = RESULTS_DIR / "05c_control_module_voltage_summary.txt"


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(column):

    import re

    name = str(column).strip().upper()

    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"[^A-Z0-9]+", "_", name)

    return name.strip("_")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FILE 05C - CONTROL MODULE VOLTAGE INVESTIGATION")
    print("=" * 70)

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    print(f"\nFiles found: {len(csv_files)}")

    all_values = []
    file_results = []

    # --------------------------------------------------------
    # Read every recording
    # --------------------------------------------------------

    for index, file_path in enumerate(csv_files, start=1):

        print(
            f"Processing {index}/{len(csv_files)}: "
            f"{file_path.name}"
        )

        df = pd.read_csv(file_path)

        rename_map = {
            column: normalize_column_name(column)
            for column in df.columns
        }

        df = df.rename(columns=rename_map)

        if "CONTROL_MODULE_VOLTAGE" not in df.columns:
            continue

        series = pd.to_numeric(
            df["CONTROL_MODULE_VOLTAGE"],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        # ----------------------------------------------------
        # Collect values
        # ----------------------------------------------------

        all_values.extend(series.tolist())

        # ----------------------------------------------------
        # Classify values
        #
        # These are investigation bands, NOT diagnostic
        # thresholds.
        # ----------------------------------------------------

        normal_12v = (
            (series >= 10.0)
            & (series <= 16.0)
        )

        low_voltage = series < 10.0

        above_16 = series > 16.0

        above_20 = series > 20.0

        above_30 = series > 30.0

        above_50 = series > 50.0

        above_standard = series > 65.535

        # ----------------------------------------------------
        # Per-file result
        # ----------------------------------------------------

        file_results.append({

            "file": file_path.name,

            "rows": len(series),

            "minimum": series.min(),

            "maximum": series.max(),

            "mean": series.mean(),

            "median": series.median(),

            "p01": series.quantile(0.01),

            "p05": series.quantile(0.05),

            "p25": series.quantile(0.25),

            "p75": series.quantile(0.75),

            "p95": series.quantile(0.95),

            "p99": series.quantile(0.99),

            "values_10_to_16V":
                int(normal_12v.sum()),

            "values_below_10V":
                int(low_voltage.sum()),

            "values_above_16V":
                int(above_16.sum()),

            "values_above_20V":
                int(above_20.sum()),

            "values_above_30V":
                int(above_30.sum()),

            "values_above_50V":
                int(above_50.sum()),

            "values_above_65_535V":
                int(above_standard.sum()),

        })

    # --------------------------------------------------------
    # Save per-file analysis
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = pd.DataFrame(file_results)

    report.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Overall analysis
    # --------------------------------------------------------

    values = pd.Series(all_values)

    print("\n" + "=" * 70)
    print("OVERALL DISTRIBUTION")
    print("=" * 70)

    if values.empty:

        print("\nNo CONTROL_MODULE_VOLTAGE values found.")

        return

    normal_12v = (
        (values >= 10.0)
        & (values <= 16.0)
    )

    low_voltage = values < 10.0
    above_16 = values > 16.0
    above_20 = values > 20.0
    above_30 = values > 30.0
    above_50 = values > 50.0
    above_standard = values > 65.535

    total = len(values)

    print(f"\nTotal observations: {total:,}")

    print(f"Minimum: {values.min():.6f}")
    print(f"Maximum: {values.max():.6f}")
    print(f"Mean: {values.mean():.6f}")
    print(f"Median: {values.median():.6f}")

    print("\nPercentiles:")

    for percentile in [
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99
    ]:

        print(
            f"P{percentile * 100:05.1f}: "
            f"{values.quantile(percentile):.6f}"
        )

    print("\nValue bands:")

    print(
        f"10–16 V: "
        f"{normal_12v.sum():,} "
        f"({normal_12v.mean() * 100:.2f}%)"
    )

    print(
        f"<10 V: "
        f"{low_voltage.sum():,} "
        f"({low_voltage.mean() * 100:.2f}%)"
    )

    print(
        f">16 V: "
        f"{above_16.sum():,} "
        f"({above_16.mean() * 100:.2f}%)"
    )

    print(
        f">20 V: "
        f"{above_20.sum():,} "
        f"({above_20.mean() * 100:.2f}%)"
    )

    print(
        f">30 V: "
        f"{above_30.sum():,} "
        f"({above_30.mean() * 100:.2f}%)"
    )

    print(
        f">50 V: "
        f"{above_50.sum():,} "
        f"({above_50.mean() * 100:.2f}%)"
    )

    print(
        f">65.535 V: "
        f"{above_standard.sum():,} "
        f"({above_standard.mean() * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Most common values
    # --------------------------------------------------------

    print("\nMost frequent values:")

    rounded = values.round(3)

    print(
        rounded
        .value_counts()
        .head(30)
        .to_string()
    )

    # --------------------------------------------------------
    # Summary file
    # --------------------------------------------------------

    summary = []

    summary.append(
        "FILE 05C - CONTROL MODULE VOLTAGE INVESTIGATION"
    )

    summary.append("=" * 70)

    summary.append(
        f"Total observations: {total:,}"
    )

    summary.append(
        f"Minimum: {values.min():.6f}"
    )

    summary.append(
        f"Maximum: {values.max():.6f}"
    )

    summary.append(
        f"Mean: {values.mean():.6f}"
    )

    summary.append(
        f"Median: {values.median():.6f}"
    )

    summary.append("")

    summary.append("Percentiles:")

    for percentile in [
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99
    ]:

        summary.append(
            f"P{percentile * 100:05.1f}: "
            f"{values.quantile(percentile):.6f}"
        )

    summary.append("")

    summary.append("Value bands:")

    summary.append(
        f"10–16 V: {normal_12v.sum():,} "
        f"({normal_12v.mean() * 100:.2f}%)"
    )

    summary.append(
        f"<10 V: {low_voltage.sum():,} "
        f"({low_voltage.mean() * 100:.2f}%)"
    )

    summary.append(
        f">16 V: {above_16.sum():,} "
        f"({above_16.mean() * 100:.2f}%)"
    )

    summary.append(
        f">20 V: {above_20.sum():,} "
        f"({above_20.mean() * 100:.2f}%)"
    )

    summary.append(
        f">30 V: {above_30.sum():,} "
        f"({above_30.mean() * 100:.2f}%)"
    )

    summary.append(
        f">50 V: {above_50.sum():,} "
        f"({above_50.mean() * 100:.2f}%)"
    )

    summary.append(
        f">65.535 V: {above_standard.sum():,} "
        f"({above_standard.mean() * 100:.2f}%)"
    )

    summary.append("")

    summary.append(
        "Interpretation note:"
    )

    summary.append(
        "PID 42 is standardized as control module voltage "
        "in volts using (256*A+B)/1000."
    )

    summary.append(
        "The investigation bands above are descriptive only "
        "and are not diagnostic thresholds."
    )

    SUMMARY_FILE.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        OUTPUT_FILE.resolve()
    )

    print(
        SUMMARY_FILE.resolve()
    )

    print("\nANALYSIS COMPLETE")


if __name__ == "__main__":
    main()