from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw/obdiidata")
RESULTS_DIR = Path("results")

OUTPUT_FILE = RESULTS_DIR / "04_experimental_dataset_manifest.csv"


# Minimum requirements for an experimental file
MIN_ROWS = 100

REQUIRED_COLUMNS = [
    "ENGINE_RUN_TIME",
    "ENGINE_RPM",
    "VEHICLE_SPEED",
    "THROTTLE",
    "ENGINE_LOAD",
    "COOLANT_TEMPERATURE",
    "LONG_TERM_FUEL_TRIM_BANK_1",
    "SHORT_TERM_FUEL_TRIM_BANK_1",
    "INTAKE_MANIFOLD_PRESSURE",
]


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(column):
    """
    Normalize a column name so that formatting differences
    and known dataset naming issues do not affect matching.
    """
    name = str(column).strip().upper()

    # Remove anything inside parentheses
    import re
    name = re.sub(r"\([^)]*\)", "", name)

    # Replace non-alphanumeric characters with underscores
    name = re.sub(r"[^A-Z0-9]+", "_", name)

    name = name.strip("_")

    # Dataset-specific correction
    if name == "ENGINE_RUN_TINE":
        name = "ENGINE_RUN_TIME"

    return name

# ============================================================
# FILE ANALYSIS
# ============================================================

def analyze_file(file_path):

    try:
        df = pd.read_csv(file_path)

        original_columns = list(df.columns)

        normalized_columns = {
            normalize_column_name(col): col
            for col in df.columns
        }

        missing_columns = [
            col
            for col in REQUIRED_COLUMNS
            if normalize_column_name(col) not in normalized_columns
        ]

        rows = len(df)

        # Count exact duplicate rows
        duplicate_rows = int(df.duplicated().sum())

        if rows > 0:
            duplicate_percent = (duplicate_rows / rows) * 100
        else:
            duplicate_percent = 100.0

        # Determine file type from filename
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

        # Basic eligibility
        eligible = (
            rows >= MIN_ROWS
            and len(missing_columns) == 0
        )

        if eligible:
            reason = "Eligible"
        elif rows < MIN_ROWS:
            reason = "Insufficient rows"
        else:
            reason = "Missing required columns"

        return {
            "file": file_path.name,
            "dataset_mode": dataset_mode,
            "rows": rows,
            "duplicate_rows": duplicate_rows,
            "duplicate_percent": round(duplicate_percent, 2),
            "missing_required_columns": ", ".join(missing_columns),
            "eligible": eligible,
            "selection_reason": reason,
        }

    except Exception as e:

        return {
            "file": file_path.name,
            "dataset_mode": "unknown",
            "rows": 0,
            "duplicate_rows": 0,
            "duplicate_percent": 100.0,
            "missing_required_columns": "",
            "eligible": False,
            "selection_reason": f"Read error: {e}",
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENTAL DATASET SELECTION")
    print("=" * 70)

    print(f"\nData directory: {DATA_DIR.resolve()}")

    if not DATA_DIR.exists():
        print("\nERROR: Data directory does not exist.")
        return

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    print(f"CSV files found: {len(csv_files)}")

    if not csv_files:
        print("\nERROR: No CSV files found.")
        return

    # Analyze every file
    results = []

    for file_path in csv_files:
        result = analyze_file(file_path)
        results.append(result)

    manifest = pd.DataFrame(results)

    # Sort by dataset mode and filename
    manifest = manifest.sort_values(
        by=["dataset_mode", "file"]
    ).reset_index(drop=True)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save complete manifest
    manifest.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    eligible = manifest[manifest["eligible"] == True]

    print("\n" + "=" * 70)
    print("SELECTION SUMMARY")
    print("=" * 70)

    print(f"Total CSV files:        {len(manifest)}")
    print(f"Eligible files:         {len(eligible)}")
    print(f"Rejected files:         {len(manifest) - len(eligible)}")

    print("\nEligible files by dataset mode:")

    print(
        eligible["dataset_mode"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nRows by dataset mode:")

    rows_by_mode = (
        eligible
        .groupby("dataset_mode")["rows"]
        .agg(["count", "sum", "mean"])
        .round(2)
    )

    print(rows_by_mode.to_string())

    print("\nDuplicate percentage among eligible files:")

    print(
        eligible["duplicate_percent"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nTop files by row count:")

    print(
        eligible[
            ["file", "dataset_mode", "rows", "duplicate_percent"]
        ]
        .sort_values("rows", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("RESULT SAVED")
    print("=" * 70)

    print(OUTPUT_FILE.resolve())

    print("\nANALYSIS COMPLETE")


if __name__ == "__main__":
    main()