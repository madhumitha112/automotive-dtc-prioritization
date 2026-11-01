from pathlib import Path
import pandas as pd
import re


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw/obdiidata")
RESULTS_DIR = Path("results")

OUTPUT_FILE = RESULTS_DIR / "03_duplicate_structure_results.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column_name(column_name):
    """
    Normalize a column name for reliable matching.

    Examples:
        ENGINE_RUN_TIME ()
        ENGINE_RUN_TIME
        engine_run_time
        ENGINE_RUN_TIME (s)

    All become:
        ENGINE_RUN_TIME
    """

    name = str(column_name).strip().upper()

    # Remove anything inside parentheses, including ()
    name = re.sub(r"\([^)]*\)", "", name)

    # Replace non-alphanumeric characters with _
    name = re.sub(r"[^A-Z0-9]+", "_", name)

    # Remove leading/trailing underscores
    name = name.strip("_")

    return name


def find_engine_run_time_column(columns):
    """Find ENGINE_RUN_TIME regardless of formatting/units."""

    for column in columns:
        name = str(column).strip().upper()

        # Remove anything inside parentheses
        name = re.sub(r"\([^)]*\)", "", name).strip()

        # Normalize spaces, hyphens, etc.
        name = re.sub(r"[^A-Z0-9]+", "_", name).strip("_")

        if name in ("ENGINE_RUN_TIME", "ENGINE_RUN_TINE"):
            return column

    return None


def calculate_consecutive_duplicates(df):
    """
    Count rows that are identical to the immediately
    preceding row.

    This is different from total/exact duplicates.
    """

    if len(df) <= 1:
        return 0

    duplicate_mask = df.eq(df.shift(1)).all(axis=1)

    return int(duplicate_mask.sum())


def analyze_engine_run_time(df, column):
    """
    Analyze ENGINE_RUN_TIME for:

    - minimum
    - maximum
    - number of decreasing points
    - number of non-decreasing points
    """

    if column is None:
        return {
            "engine_time_column": None,
            "time_non_decreasing": None,
            "time_decreasing_points": None,
            "engine_time_min": None,
            "engine_time_max": None,
        }

    values = pd.to_numeric(df[column], errors="coerce")

    # Remove missing values
    values = values.dropna()

    if len(values) == 0:
        return {
            "engine_time_column": column,
            "time_non_decreasing": None,
            "time_decreasing_points": None,
            "engine_time_min": None,
            "engine_time_max": None,
        }

    differences = values.diff().dropna()

    decreasing_points = int((differences < 0).sum())

    return {
        "engine_time_column": column,
        "time_non_decreasing": decreasing_points == 0,
        "time_decreasing_points": decreasing_points,
        "engine_time_min": float(values.min()),
        "engine_time_max": float(values.max()),
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"No CSV files found in:")
        print(DATA_DIR.resolve())
        print()
        print("Check that your dataset is located at:")
        print("data/raw/obdidata/")
        return

    print("=" * 70)
    print("DUPLICATE & DATA STRUCTURE ANALYSIS")
    print("=" * 70)

    print()
    print(f"Data directory : {DATA_DIR.resolve()}")
    print(f"CSV files found: {len(files)}")
    print()

    results = []

    total_rows = 0
    total_exact_duplicates = 0
    total_consecutive_duplicates = 0

    files_with_duplicates = 0
    files_with_non_decreasing_engine_time = 0
    files_with_decreasing_engine_time = 0

    all_columns = None

    # ========================================================
    # FILE-BY-FILE ANALYSIS
    # ========================================================

    for file_path in files:

        try:
            df = pd.read_csv(file_path)

        except Exception as exc:
            print(f"ERROR reading {file_path.name}: {exc}")
            continue

        rows = len(df)

        total_rows += rows

        # ----------------------------------------------------
        # Column structure
        # ----------------------------------------------------

        if all_columns is None:
            all_columns = list(df.columns)

            print("=" * 70)
            print(f"FILE: {file_path.name}")
            print("=" * 70)
            print()
            print("COLUMNS DETECTED:")

            for i, column in enumerate(df.columns, start=1):
                print(f"{i:2}. {column!r}")

            print()

        # ----------------------------------------------------
        # Exact duplicates
        # ----------------------------------------------------

        exact_duplicates = int(df.duplicated(keep="first").sum())

        if exact_duplicates > 0:
            files_with_duplicates += 1

        total_exact_duplicates += exact_duplicates

        # ----------------------------------------------------
        # Consecutive duplicates
        # ----------------------------------------------------

        consecutive_duplicates = calculate_consecutive_duplicates(df)

        total_consecutive_duplicates += consecutive_duplicates

        # ----------------------------------------------------
        # ENGINE_RUN_TIME detection
        # ----------------------------------------------------

        engine_time_column = find_engine_run_time_column(df.columns)

        engine_time_analysis = analyze_engine_run_time(
            df,
            engine_time_column
        )

        if engine_time_column is not None:

            if engine_time_analysis["time_decreasing_points"] == 0:
                files_with_non_decreasing_engine_time += 1
            else:
                files_with_decreasing_engine_time += 1

        # ----------------------------------------------------
        # Duplicate percentage
        # ----------------------------------------------------

        if rows > 0:
            duplicate_percent = (
                exact_duplicates / rows
            ) * 100
        else:
            duplicate_percent = 0.0

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "file": file_path.name,
            "rows": rows,
            "duplicates": exact_duplicates,
            "duplicate_percent": round(duplicate_percent, 2),
            "consecutive_duplicates": consecutive_duplicates,
            **engine_time_analysis,
        }

        results.append(result)

        # ----------------------------------------------------
        # Print file result
        # ----------------------------------------------------

        print(
            f"{file_path.name:<18}"
            f"{rows:>8}"
            f"{exact_duplicates:>10}"
            f"{duplicate_percent:>12.2f}%"
            f"{consecutive_duplicates:>12}"
        )

    # ========================================================
    # CREATE RESULTS DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(results)

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    print(f"Total files: {len(result_df):,}")
    print(f"Total rows: {total_rows:,}")
    print(f"Files with duplicates: {files_with_duplicates:,}")
    print(f"Total exact duplicates: {total_exact_duplicates:,}")
    print(
        f"Total consecutive duplicates: "
        f"{total_consecutive_duplicates:,}"
    )

    print(
        f"Files with non-decreasing ENGINE_RUN_TIME: "
        f"{files_with_non_decreasing_engine_time:,}"
    )

    print(
        f"Files with decreasing ENGINE_RUN_TIME: "
        f"{files_with_decreasing_engine_time:,}"
    )

    # ========================================================
    # ENGINE_RUN_TIME DETECTION SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("ENGINE_RUN_TIME COLUMN DETECTION")
    print("=" * 70)

    detected_engine_columns = (
        result_df["engine_time_column"]
        .dropna()
        .unique()
        .tolist()
    )

    if detected_engine_columns:

        print("ENGINE_RUN_TIME detected successfully.")

        print()
        print("Detected column names:")

        for column in detected_engine_columns:
            print(f"  - {column}")

    else:

        print("WARNING:")
        print("ENGINE_RUN_TIME column was not detected in any file.")

    # ========================================================
    # FILES WITH DECREASING ENGINE_RUN_TIME
    # ========================================================

    print()
    print("=" * 70)
    print("FILES WITH DECREASING ENGINE_RUN_TIME")
    print("=" * 70)

    if "time_decreasing_points" in result_df.columns:

        decreasing_df = result_df[
            result_df["time_decreasing_points"].fillna(0) > 0
        ].copy()

        if len(decreasing_df) == 0:

            print("None found.")

        else:

            print(
                decreasing_df[
                    [
                        "file",
                        "rows",
                        "time_decreasing_points",
                        "engine_time_min",
                        "engine_time_max",
                    ]
                ].to_string(index=False)
            )

    # ========================================================
    # TOP FILES BY DUPLICATE PERCENTAGE
    # ========================================================

    print()
    print("=" * 70)
    print("TOP FILES BY DUPLICATE PERCENTAGE")
    print("=" * 70)

    top_duplicates = (
        result_df
        .sort_values(
            by="duplicate_percent",
            ascending=False
        )
        .head(20)
    )

    columns_to_display = [
        "file",
        "rows",
        "duplicates",
        "duplicate_percent",
        "consecutive_duplicates",
        "engine_time_column",
        "time_non_decreasing",
        "time_decreasing_points",
        "engine_time_min",
        "engine_time_max",
    ]

    # Only display columns that exist
    columns_to_display = [
        column
        for column in columns_to_display
        if column in top_duplicates.columns
    ]

    print(
        top_duplicates[
            columns_to_display
        ].to_string(index=False)
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("RESULT SAVED")
    print("=" * 70)

    print(OUTPUT_FILE.resolve())

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()