from pathlib import Path
import pandas as pd
import re

DATA_DIR = Path("data/raw/obdiidata")

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 70)
print("carOBD DATASET PROFILE")
print("=" * 70)

print(f"\nTotal CSV files: {len(csv_files)}")

# ---------------------------------------------------------
# 1. Identify recording type
# ---------------------------------------------------------

file_groups = {}

for file in csv_files:
    match = re.match(r"([a-zA-Z]+)(\d+)\.csv$", file.name)

    if match:
        group = match.group(1).lower()
        file_groups.setdefault(group, []).append(file)

print("\nRecording groups:")

for group, files in sorted(file_groups.items()):
    print(f"  {group:<10} {len(files):>3} files")


# ---------------------------------------------------------
# 2. File-level statistics
# ---------------------------------------------------------

summary = []

for file in csv_files:

    df = pd.read_csv(file)

    summary.append({
        "file": file.name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    })

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)

print(f"Total observations: {summary_df['rows'].sum():,}")
print(f"Minimum rows/file: {summary_df['rows'].min():,}")
print(f"Maximum rows/file: {summary_df['rows'].max():,}")
print(f"Average rows/file: {summary_df['rows'].mean():,.1f}")

print("\n" + "=" * 70)
print("FILE STATISTICS")
print("=" * 70)

print(summary_df.describe().to_string())


# ---------------------------------------------------------
# 3. Column consistency
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("COLUMN CONSISTENCY")
print("=" * 70)

column_sets = {}

for file in csv_files:

    df = pd.read_csv(file)

    columns = tuple(df.columns)

    column_sets.setdefault(columns, []).append(file.name)

print(f"\nUnique column structures: {len(column_sets)}")

for i, (columns, files) in enumerate(column_sets.items(), start=1):

    print(f"\nStructure {i}:")
    print(f"Files using this structure: {len(files)}")
    print(f"Number of columns: {len(columns)}")

    for column in columns:
        print(f"  - {column}")


# ---------------------------------------------------------
# 4. Missing values
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print(f"Total missing values across all files: "
      f"{summary_df['missing_values'].sum():,}")

print(f"Total duplicate rows across all files: "
      f"{summary_df['duplicate_rows'].sum():,}")


# ---------------------------------------------------------
# 5. Representative file
# ---------------------------------------------------------

if csv_files:

    sample_file = csv_files[0]
    sample_df = pd.read_csv(sample_file)

    print("\n" + "=" * 70)
    print("REPRESENTATIVE FILE")
    print("=" * 70)

    print(f"\nFile: {sample_file.name}")
    print(f"Rows: {len(sample_df):,}")
    print(f"Columns: {len(sample_df.columns)}")

    print("\nFirst 5 rows:")
    print(sample_df.head().to_string(index=False))


print("\n" + "=" * 70)
print("PROFILE COMPLETE")
print("=" * 70)