from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/obdiidata")

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 70)
print("carOBD DATA QUALITY ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# 1. Missing values by column
# ---------------------------------------------------------

all_missing = {}

for file in csv_files:
    df = pd.read_csv(file)

    for column in df.columns:
        all_missing[column] = (
            all_missing.get(column, 0)
            + int(df[column].isna().sum())
        )

missing_df = pd.DataFrame(
    list(all_missing.items()),
    columns=["column", "missing_values"]
)

missing_df["missing_percent"] = (
    missing_df["missing_values"]
    / sum(len(pd.read_csv(f)) for f in csv_files)
    * 100
)

missing_df = missing_df.sort_values(
    "missing_values",
    ascending=False
)

print("\n" + "=" * 70)
print("MISSING VALUES BY COLUMN")
print("=" * 70)

print(missing_df.to_string(index=False))


# ---------------------------------------------------------
# 2. Duplicate rows by recording group
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE ROWS BY FILE")
print("=" * 70)

duplicate_results = []

for file in csv_files:

    df = pd.read_csv(file)

    duplicate_results.append({
        "file": file.name,
        "rows": len(df),
        "duplicates": int(df.duplicated().sum()),
        "duplicate_percent":
            round(df.duplicated().mean() * 100, 2)
    })

duplicates_df = pd.DataFrame(duplicate_results)

print(
    duplicates_df
    .sort_values("duplicates", ascending=False)
    .to_string(index=False)
)


# ---------------------------------------------------------
# 3. Overall numerical ranges
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("REPRESENTATIVE VALUE RANGES")
print("=" * 70)

# Use the first file for an initial view
sample_file = csv_files[0]
sample_df = pd.read_csv(sample_file)

for column in sample_df.columns:

    series = sample_df[column]

    print(
        f"{column:<45} "
        f"min={series.min():>12.3f} "
        f"max={series.max():>12.3f}"
    )


print("\n" + "=" * 70)
print("DATA QUALITY ANALYSIS COMPLETE")
print("=" * 70)