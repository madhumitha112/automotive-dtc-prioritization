
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGET_FILE = ROOT / "experiments" / "04c_generate_expanded_benchmark.py"

REPLACEMENTS = {
    "data/expanded_v4/raw/can_diagnostic_dataset_v4.csv":
        "data/raw/can_diagnostic_dataset_v4.csv",

    "data/expanded_v4/raw/dataset_manifest_v4.json":
        "data/raw/dataset_manifest_v4.json",

    "data/expanded_v4/raw/diagnostic_events_v4.csv":
        "data/raw/diagnostic_events_v4.csv",

    "results/expanded_v4/diagnostic_cases_v4.csv":
        "results/diagnostic_cases_v4.csv",

    "results/expanded_v4/scenario_manifest_v4.csv":
        "results/scenario_manifest_v4.csv",

    "results/expanded_v4/04c_dtc_coverage_summary.csv":
        "results/04c_dtc_coverage_summary.csv",
}


def main():
    if not TARGET_FILE.exists():
        raise FileNotFoundError(TARGET_FILE)

    original = TARGET_FILE.read_text(encoding="utf-8")
    updated = original

    for old, new in REPLACEMENTS.items():
        updated = updated.replace(old, new)

    if updated == original:
        print("No changes made.")
        print("The old paths may already have been updated,")
        print("or the script uses a different path format.")
        return

    TARGET_FILE.write_text(updated, encoding="utf-8")

    print("V4 path migration completed.")
    print(f"Updated file: {TARGET_FILE}")

    print("\nPlease review the changes before committing.")
    print("Do NOT execute the V4 generator.")


if __name__ == "__main__":
    main()
