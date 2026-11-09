
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGET_FILE = ROOT / "experiments" / "04c_generate_expanded_benchmark.py"

OLD_PATHS = [
    'RAW_DIR = ROOT / "data" / "expanded_v4" / "raw"',
    'RESULTS_DIR = ROOT / "results" / "expanded_v4"',
]

NEW_PATHS = [
    'RAW_DIR = ROOT / "data" / "raw"',
    'RESULTS_DIR = ROOT / "results"',
]


def main():
    if not TARGET_FILE.exists():
        raise FileNotFoundError(TARGET_FILE)

    original = TARGET_FILE.read_text(encoding="utf-8")

    for old in OLD_PATHS:
        if original.count(old) != 1:
            raise ValueError(
                f"Expected exactly one occurrence of:\n{old}\n"
                f"Found: {original.count(old)}"
            )

    updated = original

    for old, new in zip(OLD_PATHS, NEW_PATHS):
        updated = updated.replace(old, new, 1)

    if updated == original:
        raise RuntimeError("No changes were made.")

    TARGET_FILE.write_text(updated, encoding="utf-8")

    print("V4 generator paths updated successfully.")
    print("\nChanges made:")
    for old, new in zip(OLD_PATHS, NEW_PATHS):
        print(f"\nOLD: {old}")
        print(f"NEW: {new}")


if __name__ == "__main__":
    main()
