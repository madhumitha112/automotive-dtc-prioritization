# Dataset Validation

Run this before building the DTC prioritization algorithm:

```bash
python experiments/01_validate_dataset.py
```

The script reads only from `data/raw/` and writes:

`results/dataset_validation_report.json`

It does not modify the raw data.
