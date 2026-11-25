# Automotive DTC Prioritization

Comparative evaluation of static and lightweight CAN-context-based prioritization of automotive Diagnostic Trouble Codes (DTCs).

## Research objective

This repository evaluates whether selected CAN-context evidence can improve the ranking of candidate DTCs compared with a fixed static-priority baseline.

The methods rank a **provided candidate DTC set**. They do not detect faults, generate DTC candidates, or replace an OEM diagnostic system.

The term *lightweight* describes the transparent, rule-based context refinement used in this study. The experiments do not measure ECU execution time, memory use, or production real-time suitability.

## Dataset and research integrity

The V3 and V4 benchmarks in this repository are **controlled synthetic datasets**, not recordings collected from physical vehicles.

- **V3:** 48 diagnostic cases. Dataset details are recorded in [the V3 manifest](data/raw/dataset_manifest.json).
- **V4:** 24 cases based on 12 scenario templates, with two repetitions per template. See [the V4 manifest](data/raw/dataset_manifest_v4.json).
- The primary-DTC labels and scenario/context metadata are evaluation information. They must not be used as ranking inputs.
- V4 repetitions are not independent scenario families. The benchmark represents only a subset of possible DTC candidate pairs.
- Some CAN-context conditions are signal proxies; for example, the V4 P0335 scenario uses indirect RPM-instability evidence, not a simulated CKP waveform.

Results should therefore be interpreted as performance **within the evaluated synthetic benchmarks**, not as real-vehicle validation or proof of general diagnostic performance.

## Repository layout

| Path | Contents |
|---|---|
| `data/raw/` | Raw benchmark CAN frames, diagnostic events, and dataset manifests |
| `results/diagnostic_cases.csv` | Extracted V3 diagnostic cases |
| `results/diagnostic_cases_v4.csv` | V4 benchmark cases |
| `experiments/` | Validation, extraction, baseline, context, audit, and comparison scripts |
| `results/` | Experiment outputs, metrics, reports, and audit artifacts |
| `docs/experimental_conclusions_and_limitations.md` | Detailed experimental findings and limitations |

## Environment setup

Use Python and install the project's pinned environment requirements:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The experiment scripts use pandas for CSV processing. The repository's `requirements.txt` is a pinned environment file and includes additional notebook and analysis packages.

Run commands from the repository root.

## Experiment workflow

The repository contains frozen baseline/context experiments, validation and audit scripts, and comparative evaluations. Existing committed result files are retained as experiment artifacts.

### Dataset validation and case extraction

```bash
python experiments/01_validate_dataset.py
python experiments/02_extract_diagnostic_cases.py
```

The validator checks the V1 raw CAN and diagnostic-event files and writes `results/dataset_validation_report.json`. Case extraction writes `results/diagnostic_cases.csv`.

V4 cases are stored in `results/diagnostic_cases_v4.csv` and are described by the V4 manifest.

### Frozen static and CAN-context experiments

The original V3/V4 static baselines and context-refinement scripts are maintained separately:

- `experiments/03_static_baseline.py`
- `experiments/03_static_baseline_v4.py`
- `experiments/04_context_refinement.py`
- `experiments/04_context_refinement_v4.py`

These frozen methods and their original outputs form the static-versus-context comparison. The context method ranks the same candidate set using its configured context evidence.

### Comparative evaluation and analysis

```bash
python experiments/05_v4_comparative_evaluation.py
python experiments/08_v3_v4_comparative_analysis.py
```

The V4 comparison writes case-level and metric outputs under `results/`. Experiment 08 compares the static and context results across V3 and V4 and writes the comparative tables and report.

Other scripts in `experiments/` perform rule audits, rule-validity/coverage checks, and V4 benchmark coverage analysis. Refer to each script's header and configured input/output paths before running it.

### Separate evidence-informed static baseline variant

Experiment 03E1 is a **separate, study-defined static baseline variant**. It does not overwrite or replace the frozen V3/V4 static baseline used in the CAN-context comparative analysis.

Configuration:
- `experiments/static_priority_evidence_v1.json`

Runner:
```bash
python experiments/03_static_baseline_evidence_v1.py
```

Outputs:
- `results/static_baseline_evidence_v1_v3.csv`
- `results/static_baseline_evidence_v1_v4.csv`
- `results/static_baseline_evidence_v1_metrics.json`

The priority classes and numeric scores are an experimental rubric, not OEM severity ratings, universal urgency levels, or safety classifications.

## Selected results

The frozen static baseline and CAN-context refinement produced the following results in the committed comparative analysis:

| Benchmark | Cases | Static Top-1 | Context Top-1 | Context MRR | Context mean primary rank |
|---|---:|---:|---:|---:|---:|
| V3 | 48 | 41.67% | 58.33% | 0.7917 | 1.4167 |
| V4 | 24 | 50.00% | 83.33% | 0.9167 | 1.1667 |

For both benchmarks, context refinement improved the primary-DTC rank in eight cases and worsened it in zero cases within the evaluated data. These are descriptive benchmark results; V4 includes repeated templates and does not establish generalization to unseen scenarios.

The separate evidence-informed static baseline results are available in the [metrics JSON](results/static_baseline_evidence_v1_metrics.json) and the two case-level CSVs linked above. They must not be confused with the frozen static baseline in the main comparative experiment.

## Limitations

- Synthetic controlled benchmarks; no physical-vehicle recordings or real-vehicle validation.
- Candidate DTCs are supplied upstream; candidate detection is outside the study.
- Limited scenario/template and candidate-pair coverage.
- Context features are rule-based proxies and are not calibrated fault probabilities.
- No claim of statistical significance, OEM-wide priority validity, or production ECU real-time suitability is made.

For the detailed results, rule-ablation findings, coverage analysis, and limitations, see [Experimental Conclusions and Limitations](docs/experimental_conclusions_and_limitations.md).
