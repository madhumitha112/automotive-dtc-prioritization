# Experiment Guide

This directory contains the Python scripts used to validate the benchmark data, extract diagnostic cases, evaluate static DTC prioritization, apply CAN-context-based refinement, and compare the experimental methods.

All commands below should be executed from the **repository root directory**, not from inside the `experiments/` folder.

## Research Objective

The experiments evaluate whether lightweight CAN-context-based prioritization can improve the ranking of candidate Automotive Diagnostic Trouble Codes (DTCs) compared with a static-priority baseline.

The methods operate on a candidate DTC list supplied to the ranking process. They do not detect faults, generate candidate DTCs, or replace an OEM diagnostic system.

The term *lightweight* refers to the rule-based and computationally simple design of the context-refinement method. The experiments do not measure production ECU execution time, memory consumption, or real-time suitability.

## Dataset Overview

The repository contains two controlled diagnostic benchmarks:

- **V3:** 48 diagnostic cases.
- **V4:** 24 diagnostic cases based on 12 scenario templates, with two repetitions per template.

The benchmarks are synthetic and controlled. They are not recordings collected from physical vehicles.

Primary-DTC labels and evaluation metadata are used to assess ranking performance. They must not be used as ranking inputs.

V4 repetitions are not independent scenario families. Results should be interpreted within the evaluated benchmark rather than as proof of generalization to unseen vehicle conditions.

## Experiment Workflow

### 1. Dataset Validation

Script:

```bash
python experiments/01_validate_dataset.py
```

Purpose:

- Validate the raw benchmark data.
- Check the structure and consistency of the input files.
- Generate a validation report.

Output:

```text
results/dataset_validation_report.json
```

The validation script reads the raw data and does not modify it.

### 2. Diagnostic Case Extraction

Script:

```bash
python experiments/02_extract_diagnostic_cases.py
```

Purpose:

- Convert the raw diagnostic-event data into structured diagnostic cases.
- Prepare the V3 cases for the prioritization experiments.

Output:

```text
results/diagnostic_cases.csv
```

The V4 benchmark cases are maintained separately:

```text
results/diagnostic_cases_v4.csv
```

### 3. Static DTC Baseline

The static-priority experiments use fixed priority scores to rank the candidate DTCs.

Scripts:

```text
03_static_baseline.py
03_static_baseline_v4.py
```

Purpose:

- Establish the static ranking baseline.
- Evaluate the baseline on the V3 and V4 benchmarks.
- Preserve the original baseline configuration for comparative evaluation.

The original V3 and V4 static baselines are frozen. Do not modify their scores as part of the completed comparative experiment.

### 4. CAN-Context-Based Refinement

Scripts:

```text
04_context_refinement.py
04_context_refinement_v4.py
```

Purpose:

- Apply selected CAN-context rules to refine the ranking of candidate DTCs.
- Evaluate whether contextual evidence improves the position of the primary DTC.
- Compare the context-refined ranking against the static baseline.

The method uses rule-based context evidence. It does not calculate calibrated fault probabilities.

### 5. V4 Comparative Evaluation

Script:

```bash
python experiments/05_v4_comparative_evaluation.py
```

Purpose:

- Compare the static baseline and CAN-context refinement on V4.
- Calculate ranking metrics.
- Generate case-level and summary results.

The comparison evaluates both methods on the same candidate cases.

### 6. Rule Audit and Coverage Analysis

The following scripts support rule-level and benchmark-level analysis:

```text
04a_context_rule_audit.py
04b_rule_validity_coverage.py
04c_generate_expanded_benchmark.py
06_v4_context_rule_audit.py
07_v4_coverage_validity.py
```

Purpose:

- Examine the behavior of context rules.
- Review rule validity and coverage.
- Assess the coverage of the V4 benchmark.

These analyses support interpretation of the experimental results. They do not constitute real-vehicle validation.

### 7. V3 and V4 Comparative Analysis

Script:

```bash
python experiments/08_v3_v4_comparative_analysis.py
```

Purpose:

- Compare the static and CAN-context results across V3 and V4.
- Summarize ranking improvements and unchanged or worsened cases.
- Generate consolidated comparative outputs.

The existing comparative outputs are retained in the `results/` directory.

### 8. Separate Evidence-Informed Static Baseline

This experiment evaluates a separate static-priority configuration based on a study-defined evidence-informed rubric.

Configuration:

```text
experiments/static_priority_evidence_v1.json
```

Runner:

```bash
python experiments/03_static_baseline_evidence_v1.py
```

Outputs:

```text
results/static_baseline_evidence_v1_v3.csv
results/static_baseline_evidence_v1_v4.csv
results/static_baseline_evidence_v1_metrics.json
```

This variant is separate from the frozen static baseline used in the original V3/V4 CAN-context comparison.

The priority classes and scores are experimental values. They are not universal OEM severity ratings, safety classifications, or calibrated diagnostic urgency levels.

## Evaluation Metrics

The experiments use ranking metrics to evaluate the position of the primary DTC:

- **Top-1 Accuracy:** Proportion of cases where the primary DTC is ranked first.
- **Top-3 Accuracy:** Proportion of cases where the primary DTC appears within the first three positions.
- **Mean Reciprocal Rank (MRR):** Average reciprocal position of the primary DTC.
- **Mean Primary Rank:** Average rank assigned to the primary DTC.

These metrics evaluate ranking performance within the supplied benchmark cases.

## Reproducibility Notes

- Run commands from the repository root.
- Use the committed dataset files and experiment configurations.
- Preserve the frozen V3/V4 baseline and context-refinement scripts.
- Do not modify existing result files merely to improve reported metrics.
- If an experiment is rerun, record any changes to its inputs, configuration, or implementation.
- Review each script's configured input and output paths before executing it.

## Research Limitations

- The benchmarks are synthetic and controlled.
- No physical-vehicle recordings or real-vehicle validation were performed.
- Candidate DTC generation is outside the scope of the ranking method.
- The number of scenario templates and candidate-DTC combinations is limited.
- Some context indicators are signal proxies rather than measured ECU signals.
- The context rules are not calibrated fault probabilities.
- The results do not establish statistical significance, generalization to unseen scenarios, or production ECU real-time suitability.

For the consolidated findings, rule-ablation observations, and detailed limitations, refer to:

```text
docs/experimental_conclusions_and_limitations.md
```
