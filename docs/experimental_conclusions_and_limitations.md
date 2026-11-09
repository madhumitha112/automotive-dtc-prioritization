
# Experimental Conclusions and Limitations

## 1. Research Objective

The study, titled **"Comparative Evaluation of Static and Lightweight CAN-Context-Based Prioritization of Automotive Diagnostic Trouble Codes for Improved Diagnostic Efficiency,"** investigated whether observable CAN-context evidence could improve the prioritization of candidate Diagnostic Trouble Codes (DTCs) over a fixed static-priority baseline.

The experimental approach compared:

1. A static DTC prioritization baseline using predefined priority scores.
2. A lightweight CAN-context-based refinement method that combines static priorities with context-derived evidence.

The evaluation focused on the ranking position of the known primary DTC within a supplied candidate set. The study did not evaluate the initial detection or generation of DTC candidates.

## 2. Summary of Experimental Results

### 2.1 V3 Benchmark Results

The V3 benchmark contained 48 diagnostic cases. The static baseline and CAN-context refinement method were evaluated using the same cases and candidate sets.

| Metric | Static Baseline | CAN-Context Refinement |
|---|---:|---:|
| Number of cases | 48 | 48 |
| Top-1 accuracy | 41.67% | 58.33% |
| Top-3 accuracy | 100.00% | 100.00% |
| Mean Reciprocal Rank (MRR) | 0.6806 | 0.7917 |
| Mean primary-DTC rank | 1.7500 | 1.4167 |
| Median primary-DTC rank | 2.00 | 1.00 |

The CAN-context refinement method improved Top-1 accuracy by 16.67 percentage points compared with the static baseline.

The paired case analysis showed:

- 8 cases improved in ranking.
- 0 cases worsened.
- 40 cases remained unchanged.
- 8 cases changed from an incorrect Top-1 ranking to a correct Top-1 ranking.

These results indicate that the context-based refinement improved the ordering of candidate DTCs in a subset of the V3 benchmark cases without producing a ranking regression in the evaluated cases.

### 2.2 V4 Benchmark Results

The V4 benchmark contained 24 diagnostic cases derived from 12 scenario templates, with two repetitions per template.

The V3 static-priority configuration and CAN-context refinement rules were retained for the V4 evaluation. The V4 evaluation therefore examined the performance of the existing method on a separate benchmark configuration without changing its ranking rules or context weight.

| Metric | Static Baseline | CAN-Context Refinement |
|---|---:|---:|
| Number of cases | 24 | 24 |
| Top-1 accuracy | 50.00% | 83.33% |
| Top-3 accuracy | 100.00% | 100.00% |
| Mean Reciprocal Rank (MRR) | 0.7500 | 0.9167 |
| Mean primary-DTC rank | 1.5000 | 1.1667 |
| Median primary-DTC rank | 1.50 | 1.00 |

The CAN-context refinement method improved Top-1 accuracy by 33.33 percentage points compared with the V4 static baseline.

The paired case analysis showed:

- 8 cases improved in ranking.
- 0 cases worsened.
- 16 cases remained unchanged.
- 8 cases changed from an incorrect Top-1 ranking to a correct Top-1 ranking.

The context method achieved the correct Top-1 ranking in 20 of the 24 V4 cases.

However, the 24 cases represent only 12 distinct scenario templates. The two repetitions within each template should not be interpreted as independent scenario families or independent real-vehicle trials.

### 2.3 Comparative Interpretation of V3 and V4

The context-based method improved Top-1 accuracy in both benchmark evaluations.

| Measure | V3 Improvement | V4 Improvement |
|---|---:|---:|
| Top-1 accuracy | +16.67 percentage points | +33.33 percentage points |
| MRR | +0.1111 | +0.1667 |
| Mean primary-DTC rank | Improved by 0.3333 positions | Improved by 0.3333 positions |

Both benchmarks recorded eight cases in which the context method corrected an incorrect static Top-1 ranking. Neither benchmark recorded a case in which the context method changed a correct static Top-1 ranking into an incorrect one.

The larger percentage-point improvement observed in V4 should not be interpreted as proof that the method generalizes better to a wider range of diagnostic conditions. V3 and V4 differ in case count, candidate-set composition and scenario design. In addition, V4 contains repeated instances of 12 templates rather than 24 distinct scenario families.

Consequently, the comparison demonstrates consistent improvement within the two evaluated benchmark configurations, but does not isolate the effect of benchmark size or establish generalization to unseen vehicle conditions.

## 3. V4 Context-Rule Audit Findings

The V4 context-rule audit examined the contribution of individual DTC context rules using a leave-one-rule-out ablation procedure.

The full context model achieved:

- Top-1 accuracy: 83.33%.
- MRR: 0.9167.
- Mean primary-DTC rank: 1.1667.

The ablation results identified different levels of contribution across the rules.

### 3.1 Rules with Positive Ranking Contributions

Removing the P0121 context contribution reduced Top-1 accuracy from 83.33% to 66.67%.

Removing either the P0172 or P0128 context contribution reduced Top-1 accuracy from 83.33% to 75.00%.

These results indicate that the corresponding context contributions were useful for ranking performance within the evaluated V4 benchmark.

### 3.2 P0130 Context Rule

Removing the P0130 context contribution increased Top-1 accuracy from 83.33% to 91.67%.

The P0130 context contribution therefore had an adverse effect on aggregate ranking performance in the V4 benchmark.

The failure analysis identified two affected cases:

- V4_019
- V4_020

In both cases, P0171 was the known primary DTC, but P0130 was ranked first.

The P0130 context score was 0.9973, while the P0171 context score was 1.0000. The small difference in context evidence, combined with the static priority scores, resulted in P0130 remaining above P0171.

This indicates that the combination of static priority and context evidence does not always produce the intended ranking, even when the primary DTC has a slightly higher context score.

The finding is specific to the current benchmark and does not establish that the P0130 rule would be counterproductive under all operating conditions.

### 3.3 Rules with No Aggregate Ablation Effect

Removing P0335, P0300, C0035 or P0171 individually did not change the aggregate V4 ranking metrics.

This does not establish that these rules are universally unnecessary. It indicates only that their removal did not change the measured aggregate performance in the evaluated benchmark.

## 4. V4 Benchmark Coverage and Validity Findings

### 4.1 Case and Template Structure

The V4 benchmark consisted of:

- 24 diagnostic cases.
- 12 scenario templates.
- Two repetitions per template.
- Eight distinct DTCs.
- Six represented candidate-DTC pairs.

All 24 cases passed the implemented integrity checks, including candidate membership, candidate uniqueness, observation interval consistency, repetition structure and scenario-manifest consistency.

### 4.2 DTC Coverage

DTC representation was not uniform.

P0171 had the broadest representation, with six primary and six competing appearances across three templates.

P0300 and P0121 each appeared as the primary DTC in four cases and as a competing DTC in four cases, across two templates.

P0172, P0128, P0130, P0335 and C0035 each had two primary and two competing appearances, but only across one template for each role.

The limited template diversity for these five DTCs restricts the strength of conclusions about their individual context-rule performance.

### 4.3 Candidate-Pair Coverage

There are 28 possible unordered pairs among eight DTCs.

The V4 benchmark represented six of these pairs:

1. P0171 and P0172.
2. P0128 and P0171.
3. C0035 and P0121.
4. P0300 and P0335.
5. P0130 and P0171.
6. P0121 and P0300.

All six represented pairs were evaluated in both primary-DTC directions.

However, 22 possible DTC pairs were not represented. The evaluation therefore does not establish how the prioritization method behaves across all possible candidate combinations.

### 4.4 Failure Patterns

The V4 context model produced four incorrect Top-1 rankings:

| Case | Primary DTC | Context Top-1 DTC |
|---|---|---|
| V4_013 | P0300 | P0335 |
| V4_014 | P0300 | P0335 |
| V4_019 | P0171 | P0130 |
| V4_020 | P0171 | P0130 |

These four cases correspond to two scenario templates, each repeated twice.

The P0300/P0335 failure pattern indicates that the current context evidence did not sufficiently distinguish the two candidates in the MISFIRE_LOAD condition.

The P0171/P0130 failure pattern is associated with the adverse contribution of the P0130 context rule.

These findings identify specific areas requiring further investigation rather than demonstrating universal failure of the associated DTC rules.

## 5. Overall Experimental Conclusions

The experiments demonstrate that lightweight CAN-context evidence can improve the prioritization of candidate DTCs compared with a fixed static-priority baseline within the evaluated V3 and V4 benchmark configurations.

The improvement was reflected in higher Top-1 accuracy and Mean Reciprocal Rank, together with a reduction in the mean rank of the known primary DTC.

The V3 evaluation recorded a Top-1 improvement of 16.67 percentage points, while the V4 evaluation recorded an improvement of 33.33 percentage points.

The paired analyses identified eight corrected cases in each benchmark and no Top-1 regressions.

The V4 rule audit further demonstrated that context contributions were not uniformly beneficial. P0121, P0172 and P0128 contributed positively to ranking performance in the evaluated cases, while the P0130 contribution reduced aggregate performance.

The coverage analysis also identified limitations in DTC representation, candidate-pair diversity and scenario-template diversity. These limitations restrict the extent to which the results can be generalized beyond the tested benchmark configurations.

Overall, the findings support the use of lightweight CAN-context evidence as a potentially useful refinement mechanism for ordering an already available set of candidate DTCs. The experiments do not demonstrate that the method independently detects faults, identifies DTCs from raw vehicle signals, or achieves equivalent performance in physical vehicles.

## 6. Research Limitations

The following limitations apply to the interpretation of the experimental findings.

### 6.1 Controlled Synthetic Benchmark

The V4 benchmark is a controlled synthetic benchmark and does not consist of CAN recordings collected from physical vehicles.

The results therefore demonstrate performance on the defined benchmark conditions rather than validated diagnostic effectiveness in real vehicle operating environments.

### 6.2 Limited Scenario Diversity

V4 contains 12 scenario templates, each repeated twice.

The repeated cases provide variation within the defined templates but do not represent 24 independent scenario families. The effective diversity of the evaluated diagnostic conditions is therefore more limited than the raw case count suggests.

### 6.3 Limited Candidate-Pair Coverage

Only six of the 28 possible DTC pairs among the eight evaluated DTCs were represented.

The prioritization method has not been evaluated across the remaining candidate combinations.

### 6.4 Uneven DTC Representation

Several DTCs have only one template representing their primary role and one template representing their competing role.

The corresponding context-rule findings should therefore be interpreted as benchmark-specific observations rather than broad evidence of rule reliability.

### 6.5 Candidate Generation Is Outside the Scope

The method assumes that an upstream diagnostic process has already supplied a candidate DTC set.

The experiments evaluate the ordering of these candidates and do not evaluate fault detection, DTC generation or the accuracy of the upstream diagnostic process.

### 6.6 Limited Signal Representation

Some diagnostic distinctions rely on indirect signal proxies.

For example, the P0335 context rule uses RPM instability as indirect evidence rather than a direct crankshaft-position sensor waveform.

Similarly, the P0130 context rule uses the defined oxygen-signal range as a proxy and does not constitute a complete oxygen-sensor circuit diagnosis.

The signal decoding definitions are benchmark-specific and should not be interpreted as universal OEM CAN signal layouts.

### 6.7 Context Scores Are Not Fault Probabilities

The context scores represent threshold-based evidence contributions used by the ranking algorithm.

They are not calibrated probabilities of component failure or diagnostic certainty.

### 6.8 No Statistical Generalization Claim

The reported improvements are descriptive results from the evaluated benchmark cases.

The experiments do not establish statistical significance or generalization to unseen real-vehicle operating conditions.

### 6.9 Limited Vehicle and Operating-Condition Diversity

The evaluation does not cover a representative range of vehicle manufacturers, vehicle architectures, engine configurations, sensor implementations, environmental conditions or driving conditions.

### 6.10 Fixed Ranking Configuration

The static priority values, context rules and context weight were retained for the V4 evaluation.

The P0130 ablation finding identifies a potential weakness, but the main V4 evaluation was not retuned using that finding.

This preserves the distinction between evaluation of the frozen method and any subsequent exploratory improvement.

## 7. Future Work

The following directions may be considered for extending the research:

1. Evaluate the prioritization method using real vehicle CAN recordings and documented diagnostic conditions.
2. Increase the number of distinct scenario families and repetitions while distinguishing independent scenarios from repeated signal variations.
3. Expand candidate-pair coverage beyond the six combinations represented in V4.
4. Investigate the P0130 context contribution and evaluate alternative signal interpretations in a separately labelled exploratory experiment.
5. Improve the P0335 context representation through direct crankshaft-position signal evidence where suitable data are available.
6. Evaluate the method with larger candidate sets and more varied diagnostic combinations.
7. Assess the sensitivity of ranking outcomes to signal noise, missing data and variations in operating conditions.
8. Evaluate the method across different vehicle architectures and OEM-specific signal definitions.
9. Investigate calibration and interpretation of context scores without treating them as fault probabilities unless independently validated.

## 8. Final Conclusion

The study provides benchmark-level evidence that incorporating lightweight CAN-context information into a fixed static DTC-priority framework can improve the ordering of candidate diagnostic trouble codes.

The observed improvements in Top-1 accuracy and Mean Reciprocal Rank across V3 and V4 support the research proposition within the tested benchmark configurations.

However, the limited scenario diversity, restricted candidate-pair coverage, controlled synthetic data provenance and indirect signal representations prevent the findings from being interpreted as demonstrated real-vehicle diagnostic effectiveness.

The principal contribution of the study is therefore an experimental evaluation of a lightweight context-based DTC prioritization approach, together with an assessment of its rule-level contributions, failure patterns and benchmark-validity limitations.
