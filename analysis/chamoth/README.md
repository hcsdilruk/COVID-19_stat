# Chamoth - Hypothesis Testing

## Research Focus
Statistical associations and group differences in
2022 COVID-19 VAERS reports.

## Main File
- chamoth_hypotheses_validated.py

## Hypotheses
- H1: Manufacturer and recorded serious-outcome flags
- H2: Age differences by recorded outcome flag
- H3: Sex and recorded outcome flags
- H4: Symptom-count differences by death flag

## Methods
- Chi-square tests
- Welch's t-tests
- Mann-Whitney U tests
- Effect sizes and confidence intervals
- Holm multiple-testing correction

## Results
- figures/ - Four hypothesis visualizations
- outputs/chamoth_hypothesis_results.csv

## Validation
- 212,164 unique reports, lowest available ORDER per ID
- Symptom-count validation passed

## Limitations
The available age field may contain estimated ages.
VAERS associations do not establish vaccine causation.

## Data
Shared dataset: ../../data/2022_COVID_VAERS_clean.csv
