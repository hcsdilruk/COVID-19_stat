# COVID-19_stat
---

## Team Analysis Folders

This repository contains three statistical analysis components
using the 2022 COVID-19 VAERS dataset.

### 1. Suwahas - Time-Series Analysis
- Weekly reporting trends
- ARIMA forecasting
- Stationarity and autocorrelation analysis
- Forecast diagnostics and sensitivity analysis

[View Suwahas's Work](analysis/suwahas/)

### 2. Tharusha - Regression Analysis
- Multiple linear regression
- Median regression
- Logistic regression
- Model diagnostics and sensitivity analysis

[View Tharusha's Work](analysis/tharusha/)

### 3. Chamoth - Hypothesis Testing
- Manufacturer association testing
- Age-group differences
- Sex association testing
- Symptom-count comparison
- Holm multiple-testing correction

[View Chamoth's Work](analysis/chamoth/)

## Shared Project Structure

- `analysis/` - Individual student analyses
- `data/` - Shared local dataset
- `legacy/` - Original scripts retained for reference

## Important Research Notes

The primary flag records whether hospitalization or death
was explicitly marked Yes in a VAERS report.

The analyses describe patterns among submitted VAERS reports.
They do not establish vaccine causation or population risk.

All three analysis components select the lowest available
ORDER per VAERS ID, retaining 212,164 primary reports.

The time-series analysis uses 51 complete weeks, covering
207,803 reports. The remaining 4,361 reports fall outside
that complete-week window: 1,916 before January 3 and
2,445 on or after December 26, 2022.

On the corrected cohort, the previous-week baseline
outperformed both ARIMA forecasting alternatives.

Some ages may be estimated; the original age-imputation
indicator has not been recovered.
