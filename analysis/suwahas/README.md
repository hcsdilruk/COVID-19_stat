# Suwahas - Time-Series Analysis

## Research Focus
Weekly trends in reported hospitalization or death flags
among 2022 COVID-19 VAERS reports.

## Main File
- suwahas_time_series.ipynb

## Methods
- Weekly aggregation and trend analysis
- ADF stationarity testing
- Autocorrelation analysis
- ARIMA forecasting
- Forecast evaluation and sensitivity analysis

## Results
- figures/ - Six visualizations
- outputs/ - Five CSV result files

## Important Note
The primary cohort selects the lowest available ORDER
per VAERS ID: 212,164 unique reports.

The weekly analysis retains 51 complete weeks, from
January 3 through December 25, 2022:
- Reports included: 207,803
- Reports before January 3: 1,916
- Reports on or after December 26: 2,445

The 10-week forecasting evaluation found that the
previous-week baseline outperformed both ARIMA models.

MAE:
- Previous-week baseline: 1.2502
- Original ARIMA: 1.3008
- Adjusted-training ARIMA: 1.4607

These results describe reported outcomes, not vaccine
causation or population-level adverse-event risks.

## Data
Shared dataset: ../../data/2022_COVID_VAERS_clean.csv
