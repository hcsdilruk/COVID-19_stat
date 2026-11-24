
from pathlib import Path
import pandas as pd

# Load dataset
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "2022_COVID_VAERS_clean.csv"

df = pd.read_csv(DATA, low_memory=False)

print("Original rows:", len(df))
print("Original columns:", len(df.columns))

# Match Suwahas's current primary cohort

# Retain the lowest available ORDER for each VAERS ID
df["ORDER_NUM"] = pd.to_numeric(
    df["ORDER"], errors="coerce"
)

assert df["ORDER_NUM"].notna().all()

primary = (
    df.sort_values(
        ["VAERS_ID", "ORDER_NUM"],
        kind="stable"
    )
    .drop_duplicates(
        subset="VAERS_ID",
        keep="first"
    )
    .copy()
)

assert primary["VAERS_ID"].is_unique
assert len(primary) == 212164


assert primary["VAERS_ID"].is_unique

# Shared outcome: hospitalization OR death recorded Yes
hospital = (
    primary["HOSPITAL_STATUS"]
    .astype("string")
    .str.strip()
    .str.lower()
    .eq("yes")
    .fillna(False)
)

death = (
    primary["DIED_STATUS"]
    .astype("string")
    .str.strip()
    .str.lower()
    .eq("yes")
    .fillna(False)
)

primary["FLAG_YES"] = (hospital | death).astype(int)

print("\n--- DATA VALIDATION ---")
print("Primary reports:", len(primary))
print("Unique report IDs:", primary["VAERS_ID"].nunique())
print("Hospitalization Yes:", hospital.sum())
print("Death Yes:", death.sum())
print("Any selected Yes:", primary["FLAG_YES"].sum())

print("\nOutcome counts:")
print(primary["FLAG_YES"].value_counts())

print("\nMissing ages:", primary["AGE_YRS"].isna().sum())
print("\nPreparation successful!")


# ==========================================
# H1 - MANUFACTURER ASSOCIATION
# ==========================================

import json
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

BRANDS = {
    "PFIZER\\BIONTECH": "Pfizer-BioNTech",
    "MODERNA": "Moderna",
    "JANSSEN": "Janssen",
    "NOVAVAX": "Novavax"
}

def identify_manufacturer(value):
    if pd.isna(value):
        return None

    items = json.loads(value)

    if not isinstance(items, list):
        raise ValueError("Expected manufacturer list")

    # Include only exactly one named manufacturer
    if len(items) != 1:
        return None

    name = str(items[0]).strip().upper()
    return BRANDS.get(name)

primary["MANUFACTURER"] = (
    primary["COVID_MANUFACTURERS"]
    .apply(identify_manufacturer)
)

h1 = primary.dropna(
    subset=["MANUFACTURER"]
).copy()

table = pd.crosstab(
    h1["MANUFACTURER"],
    h1["FLAG_YES"]
).reindex(
    index=list(BRANDS.values()),
    columns=[0, 1],
    fill_value=0
)

print("\n========== H1 RESULTS ==========")
print("\nContingency table:")
print(table)

chi2, p_value, dof, expected = (
    chi2_contingency(table)
)

n = table.to_numpy().sum()

cramers_v = np.sqrt(
    chi2 / (n * min(table.shape[0]-1,
                    table.shape[1]-1))
)

print("\nChi-square:", chi2)
print("Degrees of freedom:", dof)
print("P-value:", p_value)
print("Cramer's V:", cramers_v)
print("Minimum expected count:", expected.min())
print("Reports included:", n)
print("Reports excluded:", len(primary)-n)

print("\nManufacturer percentages and 95% CIs:")

for name, row in table.iterrows():
    total = int(row.sum())
    flagged = int(row[1])

    if total == 0:
        continue

    rate = 100 * flagged / total

    low, high = proportion_confint(
        flagged, total,
        alpha=0.05,
        method="wilson"
    )

    print(
        f"{name}: {rate:.2f}% "
        f"(95% CI {100*low:.2f}% - "
        f"{100*high:.2f}%), n={total}"
    )

print("\nDecision:")

if p_value < 0.05:
    print("Reject H0: Association detected.")
else:
    print("Fail to reject H0.")

print("\nH1 completed.")


# ==========================================
# H2 - AGE DIFFERENCE TESTING
# ==========================================

from scipy import stats
import numpy as np

print("\n========== H2 RESULTS ==========")

# Convert age values to numbers
h2 = primary.copy()
h2["AGE_NUMERIC"] = pd.to_numeric(
    h2["AGE_YRS"], errors="coerce"
)

# Exclude unavailable and implausible ages
valid_age = h2["AGE_NUMERIC"].between(0, 120)

print("Excluded age records:", (~valid_age).sum())

h2 = h2.loc[valid_age].copy()

# Split by flag recording
flagged_age = h2.loc[
    h2["FLAG_YES"] == 1, "AGE_NUMERIC"
].to_numpy(dtype=float)

not_flagged_age = h2.loc[
    h2["FLAG_YES"] == 0, "AGE_NUMERIC"
].to_numpy(dtype=float)

n1 = len(flagged_age)
n0 = len(not_flagged_age)

print("\nSample sizes:")
print("Flag recorded:", n1)
print("No selected Yes recorded:", n0)

# Descriptive statistics
mean1 = flagged_age.mean()
mean0 = not_flagged_age.mean()

median1 = np.median(flagged_age)
median0 = np.median(not_flagged_age)

sd1 = flagged_age.std(ddof=1)
sd0 = not_flagged_age.std(ddof=1)

print("\nMean ages:")
print("Flag recorded:", round(mean1, 3))
print("No selected Yes recorded:", round(mean0, 3))

print("\nMedian ages:")
print("Flag recorded:", median1)
print("No selected Yes recorded:", median0)

# Welch's independent-samples t-test
t_result = stats.ttest_ind(
    flagged_age,
    not_flagged_age,
    equal_var=False,
    alternative="two-sided"
)

mean_difference = mean1 - mean0

# Welch confidence interval
v1 = sd1**2 / n1
v0 = sd0**2 / n0

se_difference = np.sqrt(v1 + v0)

welch_df = (v1 + v0)**2 / (
    (v1**2 / (n1 - 1)) +
    (v0**2 / (n0 - 1))
)

critical = stats.t.ppf(0.975, welch_df)

ci_low = mean_difference - critical * se_difference
ci_high = mean_difference + critical * se_difference

# Cohen's d: standardized mean difference
pooled_sd = np.sqrt(
    ((n1-1)*sd1**2 + (n0-1)*sd0**2) /
    (n1+n0-2)
)

cohens_d = mean_difference / pooled_sd

print("\n--- WELCH T-TEST ---")
print("T-statistic:", t_result.statistic)
print("P-value:", t_result.pvalue)
print("Degrees of freedom:", welch_df)

print("\nMean age difference:", mean_difference)
print("95% CI:", ci_low, "to", ci_high)
print("Cohen's d:", cohens_d)

# Mann-Whitney U test
mw_result = stats.mannwhitneyu(
    flagged_age,
    not_flagged_age,
    alternative="two-sided",
    method="asymptotic"
)

rank_biserial = (
    2 * mw_result.statistic / (n1*n0) - 1
)

print("\n--- MANN-WHITNEY U TEST ---")
print("U-statistic:", mw_result.statistic)
print("P-value:", mw_result.pvalue)
print("Rank-biserial effect:", rank_biserial)

print("\n--- H2 DECISIONS ---")

if t_result.pvalue < 0.05:
    print("Welch: Reject H0")
else:
    print("Welch: Fail to reject H0")

if mw_result.pvalue < 0.05:
    print("Mann-Whitney: Reject H0")
else:
    print("Mann-Whitney: Fail to reject H0")

print("\nH2 completed.")


# ==========================================
# H3 - SEX AND OUTCOME ASSOCIATION
# ==========================================

from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

print("\n========== H3 RESULTS ==========")

# Keep only reports with male or female recorded sex
h3 = primary.copy()

h3["SEX_GROUP"] = (
    h3["SEX"]
    .astype("string")
    .str.strip()
    .str.upper()
)

h3 = h3[
    h3["SEX_GROUP"].isin(["M", "F"])
].copy()

# Contingency table
table_h3 = pd.crosstab(
    h3["SEX_GROUP"],
    h3["FLAG_YES"]
).reindex(
    index=["F", "M"],
    columns=[0, 1],
    fill_value=0
)

print("\nContingency table:")
print(table_h3)

# Chi-square test
chi2_h3, p_h3, dof_h3, expected_h3 = (
    chi2_contingency(table_h3)
)

n_h3 = table_h3.to_numpy().sum()

cramers_v_h3 = np.sqrt(
    chi2_h3 / (
        n_h3 * min(
            table_h3.shape[0] - 1,
            table_h3.shape[1] - 1
        )
    )
)

print("\n--- CHI-SQUARE RESULTS ---")
print("Chi-square:", chi2_h3)
print("Degrees of freedom:", dof_h3)
print("P-value:", p_h3)
print("Cramer's V:", cramers_v_h3)
print("Minimum expected count:", expected_h3.min())

print("\nReports included:", n_h3)
print("Reports excluded:", len(primary) - n_h3)

# Percentages with 95% Wilson confidence intervals
print("\n--- SEX GROUP PERCENTAGES ---")

for sex, row in table_h3.iterrows():

    total = int(row.sum())
    flagged = int(row[1])

    percentage = 100 * flagged / total

    low, high = proportion_confint(
        flagged,
        total,
        alpha=0.05,
        method="wilson"
    )

    name = "Female" if sex == "F" else "Male"

    print(
        f"{name}: {percentage:.2f}% "
        f"(95% CI {100*low:.2f}% - "
        f"{100*high:.2f}%), n={total}"
    )

# Risk difference within the selected report pool
female_rate = (
    table_h3.loc["F", 1] / table_h3.loc["F"].sum()
)

male_rate = (
    table_h3.loc["M", 1] / table_h3.loc["M"].sum()
)

print(
    "\nMale - Female flagged percentage difference:",
    round(100 * (male_rate - female_rate), 4),
    "percentage points"
)

print("\n--- H3 DECISION ---")

if p_h3 < 0.05:
    print("Reject H0: Association detected.")
else:
    print("Fail to reject H0.")

print("\nH3 completed.")


# ==========================================
# H4 - SYMPTOM COUNT DIFFERENCE
# ==========================================

from scipy import stats
import numpy as np

print("\n========== H4 RESULTS ==========")

h4 = primary.copy()

# Convert symptom counts to numeric values
h4["SYMPTOM_NUMERIC"] = pd.to_numeric(
    h4["SYMPTOM_COUNT"], errors="coerce"
)

# Standardize death-recording status
h4["DEATH_STATUS"] = (
    h4["DIED_STATUS"]
    .astype("string")
    .str.strip()
    .str.lower()
)

print("\nOriginal death-status distribution:")
print(h4["DEATH_STATUS"].value_counts(dropna=False))

# Audit symptom counts
valid_count = (
    h4["SYMPTOM_NUMERIC"].notna()
    & h4["SYMPTOM_NUMERIC"].ge(0)
    & h4["SYMPTOM_NUMERIC"].eq(
        h4["SYMPTOM_NUMERIC"].round()
    )
)

valid_status = h4["DEATH_STATUS"].isin(
    ["yes", "unknown"]
)

print(
    "\nInvalid or unavailable symptom counts:",
    (~valid_count).sum()
)

print(
    "Reports with other or missing death status:",
    (~valid_status).sum()
)

# Retain valid records
h4 = h4.loc[
    valid_count & valid_status
].copy()

print("\nSymptom count summary:")
print(
    h4["SYMPTOM_NUMERIC"].describe(
        percentiles=[0.5, 0.9, 0.99]
    )
)

# Separate groups
death_yes = h4.loc[
    h4["DEATH_STATUS"] == "yes",
    "SYMPTOM_NUMERIC"
].to_numpy(dtype=float)

death_unknown = h4.loc[
    h4["DEATH_STATUS"] == "unknown",
    "SYMPTOM_NUMERIC"
].to_numpy(dtype=float)

n1 = len(death_yes)
n0 = len(death_unknown)

assert n1 >= 2 and n0 >= 2

mean1 = death_yes.mean()
mean0 = death_unknown.mean()

median1 = np.median(death_yes)
median0 = np.median(death_unknown)

sd1 = death_yes.std(ddof=1)
sd0 = death_unknown.std(ddof=1)

print("\n--- GROUP SUMMARY ---")
print("Death Yes reports:", n1)
print("Death Unknown reports:", n0)

print("\nMean symptom counts:")
print("Death Yes:", mean1)
print("Death Unknown:", mean0)

print("\nMedian symptom counts:")
print("Death Yes:", median1)
print("Death Unknown:", median0)

# Welch's independent-samples t-test
welch = stats.ttest_ind(
    death_yes,
    death_unknown,
    equal_var=False,
    alternative="two-sided"
)

difference = mean1 - mean0

v1 = sd1**2 / n1
v0 = sd0**2 / n0

se = np.sqrt(v1 + v0)

df_welch = (v1 + v0)**2 / (
    v1**2 / (n1 - 1)
    + v0**2 / (n0 - 1)
)

critical = stats.t.ppf(0.975, df_welch)

ci_lower = difference - critical * se
ci_upper = difference + critical * se

# Cohen's d
pooled_sd = np.sqrt(
    ((n1 - 1)*sd1**2 + (n0 - 1)*sd0**2)
    / (n1 + n0 - 2)
)

cohens_d_h4 = difference / pooled_sd

print("\n--- WELCH T-TEST ---")
print("T-statistic:", welch.statistic)
print("P-value:", welch.pvalue)
print("Degrees of freedom:", df_welch)
print("Mean difference:", difference)
print("95% CI:", ci_lower, "to", ci_upper)
print("Cohen's d:", cohens_d_h4)

# Mann-Whitney U test
mw_h4 = stats.mannwhitneyu(
    death_yes,
    death_unknown,
    alternative="two-sided",
    method="asymptotic"
)

rank_effect = (
    2 * mw_h4.statistic / (n1*n0) - 1
)

print("\n--- MANN-WHITNEY U ---")
print("U-statistic:", mw_h4.statistic)
print("P-value:", mw_h4.pvalue)
print("Rank-biserial effect:", rank_effect)

print("\n--- H4 DECISION ---")

if welch.pvalue < 0.05:
    print("Welch: Reject H0")
else:
    print("Welch: Fail to reject H0")

if mw_h4.pvalue < 0.05:
    print("Mann-Whitney: Reject H0")
else:
    print("Mann-Whitney: Fail to reject H0")

print("\nH4 completed.")


# ==========================================
# FINAL RESULTS - HOLM CORRECTION
# ==========================================

from pathlib import Path
from statsmodels.stats.multitest import multipletests

print("\n========== FINAL HYPOTHESIS SUMMARY ==========")

# Primary tests only: H1, H2 Welch, H3
primary_pvalues = [
    p_value,          # H1
    t_result.pvalue,  # H2 Welch
    p_h3              # H3
]

rejected, adjusted_p, _, _ = multipletests(
    primary_pvalues,
    alpha=0.05,
    method="holm"
)

summary = pd.DataFrame({
    "Hypothesis": ["H1", "H2", "H3", "H4"],
    "Test": [
        "Chi-square: Manufacturer",
        "Welch t-test: Age",
        "Chi-square: Sex",
        "Welch t-test: Symptom count"
    ],
    "Sample_size": [
        int(n),
        int(n1 + n0),  # Updated below for H2
        int(n_h3),
        int(len(h4))
    ],
    "Statistic": [
        chi2,
        t_result.statistic,
        chi2_h3,
        welch.statistic
    ],
    "Raw_p": [
        p_value,
        t_result.pvalue,
        p_h3,
        welch.pvalue
    ],
    "Holm_p": [
        adjusted_p[0],
        adjusted_p[1],
        adjusted_p[2],
        float("nan")  # H4 is exploratory
    ],
    "Effect_size": [
        cramers_v,
        cohens_d,
        cramers_v_h3,
        cohens_d_h4
    ],
    "Effect_measure": [
        "Cramer's V",
        "Cohen's d",
        "Cramer's V",
        "Cohen's d"
    ]
})

# Explicitly set the sample sizes for each test
summary["Sample_size"] = [
    len(h1),
    len(h2),
    len(h3),
    len(h4)
]

print(summary.to_string(index=False))

# Save results
output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "chamoth_hypothesis_results.csv"
summary.to_csv(output_path, index=False)

print("\nHolm-adjusted primary decisions:")
for hypothesis, decision in zip(
    ["H1", "H2", "H3"], rejected
):
    print(
        hypothesis,
        "Reject H0" if decision else "Fail to reject H0"
    )

print("\nH4 is exploratory; no Holm adjustment applied.")
print("Results saved to:", output_path)


# ==========================================
# STEP 9 - STATISTICAL VISUALIZATIONS
# ==========================================

import matplotlib.pyplot as plt
import numpy as np

figure_dir = ROOT / "figures"
figure_dir.mkdir(exist_ok=True)

# Figure 1: Manufacturer flag percentages
manufacturer_rates = 100 * table[1] / table.sum(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
manufacturer_rates.plot.bar(ax=ax, color="#4263a0")
ax.set_title("Recorded Flag Percentage by Manufacturer")
ax.set_ylabel("Flagged reports (%)")
ax.set_xlabel("Manufacturer")
ax.tick_params(axis="x", rotation=20)
ax.set_ylim(0, 35)

for i, value in enumerate(manufacturer_rates):
    ax.text(i, value + 0.5, f"{value:.2f}%",
            ha="center")

fig.tight_layout()
fig.savefig(
    figure_dir / "chamoth_h1_manufacturer.png",
    dpi=200
)
plt.close(fig)

# Figure 2: Age distribution by flag recording
fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(
    not_flagged_age,
    bins=40,
    density=True,
    alpha=0.55,
    label="No selected Yes recorded"
)
ax.hist(
    flagged_age,
    bins=40,
    density=True,
    alpha=0.55,
    label="Hospitalization or death Yes"
)

ax.set_title("Recorded Age Distributions")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Density")
ax.legend()

fig.tight_layout()
fig.savefig(
    figure_dir / "chamoth_h2_age.png",
    dpi=200
)
plt.close(fig)

# Figure 3: Flag percentages by sex
sex_rates = 100 * table_h3[1] / table_h3.sum(axis=1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(
    ["Female", "Male"],
    sex_rates.values,
    color=["#4263a0", "#d08b37"]
)

for i, value in enumerate(sex_rates):
    ax.text(i, value + 0.4, f"{value:.2f}%",
            ha="center")

ax.set_title("Recorded Flag Percentage by Sex")
ax.set_ylabel("Flagged reports (%)")
ax.set_ylim(0, 25)

fig.tight_layout()
fig.savefig(
    figure_dir / "chamoth_h3_sex.png",
    dpi=200
)
plt.close(fig)

# Figure 4: Symptom-count distribution
fig, ax = plt.subplots(figsize=(8, 5))

ax.boxplot(
    [death_unknown, death_yes],
    tick_labels=[
        "Death Unknown",
        "Death Yes"
    ],
    showfliers=False
)

ax.set_title("Recorded Symptom Counts by Death Flag")
ax.set_ylabel("Number of reported symptom terms")

fig.tight_layout()
fig.savefig(
    figure_dir / "chamoth_h4_symptoms.png",
    dpi=200
)
plt.close(fig)

print("\nFour statistical figures saved to:")
print(figure_dir)


# ==========================================
# STEP 11 - SYMPTOM COUNT VALIDATION
# ==========================================

import json
import numpy as np

print("\n========== SYMPTOM VALIDATION ==========")

def count_symptom_terms(value):
    if not isinstance(value, str):
        return np.nan

    try:
        symptoms = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return np.nan

    if not isinstance(symptoms, list):
        return np.nan

    return len(symptoms)


audit = primary[
    ["VAERS_ID", "ORDER", "SYMPTOMS", "SYMPTOM_COUNT"]
].copy()

audit["PARSED_COUNT"] = (
    audit["SYMPTOMS"].apply(count_symptom_terms)
)

audit["REPORTED_COUNT"] = pd.to_numeric(
    audit["SYMPTOM_COUNT"],
    errors="coerce"
)

invalid = audit["PARSED_COUNT"].isna()

mismatches = audit.loc[
    ~invalid &
    audit["PARSED_COUNT"].ne(audit["REPORTED_COUNT"])
]

print("Total primary reports:", len(audit))
print("Unparseable symptom lists:", invalid.sum())
print("Count mismatches:", len(mismatches))

print("\nLargest reported symptom counts:")
print(
    audit.nlargest(5, "REPORTED_COUNT")[
        ["VAERS_ID", "ORDER", "REPORTED_COUNT", "PARSED_COUNT"]
    ].to_string(index=False)
)

if len(mismatches) > 0:
    print("\nExample mismatches:")
    print(
        mismatches[
            ["VAERS_ID", "REPORTED_COUNT", "PARSED_COUNT"]
        ].head(10).to_string(index=False)
    )

if invalid.sum() == 0 and len(mismatches) == 0:
    print("\nPASS: All primary symptom counts match their lists.")
else:
    print("\nCHECK: Some symptom records require investigation.")


# ==========================================
# STEP 13 - CORRECT COHORT COMPARISON
# ==========================================

print("\n========== CORRECT COHORT COMPARISON ==========")

# Original ORDER = 1 selection
order1 = df.loc[
    pd.to_numeric(df["ORDER"], errors="coerce").eq(1)
].copy()

# Lowest ORDER per ID is already stored in primary
lowest = primary.copy()

def get_flag(data):
    return (
        data[["HOSPITAL_STATUS", "DIED_STATUS"]]
        .astype("string")
        .apply(lambda col: col.str.strip().str.lower())
        .eq("yes")
        .any(axis=1)
        .astype(int)
    )

order1["FLAG_YES"] = get_flag(order1)

def age_difference(data):
    age = pd.to_numeric(data["AGE_YRS"], errors="coerce")
    return (
        age[data["FLAG_YES"] == 1].mean()
        - age[data["FLAG_YES"] == 0].mean()
    )

extra = lowest.loc[
    ~lowest["VAERS_ID"].isin(order1["VAERS_ID"])
]

print("ORDER = 1 reports:", len(order1))
print("Lowest ORDER reports:", len(lowest))
print("Additional IDs:", len(extra))
print("Additional flagged reports:", extra["FLAG_YES"].sum())

print("\nORDER = 1 flagged percentage:",
      round(100 * order1["FLAG_YES"].mean(), 3))

print("Lowest ORDER flagged percentage:",
      round(100 * lowest["FLAG_YES"].mean(), 3))

print("\nORDER = 1 mean age difference:",
      round(age_difference(order1), 3))

print("Lowest ORDER mean age difference:",
      round(age_difference(lowest), 3))

print("\nCohort comparison completed.")
