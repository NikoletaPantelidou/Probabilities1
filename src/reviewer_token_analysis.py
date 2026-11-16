
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_DIR / "output_with_token_counts" / "lmmg"
)

OUTPUT_DIR = (
    PROJECT_DIR / "output" / "reviewer_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set this to True ONLY if the BOS token itself is included
# in the number of tokens whose log-probabilities are summed.
# If BOS is only supplied as context, leave this False.
BOS_IS_SCORED_TOKEN = False


# ============================================================
# 2. LOAD THE EIGHT MODEL RESULT FILES
# ============================================================

files = sorted(INPUT_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(
        f"No CSV files found in: {INPUT_DIR}"
    )

all_data = []

for file in files:

    df = pd.read_csv(file)

    required_columns = [
        "direct_sum_logprob_sentence1",
        "direct_sum_logprob_sentence2",
        "token_count_sentence1",
        "token_count_sentence2",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{file.name} is missing columns: {missing}"
        )

    if "model" not in df.columns:
        df["model"] = file.stem
    else:
        df["model"] = df["model"].fillna(file.stem)

    df["source_file"] = file.name

    all_data.append(df)


data = pd.concat(all_data, ignore_index=True)

print("\nFiles loaded:")
for file in files:
    print(" -", file.name)

print(f"\nTotal observations: {len(data)}")
print("Models:", sorted(data["model"].astype(str).unique()))


# ============================================================
# 3. CHECK AND CLEAN THE DATA
# ============================================================

required_columns = [
    "direct_sum_logprob_sentence1",
    "direct_sum_logprob_sentence2",
    "token_count_sentence1",
    "token_count_sentence2",
    "model",
]

data = data.dropna(subset=required_columns).copy()

numeric_columns = [
    "direct_sum_logprob_sentence1",
    "direct_sum_logprob_sentence2",
    "token_count_sentence1",
    "token_count_sentence2",
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=numeric_columns).copy()

if (
    (data["token_count_sentence1"] <= 0).any()
    or (data["token_count_sentence2"] <= 0).any()
):
    raise ValueError("Token counts must be positive.")

if BOS_IS_SCORED_TOKEN:
    data["token_count_sentence1"] += 1
    data["token_count_sentence2"] += 1

# Preserve the original item identifiers whenever available.

if "item_id" in data.columns:
    data["item_id"] = data["item_id"].astype(str)

elif "sentence1" in data.columns and "sentence2" in data.columns:
    data["item_id"] = (
        data["sentence1"].astype(str)
        + " || "
        + data["sentence2"].astype(str)
    )

else:
    raise ValueError(
        "The data need either item_id or sentence1/sentence2."
    )

# Use a single condition label if the CSVs do not contain one.

if "condition" not in data.columns:
    data["condition"] = "unspecified"

data["condition"] = (
    data["condition"].fillna("unspecified").astype(str)
)

data["model"] = data["model"].astype(str)

print("\nObservations after cleaning:", len(data))

# Check for duplicated model/item/condition observations.

duplicate_columns = ["model", "item_id", "condition"]

duplicates = data.duplicated(
    subset=duplicate_columns,
    keep=False
)

if duplicates.any():
    print(
        "\nWARNING: Duplicate model/item/condition rows found:",
        int(duplicates.sum())
    )
    print(
        "Check these before interpreting the statistical results."
    )


# ============================================================
# 4. TOKEN-COUNT DIFFERENCES
# ============================================================

data["token_count_difference"] = (
    data["token_count_sentence1"]
    - data["token_count_sentence2"]
)

data["absolute_token_count_difference"] = (
    data["token_count_difference"].abs()
)

data["mean_token_count"] = (
    data["token_count_sentence1"]
    + data["token_count_sentence2"]
) / 2

group_columns = ["model", "condition"]

token_summary = (
    data.groupby(group_columns, dropna=False)
    .agg(
        n=("item_id", "size"),

        mean_tokens_s1=(
            "token_count_sentence1", "mean"
        ),

        sd_tokens_s1=(
            "token_count_sentence1", "std"
        ),

        mean_tokens_s2=(
            "token_count_sentence2", "mean"
        ),

        sd_tokens_s2=(
            "token_count_sentence2", "std"
        ),

        mean_signed_difference=(
            "token_count_difference", "mean"
        ),

        sd_signed_difference=(
            "token_count_difference", "std"
        ),

        mean_absolute_difference=(
            "absolute_token_count_difference", "mean"
        ),

        percentage_unequal=(
            "absolute_token_count_difference",
            lambda x: 100 * (x > 0).mean()
        ),
    )
    .reset_index()
)

token_summary.to_csv(
    OUTPUT_DIR / "token_count_summary.csv",
    index=False
)


# ============================================================
# 5. ORIGINAL AND PER-TOKEN LOG-PROBABILITIES
# ============================================================

# Original summed log-probability difference.
# Positive values favour sentence 1.
# Negative values favour sentence 2.

data["original_difference"] = (
    data["direct_sum_logprob_sentence1"]
    - data["direct_sum_logprob_sentence2"]
)

# Mean log-probability per token for each sentence.

data["mean_logprob_s1"] = (
    data["direct_sum_logprob_sentence1"]
    / data["token_count_sentence1"]
)

data["mean_logprob_s2"] = (
    data["direct_sum_logprob_sentence2"]
    / data["token_count_sentence2"]
)

# Difference between per-token scores.
# Positive values favour sentence 1.

data["normalized_difference"] = (
    data["mean_logprob_s1"]
    - data["mean_logprob_s2"]
)

# Save the full dataset with the additional variables.

data.to_csv(
    OUTPUT_DIR / "sentence_level_normalized_scores.csv",
    index=False
)


# ============================================================
# 6. SUMMARIZE ORIGINAL AND NORMALIZED SCORES
# ============================================================

score_summary = (
    data.groupby(group_columns, dropna=False)
    .agg(
        n=("item_id", "size"),

        mean_original_difference=(
            "original_difference", "mean"
        ),

        sd_original_difference=(
            "original_difference", "std"
        ),

        mean_normalized_difference=(
            "normalized_difference", "mean"
        ),

        sd_normalized_difference=(
            "normalized_difference", "std"
        ),

        mean_sentence1_tokens=(
            "token_count_sentence1", "mean"
        ),

        mean_sentence2_tokens=(
            "token_count_sentence2", "mean"
        ),
    )
    .reset_index()
)

score_summary.to_csv(
    OUTPUT_DIR / "original_vs_normalized_summary.csv",
    index=False
)


# ============================================================
# 7. CHECK WHETHER NORMALIZATION CHANGES PREFERENCES
# ============================================================

# A positive difference means sentence 1 is preferred.
# A negative difference means sentence 2 is preferred.
# A zero means a tie.

data["original_preference"] = np.sign(
    data["original_difference"]
)

data["normalized_preference"] = np.sign(
    data["normalized_difference"]
)

preference_results = []

for (model, condition), group in data.groupby(
    group_columns,
    dropna=False
):

    original = group["original_preference"]
    normalized = group["normalized_preference"]

    # Exclude ties from agreement calculations.

    non_ties = (
        (original != 0)
        & (normalized != 0)
    )

    comparable = group.loc[non_ties]

    n_comparable = len(comparable)

    if n_comparable > 0:
        agreement = (
            comparable["original_preference"].values
            == comparable["normalized_preference"].values
        ).mean() * 100
    else:
        agreement = np.nan

    reversals = (
        (original != 0)
        & (normalized != 0)
        & (original != normalized)
    ).sum()

    original_ties = (original == 0).sum()
    normalized_ties = (normalized == 0).sum()

    # McNemar test:
    # Does normalization systematically change the
    # direction of sentence preference?

    table = [
        [
            int((
                (original == 1)
                & (normalized == 1)
            ).sum()),

            int((
                (original == 1)
                & (normalized == -1)
            ).sum()),
        ],
        [
            int((
                (original == -1)
                & (normalized == 1)
            ).sum()),

            int((
                (original == -1)
                & (normalized == -1)
            ).sum()),
        ],
    ]

    discordant = table[0][1] + table[1][0]

    if discordant > 0:
        test_result = mcnemar(
            table,
            exact=True
        )
        mcnemar_p = test_result.pvalue
    else:
        mcnemar_p = np.nan

    preference_results.append({
        "model": model,
        "condition": condition,
        "n_total": len(group),
        "n_comparable_non_ties": n_comparable,
        "agreement_percentage": agreement,
        "n_preference_reversals": int(reversals),
        "percentage_preference_reversals": (
            100 * reversals / len(group)
        ),
        "n_original_ties": int(original_ties),
        "n_normalized_ties": int(normalized_ties),
        "mcnemar_p": mcnemar_p,
    })


preference_summary = pd.DataFrame(preference_results)

preference_summary.to_csv(
    OUTPUT_DIR / "preference_reversals.csv",
    index=False
)


# ============================================================
# 8. REGRESSION: DO TOKEN COUNTS PREDICT SCORE DIFFERENCES?
# ============================================================

# This pooled model estimates whether token-count differences
# are associated with the original score differences.
#
# Model and condition are included as fixed effects.
# Standard errors are clustered by stimulus item.

regression_results = []

for outcome in [
    "original_difference",
    "normalized_difference",
]:

    predictors = [
        "token_count_difference",
        "mean_token_count",
        "C(model)",
        "C(condition)",
    ]

    formula = (
        outcome
        + " ~ "
        + " + ".join(predictors)
    )

    analysis_data = data.dropna(
        subset=[
            outcome,
            "token_count_difference",
            "mean_token_count",
            "item_id",
        ]
    ).copy()

    if analysis_data["item_id"].nunique() < 2:
        print(
            f"Skipping {outcome}: "
            "not enough stimulus clusters."
        )
        continue

    try:
        result = smf.ols(
            formula,
            data=analysis_data
        ).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": analysis_data["item_id"]
            }
        )

        print("\n====================================")
        print("REGRESSION:", outcome)
        print("====================================")
        print(result.summary())

        with open(
            OUTPUT_DIR / f"{outcome}_covariate_model.txt",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(result.summary().as_text())

        for term in result.params.index:
            regression_results.append({
                "outcome": outcome,
                "predictor": term,
                "coefficient": result.params[term],
                "standard_error": result.bse[term],
                "p_value": result.pvalues[term],
                "ci_lower": result.conf_int().loc[term, 0],
                "ci_upper": result.conf_int().loc[term, 1],
                "n_observations": int(result.nobs),
                "r_squared": result.rsquared,
            })

    except Exception as e:
        print(
            f"Could not fit regression for {outcome}: {e}"
        )


if regression_results:
    pd.DataFrame(regression_results).to_csv(
        OUTPUT_DIR / "token_count_regression_results.csv",
        index=False
    )


# ============================================================
# 9. OPTIONAL: ITEM-FIXED-EFFECTS ROBUSTNESS MODEL
# ============================================================

# This additional model controls for stable differences
# between stimulus pairs, while retaining token counts
# and model identity as predictors.

item_fe_results = []

for outcome in [
    "original_difference",
    "normalized_difference",
]:

    formula = (
        f"{outcome} ~ "
        "token_count_difference "
        "+ mean_token_count "
        "+ C(model) "
        "+ C(item_id)"
    )

    analysis_data = data.dropna(
        subset=[
            outcome,
            "token_count_difference",
            "mean_token_count",
            "item_id",
        ]
    ).copy()

    if analysis_data["item_id"].nunique() < 2:
        continue

    try:
        result = smf.ols(
            formula,
            data=analysis_data
        ).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": analysis_data["item_id"]
            }
        )

        print("\n====================================")
        print("ITEM FIXED EFFECTS:", outcome)
        print("====================================")
        print(result.summary())

        with open(
            OUTPUT_DIR / f"{outcome}_item_fixed_effects.txt",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(result.summary().as_text())

        for term in [
            "token_count_difference",
            "mean_token_count",
        ]:
            if term in result.params.index:
                item_fe_results.append({
                    "outcome": outcome,
                    "predictor": term,
                    "coefficient": result.params[term],
                    "standard_error": result.bse[term],
                    "p_value": result.pvalues[term],
                    "ci_lower": result.conf_int().loc[term, 0],
                    "ci_upper": result.conf_int().loc[term, 1],
                    "n_observations": int(result.nobs),
                    "r_squared": result.rsquared,
                })

    except Exception as e:
        print(
            f"Could not fit item fixed-effects model "
            f"for {outcome}: {e}"
        )


if item_fe_results:
    pd.DataFrame(item_fe_results).to_csv(
        OUTPUT_DIR / "item_fixed_effects_results.csv",
        index=False
    )


# ============================================================
# 10. FINISH
# ============================================================

print("\n====================================")
print("REVIEWER TOKEN ANALYSIS COMPLETED")
print("====================================")

print(f"Input folder: {INPUT_DIR}")
print(f"Output folder: {OUTPUT_DIR}")

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    if file.is_file():
        print(" -", file.name)