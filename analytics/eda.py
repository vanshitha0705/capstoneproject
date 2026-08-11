"""
eda.py -- Module 2, Part A: Profiling, cleaning, and the data story

Loads the Titanic dataset via sns.load_dataset('titanic') -- the ONE AND
ONLY raw load for the whole module -- profiles it, cleans it per the
missing-value threshold rule, saves the cleaned frame to titanic.csv (the
offline fallback), and produces the full EDA story: univariate, bivariate,
correlation heatmap, >=4 multivariate charts, and an exploratory
standardization sanity check.

All charts are saved as PNGs under analytics/charts/ (scripts can't pop up
interactive plot windows one after another cleanly, so this is the more
reliable option). Every printed number below is meant to be copied into
the README's written interpretations.

Run:
    python eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, we only save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CHARTS_DIR = Path(__file__).parent / "charts"
CSV_PATH = Path(__file__).parent / "titanic.csv"

sns.set_theme(style="whitegrid")


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def iqr_outlier_count(series: pd.Series) -> tuple[int, float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers), lower, upper


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load (the ONE raw load for the whole module) and profile
    # ------------------------------------------------------------------
    section("Loading dataset via sns.load_dataset('titanic')")
    df = sns.load_dataset("titanic")

    section("df.info()")
    df.info()

    section("df.describe()")
    print(df.describe(include="all"))

    section("df.shape")
    print(df.shape)

    section("Missing values per column (count and %)")
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_report = pd.DataFrame(
        {"missing_count": missing_counts, "missing_pct": missing_pct}
    )
    missing_report = missing_report[missing_report["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )
    print(missing_report)

    # ------------------------------------------------------------------
    # 2. Missing-value handling per the threshold rule:
    #    <5% missing  -> drop those rows
    #    5-30% missing -> impute
    #    very high%   -> drop column OR encode "missing" as its own category
    # ------------------------------------------------------------------
    section("Applying missing-value threshold rule")
    for col, row in missing_report.iterrows():
        print(f"  {col}: {row['missing_pct']}% missing")

    # embarked / embark_town: ~0.22% missing -> under 5% -> drop those rows
    before = len(df)
    df = df.dropna(subset=["embarked", "embark_town"])
    print(
        f"\nDropped {before - len(df)} row(s) with missing embarked/embark_town "
        f"(missing rate was under 5%, so per the threshold rule we drop rows "
        f"rather than impute)."
    )

    # age: ~19.9% missing -> within 5-30% -> impute (median, robust to the
    # right-skew we'll confirm below)
    age_missing_pct = missing_report.loc["age", "missing_pct"] if "age" in missing_report.index else 0
    age_median = df["age"].median()
    df["age"] = df["age"].fillna(age_median)
    print(
        f"\nImputed 'age' missing values ({age_missing_pct}% missing, within "
        f"the 5-30% band) with the median age ({age_median}). Median is used "
        f"instead of mean because age/fare-like distributions are typically "
        f"skewed, and the median is more robust to that skew than the mean."
    )

    # deck: ~77% missing -> far too high to impute reliably -> DROP THE COLUMN
    deck_missing_pct = missing_report.loc["deck", "missing_pct"] if "deck" in missing_report.index else None
    if "deck" in df.columns:
        df = df.drop(columns=["deck"])
        print(
            f"\nDropped the 'deck' column entirely ({deck_missing_pct}% missing). "
            f"At this missing rate, imputation would be mostly fabricating "
            f"values rather than recovering them, and deck is largely "
            f"redundant with pclass/fare (higher class cabins map to known "
            f"deck letters) -- so little unique signal is lost by dropping it."
        )

    section("Missing values after cleaning (should be empty)")
    print(df.isna().sum()[df.isna().sum() > 0])

    # ------------------------------------------------------------------
    # 3. Save the ONE committed offline fallback
    # ------------------------------------------------------------------
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved cleaned dataset to {CSV_PATH} (offline fallback for grading).")

    # ------------------------------------------------------------------
    # 4. Univariate analysis: age and fare
    # ------------------------------------------------------------------
    section("Univariate analysis: age and fare")

    for col in ["age", "fare"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f"{col} - histogram")
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"{col} - box plot")
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"univariate_{col}.png", dpi=120)
        plt.close(fig)

        n_out, lower, upper = iqr_outlier_count(df[col])
        print(
            f"{col}: IQR outlier bounds = [{lower:.2f}, {upper:.2f}] -> "
            f"{n_out} outlier(s) by the 1.5*IQR rule"
        )

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode().iloc[0]
    print(f"\nfare: mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
    if fare_mean > fare_median > fare_mode:
        skew_note = "RIGHT-SKEWED (mean > median > mode) -- a long tail of high fares pulls the mean upward."
    elif fare_mean < fare_median < fare_mode:
        skew_note = "LEFT-SKEWED (mean < median < mode)."
    else:
        skew_note = "roughly SYMMETRIC (mean, median, mode are close together)."
    print(f"fare distribution is {skew_note}")

    # ------------------------------------------------------------------
    # 5. Bivariate analysis: survival rate via boolean masking
    # ------------------------------------------------------------------
    section("Bivariate analysis: survival rate")

    print("Survival rate by sex:")
    for sex_value in df["sex"].unique():
        mask = df["sex"] == sex_value
        rate = df.loc[mask, "survived"].mean()
        print(f"  {sex_value}: {rate:.3f} (n={mask.sum()})")

    print("\nSurvival rate by pclass:")
    for pclass_value in sorted(df["pclass"].unique()):
        mask = df["pclass"] == pclass_value
        rate = df.loc[mask, "survived"].mean()
        print(f"  class {pclass_value}: {rate:.3f} (n={mask.sum()})")

    print("\nSurvival rate by sex AND pclass:")
    for sex_value in sorted(df["sex"].unique()):
        for pclass_value in sorted(df["pclass"].unique()):
            mask = (df["sex"] == sex_value) & (df["pclass"] == pclass_value)
            rate = df.loc[mask, "survived"].mean()
            print(f"  sex={sex_value}, class={pclass_value}: {rate:.3f} (n={mask.sum()})")

    # ------------------------------------------------------------------
    # 6. Correlation heatmap on exactly the 6 specified columns
    # ------------------------------------------------------------------
    section("Correlation matrix (6 columns): survived, pclass, age, sibsp, parch, fare")

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[corr_cols].corr()
    print(corr)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (6 numeric columns)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    # Rank all off-diagonal pairs by |correlation|, take top 2
    pairs = []
    for i, c1 in enumerate(corr_cols):
        for j, c2 in enumerate(corr_cols):
            if i < j:
                pairs.append((c1, c2, corr.loc[c1, c2]))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    print("\nTop 2 strongest correlations (by absolute value):")
    for c1, c2, val in pairs[:2]:
        print(f"  {c1} <-> {c2}: {val:.3f}")

    # ------------------------------------------------------------------
    # 7. Multivariate "data story" -- 4+ charts, each interpreted in README
    # ------------------------------------------------------------------
    section("Multivariate charts (saved to charts/ -- interpretations go in README)")

    # Chart 1: survival rate by class and sex (grouped bar)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df, x="pclass", y="survived", hue="sex", ax=ax, errorbar=None)
    ax.set_title("Survival rate by passenger class and sex")
    ax.set_ylabel("Survival rate")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "multivariate_1_survival_by_class_sex.png", dpi=120)
    plt.close(fig)

    # Chart 2: age distribution by survival outcome (box plot)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="survived", y="age", ax=ax)
    ax.set_title("Age distribution by survival outcome")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Did not survive", "Survived"])
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "multivariate_2_age_by_survival.png", dpi=120)
    plt.close(fig)

    # Chart 3: fare vs age scatter, colored by survival
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6, ax=ax)
    ax.set_title("Fare vs age, colored by survival")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "multivariate_3_fare_vs_age.png", dpi=120)
    plt.close(fig)

    # Chart 4: family size (sibsp+parch) vs survival rate
    df["family_size"] = df["sibsp"] + df["parch"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df, x="family_size", y="survived", ax=ax, errorbar=None)
    ax.set_title("Survival rate by family size (sibsp + parch)")
    ax.set_ylabel("Survival rate")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "multivariate_4_survival_by_family_size.png", dpi=120)
    plt.close(fig)

    print(f"Saved 4 multivariate charts to {CHARTS_DIR}")

    # ------------------------------------------------------------------
    # 8. Exploratory standardization check (EDA-only, NOT used in modeling)
    # ------------------------------------------------------------------
    section("Exploratory z-score standardization check (age, fare) -- EDA only")

    for col in ["age", "fare"]:
        before_mean, before_std = df[col].mean(), df[col].std()
        z = (df[col] - before_mean) / before_std
        print(
            f"{col}: before -> mean={before_mean:.2f}, std={before_std:.2f} | "
            f"after z-score -> mean={z.mean():.4f}, std={z.std():.4f}"
        )

    print(
        "\nNote: this standardization is exploratory only and is NOT carried "
        "into the modeling pipeline in modeling.py -- the modeling pipeline "
        "fits its own StandardScaler on the training split only, to avoid "
        "any leakage from the full dataset."
    )

    section("EDA complete")
    print(f"Cleaned dataset: {df.shape}")
    print(f"Charts saved to: {CHARTS_DIR}")
    print(f"Offline fallback CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
