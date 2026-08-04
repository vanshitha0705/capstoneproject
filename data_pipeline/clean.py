"""
clean.py — Module 1: Data Pipeline
Reads raw/books_raw.csv (untyped, scraped text) and produces a cleaned,
properly typed dataset: raw/books_clean.csv

Cleaning / typing decisions (see README for the full write-up):

1. star_rating (text "One".."Five") -> rating (int 1-5)
   If a row's rating text doesn't map to one of the five known words,
   the ROW IS DROPPED. Rating is a discrete label, not a continuous
   quantity, so imputing a fake rating would misrepresent that specific
   book. Dropping preserves correctness of the remaining data.

2. price_raw (e.g. "£51.77") -> price_gbp (float)
   If a price fails to parse (unexpected format), it is MEDIAN-IMPUTED
   using the median of all successfully parsed prices. Price is a
   continuous numeric field, so median imputation is a standard,
   defensible way to keep the row without distorting the distribution.

3. availability_raw (e.g. "In stock") -> in_stock (bool)
   True if the raw text contains "In stock" (case-insensitive), else False.

4. price_gbp -> price_inr, using the project's REQUIRED FIXED BASELINE RATE:
       1 GBP = 105.50 INR
   This is an artificial, project-defined constant (not a live/historical
   market rate) and requires no API call or network access.

Run:
    python clean.py
"""

import statistics
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).parent / "raw" / "books_raw.csv"
CLEAN_PATH = Path(__file__).parent / "raw" / "books_clean.csv"

GBP_TO_INR_RATE = 105.50  # fixed project-defined constant, per assignment spec

RATING_WORD_TO_INT = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def parse_price(price_raw: str):
    """Strip currency symbol and parse to float. Returns None on failure."""
    if not price_raw:
        return None
    cleaned = price_raw.strip()
    # Strip any leading non-numeric currency symbol(s), e.g. "£"
    cleaned = "".join(ch for ch in cleaned if ch.isdigit() or ch == ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_availability(availability_raw: str) -> bool:
    """True if the raw text indicates the book is in stock."""
    if not availability_raw:
        return False
    return "in stock" in availability_raw.lower()


def main():
    df = pd.read_csv(RAW_PATH)
    original_count = len(df)
    print(f"Loaded {original_count} raw rows from {RAW_PATH}")

    # --- Rating: map text -> int, drop unmappable rows ---
    df["rating"] = df["star_rating_raw"].map(RATING_WORD_TO_INT)
    dropped_rating_mask = df["rating"].isna()
    n_dropped_rating = int(dropped_rating_mask.sum())
    if n_dropped_rating:
        print(f"Dropping {n_dropped_rating} row(s) with unparseable star rating.")
    df = df[~dropped_rating_mask].copy()
    df["rating"] = df["rating"].astype(int)

    # --- Price: parse to float, median-impute failures ---
    df["price_gbp"] = df["price_raw"].apply(parse_price)
    n_bad_price = int(df["price_gbp"].isna().sum())
    if n_bad_price:
        median_price = statistics.median(df["price_gbp"].dropna().tolist())
        print(
            f"Median-imputing {n_bad_price} row(s) with unparseable price "
            f"(median = {median_price:.2f})."
        )
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    # --- Availability: parse to bool ---
    df["in_stock"] = df["availability_raw"].apply(parse_availability)

    # --- INR conversion: fixed baseline rate ---
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

    # Final tidy column set
    final_cols = ["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]
    df_final = df[final_cols].reset_index(drop=True)

    df_final.to_csv(CLEAN_PATH, index=False)

    print(f"\nRows in:  {original_count}")
    print(f"Rows out: {len(df_final)}")
    print(f"Saved cleaned dataset to {CLEAN_PATH}")
    print("\nPreview:")
    print(df_final.head())


if __name__ == "__main__":
    main()