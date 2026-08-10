# Module 1 - Data Pipeline

Scrapes book listings from books.toscrape.com (a public scraping-practice
site - no login or API key required), cleans and types the data, converts
pricing to INR using a fixed project baseline rate, loads it into a
normalized SQLite database, and runs a set of SQL + pandas queries against
it.

## Pipeline stages

| Script       | Input                  | Output                  | Purpose                                    |
|--------------|-------------------------|--------------------------|---------------------------------------------|
| scraper.py   | books.toscrape.com      | raw/books_raw.csv        | Scrape raw (untyped) book listings          |
| clean.py     | raw/books_raw.csv       | raw/books_clean.csv      | Type conversion, cleaning, INR conversion   |
| load_db.py   | raw/books_clean.csv     | books.db                 | Build normalized SQLite schema and load data|
| queries.py   | books.db                | query_output.txt         | Run required SQL queries + pandas comparison|

## Install and run

From the data_pipeline/ folder, with a virtual environment active:

    pip install requests beautifulsoup4 pandas
    python scraper.py
    python clean.py
    python load_db.py
    $env:PYTHONIOENCODING="utf-8"
    python queries.py > query_output.txt

Each script can be re-run independently - load_db.py rebuilds books.db
from scratch each time, and queries.py only reads from it.

## Data source

books.toscrape.com - a public site built specifically for scraping
practice. No login, no API key, no rate limits imposed. scraper.py walks
the site's own category sidebar (rather than hardcoding category slugs), so
it adapts automatically if the site's category list changes.

The final dataset: 69 books across 3 categories (Travel, Mystery,
Historical Fiction), comfortably above the required minimum of 60 rows / 3
categories.

## Cleaning and typing decisions

- rating (int, 1-5): the raw star rating is scraped as a text word
  ("One" .. "Five"). Rows where this text doesn't map to one of the five
  known words are DROPPED, not imputed - a star rating is a discrete
  label, and inventing a plausible-looking rating for a specific book would
  misrepresent that book's actual (unknown) rating. No rows were dropped for
  this reason in the current dataset.
- price_gbp (float): raw price text (e.g. "GBP 51.77") has the currency
  symbol stripped and is parsed to float. If a price fails to parse, it is
  MEDIAN-IMPUTED using the median of all successfully parsed prices -
  price is a continuous numeric field, so median imputation is a standard,
  defensible way to keep the row without distorting the price distribution.
  No rows required imputation in the current dataset.
- in_stock (bool): derived from the raw availability text - True if it
  contains "In stock" (case-insensitive), else False.
- price_inr (float): computed as price_gbp * 105.50. This 105.50 rate is a
  fixed, project-defined constant specified by the assignment brief - it is
  not a live or historical market exchange rate, requires no API call, and
  has no date reference.

### Known minor data-quality note

One scraped title ("Full Moon over Noah's Ark: An Odyssey to Mount Ararat
and Beyond") contains a mis-decoded apostrophe character due to an encoding
mismatch between the source page and the local terminal's default encoding
when writing query output to file. This is cosmetic only - it does not
affect any typed column, join, or query result, and "Row sets identical:
True" in queries.py's output confirms the SQL and pandas pipelines still
agree exactly. Running scripts with $env:PYTHONIOENCODING="utf-8" set
avoids the encode error when redirecting output to a file on Windows.

## Database schema

Two tables, normalized, with a primary/foreign key relationship:

    CREATE TABLE categories (
        category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );

    CREATE TABLE books (
        book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        title       TEXT NOT NULL,
        price_gbp   REAL NOT NULL,
        price_inr   REAL NOT NULL,
        rating      INTEGER NOT NULL,
        in_stock    INTEGER NOT NULL,
        category_id INTEGER NOT NULL REFERENCES categories(category_id)
    );

## SQL queries (queries.py)

Five queries are run against the database, collectively covering every
required clause:

1. SELECT / WHERE - in-stock books priced under GBP 20
2. ORDER BY + LIMIT - 10 most expensive books
3. DISTINCT - list of category names
4. IN + BETWEEN - 4-5 star books priced between GBP 10 and GBP 40
5. JOIN (+ window function) - top 5 highest-rated books per category, using
   ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY rating DESC,
   price_gbp DESC) to correctly rank within each category (a naive
   COUNT-based threshold breaks when more than 5 books tie on the same
   rating - this was caught and fixed during development)

Full output of all five queries is saved to query_output.txt.

### pandas equivalence check

- Query 1 and Query 4 are re-read via pd.read_sql(...) to demonstrate SQL
  results loading directly into pandas.
- Query 5's JOIN is independently reproduced using pd.merge() on the
  in-memory books and categories DataFrames (no SQL), with matching
  sort/tie-break logic (rating DESC, then price_gbp DESC, top 5 per
  category). The script asserts the two row sets are identical and prints
  "Row sets identical: True", confirming the SQL and pandas approaches
  produce equivalent output.

## Repository files

- scraper.py, clean.py, load_db.py, queries.py - pipeline scripts
- raw/books_raw.csv - raw scraped data
- raw/books_clean.csv - cleaned, typed data
- books.db - SQLite database (also fully regenerable via load_db.py)
- query_output.txt - full output of all five queries + pandas comparison
