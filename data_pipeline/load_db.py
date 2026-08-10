"""
load_db.py — Module 1: Data Pipeline
Creates a normalized SQLite database (books.db) from raw/books_clean.csv
and loads the cleaned data into it.

Schema (two tables, PK/FK relationship):

    categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
    books(
        book_id     INTEGER PRIMARY KEY,
        title       TEXT,
        price_gbp   REAL,
        price_inr   REAL,
        rating      INTEGER,
        in_stock    INTEGER,          -- 0/1, SQLite has no native bool
        category_id INTEGER REFERENCES categories(category_id)
    )

Run:
    python load_db.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

CLEAN_PATH = Path(__file__).parent / "raw" / "books_clean.csv"
DB_PATH = Path(__file__).parent / "books.db"

SCHEMA_SQL = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

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
"""


def main():
    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded {len(df)} cleaned rows from {CLEAN_PATH}")

    # Fresh database each run, so this script is fully re-runnable
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)

    # --- Insert categories (unique names) ---
    unique_categories = sorted(df["category"].unique())
    cur.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(name,) for name in unique_categories],
    )
    conn.commit()

    # Build a name -> category_id lookup
    cur.execute("SELECT category_id, category_name FROM categories")
    category_id_by_name = {name: cid for cid, name in cur.fetchall()}

    # --- Insert books ---
    book_rows = [
        (
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            1 if bool(row["in_stock"]) else 0,
            category_id_by_name[row["category"]],
        )
        for _, row in df.iterrows()
    ]
    cur.executemany(
        """
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        book_rows,
    )
    conn.commit()

    # --- Sanity checks ---
    cur.execute("SELECT COUNT(*) FROM categories")
    n_categories = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM books")
    n_books = cur.fetchone()[0]

    print(f"Inserted {n_categories} categories and {n_books} books into {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    main()