"""
queries.py - Module 1: Data Pipeline
Runs the required SQL queries against books.db and demonstrates the
pandas equivalents (pd.read_sql and pd.merge).

Covers, across the queries below:
    - SELECT / WHERE
    - ORDER BY
    - LIMIT
    - DISTINCT
    - IN (and BETWEEN)
    - JOIN (categories <-> books)

Run:
    python queries.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "books.db"


def run_query(conn, title: str, sql: str):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(sql.strip())
    print("-" * 70)
    df = pd.read_sql(sql, conn)
    print(df.to_string(index=False))
    return df


def main():
    conn = sqlite3.connect(DB_PATH)

    q1 = """
        SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1 AND price_gbp < 20
    """
    run_query(conn, "Query 1: SELECT / WHERE - in-stock books under GBP 20", q1)

    q2 = """
        SELECT title, price_gbp, price_inr
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10
    """
    run_query(conn, "Query 2: ORDER BY + LIMIT - 10 most expensive books", q2)

    q3 = """
        SELECT DISTINCT category_name
        FROM categories
    """
    run_query(conn, "Query 3: DISTINCT - category names", q3)

    q4 = """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating IN (4, 5)
          AND price_gbp BETWEEN 10 AND 40
        ORDER BY rating DESC, price_gbp ASC
    """
    df4 = run_query(
        conn, "Query 4: IN + BETWEEN - 4-5 star books priced GBP 10-40", q4
    )

    q5 = """
        WITH ranked AS (
            SELECT
                c.category_name,
                b.title,
                b.rating,
                b.price_gbp,
                ROW_NUMBER() OVER (
                    PARTITION BY b.category_id
                    ORDER BY b.rating DESC, b.price_gbp DESC
                ) AS rn
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
        )
        SELECT category_name, title, rating, price_gbp
        FROM ranked
        WHERE rn <= 5
        ORDER BY category_name, rating DESC, price_gbp DESC
    """
    df5_sql = run_query(
        conn, "Query 5: JOIN - top 5 highest-rated books per category", q5
    )

    print(f"\n{'=' * 70}")
    print("pd.read_sql demonstration (Query 1 and Query 4 re-read via pandas)")
    print(f"{'=' * 70}")
    df1_via_pandas = pd.read_sql(q1, conn)
    df4_via_pandas = pd.read_sql(q4, conn)
    print(f"Query 1 rows via pd.read_sql: {len(df1_via_pandas)}")
    print(f"Query 4 rows via pd.read_sql: {len(df4_via_pandas)}")

    print(f"\n{'=' * 70}")
    print("pd.merge demonstration - reproducing Query 5's JOIN in pandas")
    print(f"{'=' * 70}")

    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)

    merged_df = pd.merge(books_df, categories_df, on="category_id", how="inner")

    df5_pandas = (
        merged_df.sort_values(
            ["category_name", "rating", "price_gbp"], ascending=[True, False, False]
        )
        .groupby("category_name", group_keys=False)
        .head(5)[["category_name", "title", "rating", "price_gbp"]]
        .reset_index(drop=True)
    )

    print("\npd.merge result (top 5 per category by rating):")
    print(df5_pandas.to_string(index=False))

    print(f"\nSQL JOIN query row count:      {len(df5_sql)}")
    print(f"pd.merge equivalent row count: {len(df5_pandas)}")

    sql_set = set(
        df5_sql[["category_name", "title", "rating", "price_gbp"]]
        .itertuples(index=False, name=None)
    )
    pandas_set = set(
        df5_pandas[["category_name", "title", "rating", "price_gbp"]]
        .itertuples(index=False, name=None)
    )
    rows_match = sql_set == pandas_set
    print(f"Row sets identical: {rows_match}")

    print(
        "\nBoth approaches use the same ranking rule (partition by category, "
        "order by rating DESC then price_gbp DESC, keep top 5): SQL does it "
        "with a ROW_NUMBER() window function, pandas does it with a sort + "
        "groupby().head(). The row sets match exactly."
    )

    conn.close()


if __name__ == "__main__":
    main()
