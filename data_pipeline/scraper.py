"""
scraper.py — Module 1: Data Pipeline
Scrapes book listings from books.toscrape.com (a public scraping-practice
site, no login/API key required) across multiple categories.

For each book, captures:
    title, price (raw, e.g. "£51.77"), star_rating (raw text, e.g. "Three"),
    availability (raw text), category

Output: raw/books_raw.csv  (one row per book, uncleaned/untyped —
cleaning + type conversion happens in a separate clean.py step)

Run:
    python scraper.py
"""

import csv
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
MIN_ROWS = 60
MIN_CATEGORIES = 3
REQUEST_DELAY_SECONDS = 0.5  # be polite to the server
OUTPUT_PATH = Path(__file__).parent / "raw" / "books_raw.csv"

session = requests.Session()
session.headers.update(
    {"User-Agent": "Zepto-Capstone-DataPipeline/1.0 (educational scraping project)"}
)


def get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return a parsed BeautifulSoup object."""
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return BeautifulSoup(resp.text, "html.parser")


def get_categories(home_url: str) -> list[tuple[str, str]]:
    """
    Parse the sidebar on the homepage to get every category name + URL.
    Returns a list of (category_name, category_url) tuples, in sidebar order.
    Skips the top-level "Books" (All products) link.
    """
    soup = get_soup(home_url)
    links = soup.select("div.side_categories ul.nav-list ul li a")
    categories = []
    for link in links:
        name = link.get_text(strip=True)
        url = urljoin(home_url, link["href"])
        categories.append((name, url))
    return categories


def scrape_category(category_name: str, category_url: str) -> list[dict]:
    """
    Scrape every book on every page of a single category (follows
    pagination via the "next" link). Returns a list of row dicts.
    """
    rows = []
    next_url = category_url

    while next_url:
        soup = get_soup(next_url)
        articles = soup.select("article.product_pod")

        for art in articles:
            # Use the <a title="..."> attribute — the visible link text
            # is often truncated with "..." in the listing view.
            title_tag = art.select_one("h3 a")
            title = title_tag["title"].strip() if title_tag and title_tag.has_attr("title") else title_tag.get_text(strip=True)

            price_tag = art.select_one("p.price_color")
            price_raw = price_tag.get_text(strip=True) if price_tag else ""

            rating_tag = art.select_one("p.star-rating")
            # class list looks like ["star-rating", "Three"] — the rating
            # word is whichever class isn't "star-rating"
            rating_raw = ""
            if rating_tag:
                classes = rating_tag.get("class", [])
                rating_words = [c for c in classes if c != "star-rating"]
                rating_raw = rating_words[0] if rating_words else ""

            avail_tag = art.select_one("p.instock.availability")
            availability_raw = avail_tag.get_text(strip=True) if avail_tag else ""

            rows.append(
                {
                    "title": title,
                    "price_raw": price_raw,
                    "star_rating_raw": rating_raw,
                    "availability_raw": availability_raw,
                    "category": category_name,
                }
            )

        # Look for a "next" pagination link
        next_link = soup.select_one("li.next a")
        next_url = urljoin(next_url, next_link["href"]) if next_link else None

    return rows


def main():
    print(f"Fetching category list from {BASE_URL} ...")
    categories = get_categories(BASE_URL)
    print(f"Found {len(categories)} categories.")

    all_rows: list[dict] = []
    categories_used = 0

    for name, url in categories:
        print(f"Scraping category: {name} ...")
        cat_rows = scrape_category(name, url)
        all_rows.extend(cat_rows)
        categories_used += 1
        print(f"  -> {len(cat_rows)} books (running total: {len(all_rows)})")

        if len(all_rows) >= MIN_ROWS and categories_used >= MIN_CATEGORIES:
            print(
                f"Reached {len(all_rows)} rows across {categories_used} categories "
                f"— stopping (minimums: {MIN_ROWS} rows / {MIN_CATEGORIES} categories)."
            )
            break

    if len(all_rows) < MIN_ROWS or categories_used < MIN_CATEGORIES:
        print(
            "WARNING: finished scraping all available categories but did not "
            f"reach the minimums (got {len(all_rows)} rows / {categories_used} categories)."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["title", "price_raw", "star_rating_raw", "availability_raw", "category"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows across {categories_used} categories to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()