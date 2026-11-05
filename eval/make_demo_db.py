"""Generate the demo e-commerce SQLite database (fixed random seed, so it is reproducible).

Usage:  python eval/make_demo_db.py
Output: data/demo.db
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_CUSTOMERS = 500
N_ORDERS = 2500
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "demo.db"

FIRST_NAMES = ["Alice", "Bruno", "Chloe", "David", "Emma", "Farid", "Gina", "Hugo", "Ines",
               "Jules", "Khadija", "Louis", "Maya", "Nadia", "Omar", "Paul", "Quentin",
               "Rosa", "Sami", "Tara"]
LAST_NAMES = ["Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand",
              "Leroy", "Moreau", "Simon", "Laurent", "Lefebvre", "Michel", "Garcia"]
COUNTRIES = {"France": 0.5, "Germany": 0.15, "Spain": 0.12, "Italy": 0.1, "Belgium": 0.08,
             "Portugal": 0.05}
# (product name, category, unit price)
PRODUCTS = [
    ("Laptop 14 inch", "Electronics", 899.0), ("Wireless Mouse", "Electronics", 25.5),
    ("Mechanical Keyboard", "Electronics", 79.9), ("USB-C Hub", "Electronics", 39.0),
    ("Noise-Cancelling Headphones", "Electronics", 199.0), ("Smartphone", "Electronics", 549.0),
    ("Running Shoes", "Sports", 89.0), ("Yoga Mat", "Sports", 29.0),
    ("Dumbbell Set", "Sports", 59.0), ("Tennis Racket", "Sports", 120.0),
    ("Cycling Helmet", "Sports", 45.0), ("Novel: The Long Road", "Books", 14.5),
    ("Cookbook: Quick Meals", "Books", 22.0), ("Data Science Handbook", "Books", 48.0),
    ("Children's Atlas", "Books", 18.0), ("Coffee Maker", "Home", 69.0),
    ("Desk Lamp", "Home", 34.0), ("Bed Sheet Set", "Home", 49.0),
    ("Cast Iron Pan", "Home", 55.0), ("Vacuum Cleaner", "Home", 159.0),
    ("Winter Jacket", "Clothing", 130.0), ("Denim Jeans", "Clothing", 70.0),
    ("Cotton T-Shirt", "Clothing", 15.0), ("Wool Scarf", "Clothing", 28.0),
    ("Leather Belt", "Clothing", 32.0),
]
STATUSES = ["delivered", "shipped", "pending", "cancelled"]
STATUS_WEIGHTS = [0.7, 0.1, 0.08, 0.12]

SCHEMA = """
CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT NOT NULL,
    country      TEXT NOT NULL,
    signup_date  TEXT NOT NULL
);
CREATE TABLE products (
    product_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    unit_price  REAL NOT NULL
);
CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date   TEXT NOT NULL,
    status       TEXT NOT NULL
);
CREATE TABLE order_items (
    order_item_id  INTEGER PRIMARY KEY,
    order_id       INTEGER NOT NULL REFERENCES orders(order_id),
    product_id     INTEGER NOT NULL REFERENCES products(product_id),
    quantity       INTEGER NOT NULL,
    unit_price     REAL NOT NULL
);
"""


def random_date(rng: random.Random, start: date, end: date) -> date:
    """A uniformly random date between start and end (inclusive)."""
    return start + timedelta(days=rng.randint(0, (end - start).days))


def build(db_path: Path = DB_PATH) -> None:
    """Create the database file from scratch."""
    rng = random.Random(SEED)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    # customers
    countries = list(COUNTRIES)
    weights = list(COUNTRIES.values())
    customers = []
    for cid in range(1, N_CUSTOMERS + 1):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        email = f"user{cid}@example.com"
        country = rng.choices(countries, weights)[0]
        signup = random_date(rng, date(2022, 1, 1), date(2023, 12, 31))
        customers.append((cid, name, email, country, signup.isoformat()))
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

    # products
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        [(i, name, cat, price) for i, (name, cat, price) in enumerate(PRODUCTS, start=1)],
    )

    # orders and order_items: an order is placed after the customer signed up
    signup_by_customer = {c[0]: date.fromisoformat(c[4]) for c in customers}
    orders, items = [], []
    item_id = 1
    for oid in range(1, N_ORDERS + 1):
        cid = rng.randint(1, N_CUSTOMERS)
        order_day = random_date(rng, signup_by_customer[cid], date(2024, 12, 31))
        status = rng.choices(STATUSES, STATUS_WEIGHTS)[0]
        orders.append((oid, cid, order_day.isoformat(), status))
        for _ in range(rng.randint(1, 4)):
            pid = rng.randint(1, len(PRODUCTS))
            list_price = PRODUCTS[pid - 1][2]
            items.append((item_id, oid, pid, rng.randint(1, 3), list_price))
            item_id += 1
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", items)

    conn.commit()
    conn.close()
    print(
        f"Wrote {db_path}: {len(customers)} customers, {len(PRODUCTS)} products, "
        f"{len(orders)} orders, {len(items)} order items"
    )


if __name__ == "__main__":
    build()
