#!/usr/bin/env python3
"""Import tau2-bench retail db.json into SQLite world.db."""

import json
import os
import sqlite3

SOURCE_PATH = os.path.join(os.path.dirname(__file__), "../../tau2-bench/data/tau2/domains/retail/db.json")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../environments/tau3_retail/world/schema.sql")
DB_PATH = os.path.join(os.path.dirname(__file__), "../environments/tau3_retail/world/world.db")


def main():
    with open(SOURCE_PATH) as f:
        db = json.load(f)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    for product_id, product in db["products"].items():
        conn.execute(
            "INSERT INTO products VALUES (?, ?)",
            (product_id, json.dumps(product, ensure_ascii=False)),
        )

    for user_id, user in db["users"].items():
        conn.execute(
            "INSERT INTO users VALUES (?, ?)",
            (user_id, json.dumps(user, ensure_ascii=False)),
        )

    for order_id, order in db["orders"].items():
        conn.execute(
            "INSERT INTO orders VALUES (?, ?)",
            (order_id, json.dumps(order, ensure_ascii=False)),
        )

    conn.commit()

    products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    orders = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    print(f"Products: {products}, Users: {users}, Orders: {orders}")

    conn.close()


if __name__ == "__main__":
    main()
