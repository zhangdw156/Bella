#!/usr/bin/env python3
"""Dump PostgreSQL databases to SQLite world.db files for Bella environments."""

import os
import sqlite3
import sys
from pathlib import Path

import psycopg2

DATABASES = ["employees", "chinook", "dvdrental", "sports"]
ENV_DIR = Path(__file__).parent.parent / "environments"

PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USER = os.getenv("POSTGRES_USERNAME", "postgres")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "password")


PG_TO_SQLITE_TYPE = {
    "integer": "INTEGER", "bigint": "INTEGER", "smallint": "INTEGER",
    "serial": "INTEGER", "bigserial": "INTEGER",
    "numeric": "REAL", "decimal": "REAL", "real": "REAL",
    "double precision": "REAL",
    "boolean": "INTEGER",
    "date": "TEXT", "timestamp without time zone": "TEXT",
    "timestamp with time zone": "TEXT",
    "text": "TEXT", "character varying": "TEXT", "character": "TEXT",
    "uuid": "TEXT", "inet": "TEXT", "jsonb": "TEXT", "json": "TEXT",
    "bytea": "BLOB", "ARRAY": "TEXT",
    "USER-DEFINED": "TEXT",
}


def map_type(pg_type: str) -> str:
    t = pg_type.lower().strip()
    if t in PG_TO_SQLITE_TYPE:
        return PG_TO_SQLITE_TYPE[t]
    for prefix in ("character varying", "character", "numeric", "decimal"):
        if t.startswith(prefix):
            return PG_TO_SQLITE_TYPE[prefix]
    return "TEXT"


def pg_connect(db: str):
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER,
                            password=PG_PASS, database=db)


def get_tables(pg_cur, schema: str) -> list[str]:
    pg_cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema=%s AND table_type='BASE TABLE' ORDER BY table_name",
        (schema,))
    return [r[0] for r in pg_cur.fetchall()]


def get_columns(pg_cur, schema: str, table: str) -> list[dict]:
    pg_cur.execute(
        "SELECT column_name, data_type, is_nullable "
        "FROM information_schema.columns "
        "WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
        (schema, table))
    return [{"name": r[0], "pg_type": r[1], "nullable": r[2] == "YES"}
            for r in pg_cur.fetchall()]


def get_primary_key(pg_cur, schema: str, table: str) -> list[str]:
    pg_cur.execute(
        "SELECT kcu.column_name "
        "FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu "
        "  ON tc.constraint_name = kcu.constraint_name "
        "  AND tc.table_schema = kcu.table_schema "
        "WHERE tc.table_schema=%s AND tc.table_name=%s "
        "  AND tc.constraint_type='PRIMARY KEY' "
        "ORDER BY kcu.ordinal_position",
        (schema, table))
    return [r[0] for r in pg_cur.fetchall()]


def coerce_value(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (list, dict)):
        import json
        return json.dumps(v, ensure_ascii=False)
    from decimal import Decimal
    if isinstance(v, Decimal):
        return float(v)
    from datetime import date, datetime
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return v


def dump_database(db: str):
    env_name = f"mcpmark_postgres_{db}"
    env_path = ENV_DIR / env_name
    world_dir = env_path / "world"
    world_db_path = world_dir / "world.db"
    schema_path = world_dir / "schema.sql"

    schema = "employees" if db == "employees" else "public"

    pg_conn = pg_connect(db)
    pg_cur = pg_conn.cursor()

    tables = get_tables(pg_cur, schema)
    print(f"\n=== {db} ({len(tables)} tables, schema={schema}) ===")

    if world_db_path.exists():
        world_db_path.unlink()

    sl_conn = sqlite3.connect(str(world_db_path))
    schema_sql_parts = []

    for table in tables:
        columns = get_columns(pg_cur, schema, table)
        pk_cols = get_primary_key(pg_cur, schema, table)

        col_defs = []
        for col in columns:
            sqlite_type = map_type(col["pg_type"])
            parts = [f'"{col["name"]}"', sqlite_type]
            if col["name"] in pk_cols and len(pk_cols) == 1:
                parts.append("PRIMARY KEY")
            if not col["nullable"] and col["name"] not in pk_cols:
                parts.append("NOT NULL")
            col_defs.append(" ".join(parts))

        if len(pk_cols) > 1:
            pk_def = ", ".join(f'"{c}"' for c in pk_cols)
            col_defs.append(f"PRIMARY KEY ({pk_def})")

        create_sql = f'CREATE TABLE "{table}" (\n  ' + ",\n  ".join(col_defs) + "\n);"
        schema_sql_parts.append(create_sql)
        sl_conn.execute(create_sql)

        qualified = f'"{schema}"."{table}"'
        col_names = [c["name"] for c in columns]
        select_cols = ", ".join(f'"{c}"' for c in col_names)

        pg_cur.execute(f"SELECT {select_cols} FROM {qualified}")
        rows = pg_cur.fetchall()
        if rows:
            placeholders = ", ".join("?" * len(col_names))
            insert_sql = f'INSERT INTO "{table}" VALUES ({placeholders})'
            sl_conn.executemany(insert_sql,
                                [tuple(coerce_value(v) for v in row) for row in rows])

        row_count = sl_conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"  {table}: {len(columns)} cols, {row_count} rows")

    sl_conn.commit()
    sl_conn.close()
    pg_cur.close()
    pg_conn.close()

    schema_path.write_text("\n\n".join(schema_sql_parts) + "\n")
    print(f"  -> {world_db_path}")
    print(f"  -> {schema_path}")


def main():
    for db in DATABASES:
        dump_database(db)
    print("\nDone.")


if __name__ == "__main__":
    main()
