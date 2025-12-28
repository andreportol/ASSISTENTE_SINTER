from __future__ import annotations

import re
from typing import Any

from django.db import connections


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_FILTER_VALUES = 200


def is_valid_identifier(value: str) -> bool:
    return bool(value and IDENTIFIER_RE.match(value))


def get_table_columns(table_name: str) -> list[str]:
    if not is_valid_identifier(table_name):
        return []
    with connections["default"].cursor() as cursor:
        description = connections["default"].introspection.get_table_description(cursor, table_name)
    return [col.name for col in description]


def get_numeric_columns(table_name: str) -> list[str]:
    if not is_valid_identifier(table_name):
        return []
    conn = connections["default"]
    columns = []
    with conn.cursor() as cursor:
        description = conn.introspection.get_table_description(cursor, table_name)
        for col in description:
            base_type = conn.introspection.get_field_type(col.type_code, col).lower()
            if any(token in base_type for token in ["int", "float", "decimal", "numeric", "double", "real"]):
                columns.append(col.name)
    return columns


def get_authorized_tables() -> list[str]:
    conn = connections["default"]
    if conn.vendor == "postgresql":
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                AND has_table_privilege(current_user, schemaname || '.' || tablename, 'SELECT')
                ORDER BY tablename;
                """
            )
            return [row[0] for row in cursor.fetchall()]
    return conn.introspection.table_names()


def get_distinct_values_with_flag(
    table_name: str,
    column_name: str,
    limit: int = MAX_FILTER_VALUES,
) -> tuple[list[str], bool]:
    if not is_valid_identifier(table_name) or not is_valid_identifier(column_name):
        return [], False
    with connections["default"].cursor() as cursor:
        cursor.execute(
            f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT %s",
            [limit + 1],
        )
        values = [str(row[0]) for row in cursor.fetchall()]
    truncated = len(values) > limit
    return values[:limit], truncated


def get_distinct_values(table_name: str, column_name: str, limit: int = MAX_FILTER_VALUES) -> list[str]:
    values, _truncated = get_distinct_values_with_flag(table_name, column_name, limit=limit)
    return values


def get_foreign_key_map(table_name: str) -> dict[str, tuple[str, str]]:
    if not is_valid_identifier(table_name):
        return {}
    conn = connections["default"]
    fk_map: dict[str, tuple[str, str]] = {}
    with conn.cursor() as cursor:
        constraints = conn.introspection.get_constraints(cursor, table_name)
    for info in constraints.values():
        if not info.get("foreign_key"):
            continue
        columns = info.get("columns") or []
        if len(columns) != 1:
            continue
        fk_table, fk_column = info["foreign_key"]
        fk_map[columns[0]] = (fk_table, fk_column)
    return fk_map


def get_display_column(table_name: str) -> str | None:
    candidates = ["nome", "name", "titulo", "title", "descricao", "description"]
    cols = get_table_columns(table_name)
    for cand in candidates:
        if cand in cols:
            return cand
    return None


def build_column_display_map(table_name: str, columns: list[str]) -> dict[str, str]:
    fk_map = get_foreign_key_map(table_name)
    display_map: dict[str, str] = {}
    for col in columns:
        ref = fk_map.get(col)
        if ref:
            ref_table, _ref_col = ref
            display_col = get_display_column(ref_table)
            if display_col:
                label = f"{ref_table}.{display_col}"
            else:
                label = f"{ref_table}.id"
            display_map[col] = label
        else:
            display_map[col] = col
    return display_map
