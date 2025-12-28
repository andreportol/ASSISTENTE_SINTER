from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from decimal import Decimal

from .sql_introspection import get_authorized_tables, get_table_columns, is_valid_identifier


def normalize_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKD", text or "")
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def normalize_identifier(text: str) -> str:
    cleaned = unicodedata.normalize("NFKD", text or "")
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", cleaned)
    return cleaned.lower()


def match_identifier(candidate: str, available: list[str]) -> str | None:
    norm = normalize_identifier(candidate)
    if not norm:
        return None
    for item in available:
        if normalize_identifier(item) == norm and is_valid_identifier(item):
            return item
    return None


def resolve_column_input(value: str, reverse_map: dict[str, str]) -> str:
    if not value:
        return ""
    if value in reverse_map:
        return reverse_map[value]
    return value


def build_agg_expression(agg: str, value_col: str | None) -> str:
    if agg == "count":
        if value_col:
            return f"COUNT({value_col})"
        return "COUNT(*)"
    if not value_col:
        raise ValueError("Coluna numerica obrigatoria para SUM ou AVG.")
    return f"{agg.upper()}(CAST({value_col} AS NUMERIC))"


def agg_label_pt(agg: str) -> str:
    labels = {"count": "Contagem", "sum": "Soma", "avg": "Média"}
    return labels.get(agg, agg.upper())


def format_pt_br(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float, Decimal)):
        dec = value if isinstance(value, Decimal) else Decimal(str(value))
        dec = dec.quantize(Decimal("0.01"))
        formatted = f"{dec:,.2f}"
        return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return str(value)


def parse_br_date(query: str) -> str | None:
    if not query:
        return None
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", query)
    if not match:
        return None
    dia, mes, ano = match.group(1), match.group(2), match.group(3)
    try:
        dt = datetime(int(ano), int(mes), int(dia))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def build_report_sql_from_query(query: str, max_rows: int) -> tuple[str, str] | None:
    text = normalize_text(query)
    if not text:
        return None

    agg = None
    if any(token in text for token in ("soma", "somatorio", "sum")):
        agg = "sum"
    elif any(token in text for token in ("media", "avg")):
        agg = "avg"
    elif any(token in text for token in ("contagem", "count", "quantidade", "total")):
        agg = "count"

    if not agg:
        return None

    tables = get_authorized_tables()
    table = None
    table_match = re.search(r"\b(?:tabela|table)\s+([a-z0-9_]+)", text)
    if table_match:
        table = match_identifier(table_match.group(1), tables)
    if not table and "imoveis" in tables:
        table = "imoveis"
    if not table and tables:
        table = tables[0]
    if not table:
        return None

    columns = get_table_columns(table)
    if not columns:
        return None

    label_raw = ""
    label_match = re.search(r"\bpor\s+([a-z0-9_]+)\b", text)
    if label_match:
        label_raw = label_match.group(1)
    label_col = match_identifier(label_raw, columns) if label_raw else None

    value_col = None
    if agg != "count":
        value_match = re.search(
            r"\b(?:soma|somatorio|sum|media|avg)\s+(?:d[aeo]s?\s+)?([a-z0-9_]+)\b",
            text,
        )
        value_raw = value_match.group(1) if value_match else ""
        value_col = match_identifier(value_raw, columns) if value_raw else None
        if not value_col:
            return None
    else:
        value_match = re.search(
            r"\b(?:quantidade|contagem|count|total)\s+(?:d[aeo]s?\s+)?([a-z0-9_]+)\b",
            text,
        )
        value_raw = value_match.group(1) if value_match else ""
        value_col = match_identifier(value_raw, columns) if value_raw else None

    if not label_col:
        return None

    select_expr = build_agg_expression(agg, value_col)
    sql = (
        f"SELECT {label_col} AS label, {select_expr} AS value "
        f"FROM {table} GROUP BY {label_col} ORDER BY value DESC LIMIT {max_rows}"
    )
    title = f"Relatorio: {label_col} x {select_expr}"
    return sql, title
