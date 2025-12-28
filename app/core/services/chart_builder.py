from __future__ import annotations

import re
from typing import Any

from django.db import connections

from .sql_introspection import get_authorized_tables, get_table_columns, is_valid_identifier
from .table_builder import build_agg_expression, match_identifier, normalize_text


MAX_CHART_ROWS = 200


def rewrite_numeric_agg(sql: str) -> str | None:
    if not sql:
        return None

    numeric_agg_re = re.compile(r"\b(sum|avg)\s*\(\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*\)", re.IGNORECASE)

    def _repl(match: re.Match) -> str:
        func = match.group(1).upper()
        col = match.group(2)
        return f"{func}(CAST({col} AS NUMERIC))"

    rewritten = numeric_agg_re.sub(_repl, sql)
    if rewritten == sql:
        return None
    return rewritten


def detect_chart_type(query: str, response_text: str) -> str:
    text = f"{query} {response_text}".lower()
    if "pizza" in text or "pie" in text:
        return "pie"
    if "linha" in text or "line" in text:
        return "line"
    return "bar"


def build_chart_data(sql: str, chart_type: str = "bar") -> dict | None:
    sql_clean = sql.strip().rstrip(";")
    if not sql_clean.lower().startswith("select"):
        raise ValueError("Somente SELECT é permitido para gerar gráficos.")

    def _run_query(stmt: str) -> tuple[list[tuple], list[str]]:
        with connections["default"].cursor() as cursor:
            cursor.execute(stmt)
            fetched = cursor.fetchmany(MAX_CHART_ROWS + 1)
            columns = [col[0] for col in (cursor.description or [])]
        return fetched, columns

    try:
        rows, cols = _run_query(sql_clean)
    except Exception as exc:
        fallback_sql = rewrite_numeric_agg(sql_clean)
        if not fallback_sql:
            raise
        try:
            rows, cols = _run_query(fallback_sql)
        except Exception:
            raise exc

    if not rows or len(cols) < 2:
        return None

    truncated = len(rows) > MAX_CHART_ROWS
    if truncated:
        rows = rows[:MAX_CHART_ROWS]

    labels: list[str] = []
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[1])
        except Exception:
            continue
        labels.append(str(row[0]))
        values.append(value)

    if not values:
        return None

    return {
        "labels": labels,
        "values": values,
        "type": chart_type,
        "title": f"Gráfico: {cols[0]} x {cols[1]}",
        "truncated": truncated,
        "max_rows": MAX_CHART_ROWS,
    }


def build_chart_data_from_rows(rows: list[tuple], cols: list[str], chart_type: str) -> dict | None:
    if not rows or len(cols) < 2:
        return None
    labels: list[str] = []
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[1])
        except Exception:
            continue
        labels.append(str(row[0]))
        values.append(value)
    if not values:
        return None
    return {
        "labels": labels,
        "values": values,
        "type": chart_type,
        "title": f"Gráfico: {cols[0]} x {cols[1]}",
        "truncated": False,
        "max_rows": len(values),
    }


def extract_chart_from_text(text: str) -> dict | None:
    if not text:
        return None
    chart_list_re = re.compile(r"^\s*(?:[-*]\s*|\d+\.\s*)?(.+?):\s*([0-9][0-9\.,]*)\s*$", re.MULTILINE)
    matches = chart_list_re.findall(text)
    if not matches:
        return None

    labels: list[str] = []
    values: list[float] = []
    for label, raw_value in matches:
        value_clean = raw_value.replace(".", "").replace(" ", "").replace(",", ".")
        try:
            value = float(value_clean)
        except Exception:
            continue
        labels.append(label.strip())
        values.append(value)

    if not values:
        return None

    def _normalize_title(raw: str) -> str:
        cleaned = raw.strip().rstrip(":")
        cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned).strip()
        return cleaned or "Gráfico gerado"

    lines = text.splitlines()
    list_idx = None
    for idx, line in enumerate(lines):
        if chart_list_re.match(line):
            list_idx = idx
            break

    title_candidate = ""
    keywords = ("contagem", "total", "registros", "por", "top")
    if list_idx is not None:
        for idx in range(list_idx - 1, -1, -1):
            line = lines[idx].strip()
            if not line:
                continue
            lower = line.lower()
            if ":" in line or any(word in lower for word in keywords):
                title_candidate = line
                break

    if not title_candidate:
        for line in lines:
            lower = line.lower()
            if ":" in line and any(word in lower for word in keywords):
                title_candidate = line
                break

    title = _normalize_title(title_candidate or "Gráfico gerado")
    chart_type = detect_chart_type("", text)

    return {
        "labels": labels,
        "values": values,
        "type": chart_type,
        "title": title,
        "truncated": False,
        "max_rows": len(values),
    }


def looks_like_chart_request(query: str) -> bool:
    text = normalize_text(query)
    if not text:
        return False
    return any(token in text for token in ("grafico", "chart", "barra", "barras", "linha", "line", "pizza", "pie"))


def build_chart_from_query(query: str) -> dict | None:
    text = normalize_text(query)
    if not text:
        return None

    agg = None
    if any(token in text for token in ("media", "avg")):
        agg = "avg"
    elif any(token in text for token in ("soma", "somatorio", "sum")):
        agg = "sum"
    elif any(token in text for token in ("contagem", "count", "quantidade", "total")):
        agg = "count"

    if not agg:
        return None

    table_match = re.search(r"\b(?:tabela|table)\s+([a-z0-9_]+)", text)
    table_raw = table_match.group(1) if table_match else ""
    if not table_raw:
        return None

    tables = get_authorized_tables()
    table = match_identifier(table_raw, tables)
    if not table:
        return None

    columns = get_table_columns(table)
    label_match = re.search(
        r"\bpor\s+(.+?)(?:\s+(?:da|na)\s+tabela\b|\s+tabela\b|$)",
        text,
    )
    label_raw = (label_match.group(1) if label_match else "").strip()
    label_col = match_identifier(label_raw, columns) if label_raw else None
    if not label_col:
        return None

    value_col = None
    if agg != "count":
        value_match = re.search(
            r"\b(?:soma|somatorio|sum|media|avg)\s+(?:d[aeo]s?\s+)?(.+?)(?:\s+por\b|$)",
            text,
        )
        value_raw = (value_match.group(1) if value_match else "").strip()
        value_col = match_identifier(value_raw, columns) if value_raw else None
        if not value_col:
            return None

    select_expr = build_agg_expression(agg, value_col)
    sql = (
        f"SELECT {label_col} AS label, {select_expr} AS value "
        f"FROM {table} GROUP BY {label_col} ORDER BY value DESC LIMIT {MAX_CHART_ROWS}"
    )
    chart_type = detect_chart_type(query, "")
    return build_chart_data(sql, chart_type=chart_type)


def build_chart_from_description(text: str) -> dict | None:
    if not text:
        return None
    norm = normalize_text(text)
    if not norm:
        return None

    agg = None
    if re.search(r"\b(soma|sum)\b", norm):
        agg = "sum"
    elif re.search(r"\b(media|avg)\b", norm):
        agg = "avg"
    elif re.search(r"\b(contagem|count|quantidade|total)\b", norm):
        agg = "count"

    if not agg:
        return None

    table_match = re.search(r"\btabela\s+([a-z0-9_]+)", norm)
    table_raw = table_match.group(1) if table_match else ""
    if not table_raw:
        return None

    tables = get_authorized_tables()
    table = match_identifier(table_raw, tables)
    if not table:
        return None

    columns = get_table_columns(table)
    label_match = re.search(r"\bpor\s+([a-z0-9_]+)", norm)
    label_raw = label_match.group(1) if label_match else ""
    label_col = match_identifier(label_raw, columns) if label_raw else None
    if not label_col:
        return None

    value_col = None
    if agg != "count":
        value_match = re.search(r"\b(?:soma|sum|media|avg)\s+([a-z0-9_]+)", norm)
        value_raw = value_match.group(1) if value_match else ""
        value_col = match_identifier(value_raw, columns) if value_raw else None
        if not value_col:
            return None

    select_expr = build_agg_expression(agg, value_col)
    sql = (
        f"SELECT {label_col} AS label, {select_expr} AS value "
        f"FROM {table} GROUP BY {label_col} ORDER BY value DESC LIMIT {MAX_CHART_ROWS}"
    )
    chart_type = detect_chart_type(text, "")
    return build_chart_data(sql, chart_type=chart_type)


def build_chart_from_history(history: list[tuple[str, str]], skip_text: str) -> dict | None:
    if not history:
        return None
    for role, content in reversed(history):
        if role != "assistant" or not content or content == skip_text:
            continue
        chart = build_chart_from_description(content)
        if chart:
            return chart
    return None
