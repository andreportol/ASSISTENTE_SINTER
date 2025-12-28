from __future__ import annotations

import json
import re

from django.db import connections

from .table_builder import normalize_text


PDF_BLOCK_RE = re.compile(r"```json\s*(\{.*?\"pdf_base64\".*?\})\s*```", re.IGNORECASE | re.DOTALL)
SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def looks_like_pdf_request(query: str) -> bool:
    text = normalize_text(query)
    if not text:
        return False
    return any(token in text for token in ("pdf", "relatorio", "formulario"))


def extract_sql_block(text: str) -> str | None:
    if not text:
        return None
    match = SQL_BLOCK_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def extract_pdf_payload(text: str) -> dict | None:
    if not text:
        return None
    block_match = PDF_BLOCK_RE.search(text)
    if block_match:
        try:
            return json.loads(block_match.group(1))
        except json.JSONDecodeError:
            pass
    for match in re.finditer(r"\{.*?\"pdf_base64\".*?\}", text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def build_pdf_report_from_sql(sql: str, max_rows: int, title: str | None = None) -> dict | None:
    sql_clean = sql.strip().rstrip(";")
    if not sql_clean.lower().startswith("select"):
        raise ValueError("Somente SELECT é permitido para gerar PDF.")

    with connections["default"].cursor() as cursor:
        cursor.execute(sql_clean)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [col[0] for col in (cursor.description or [])]

    if not rows or not columns:
        return None

    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]

    _spec = {
        "title": title or "Relatorio PDF",
        "columns": columns,
        "rows": [list(row) for row in rows],
    }

    payload = None
    if not payload or not payload.get("pdf_base64"):
        return None

    return {
        "base64": payload.get("pdf_base64"),
        "file_name": payload.get("file_name") or "relatorio.pdf",
        "title": payload.get("title") or _spec["title"],
        "rows_used": payload.get("rows_used") or len(rows),
        "truncated": bool(payload.get("truncated") or truncated),
    }
