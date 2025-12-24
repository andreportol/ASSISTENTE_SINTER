import csv
import hashlib
import json
import os
import re
import unicodedata
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.db import connections
from django.http import HttpResponse
from django.shortcuts import render
from django.utils.text import get_valid_filename
from django.views.decorators.csrf import ensure_csrf_cookie
from django.core.cache import cache

from .langchain_agents import (
    answer_question,
    generate_pdf_report,
    render_pdf_report,
    reset_rules_retriever_cache,
)
from .rag_index import ingest_docs_to_faiss, rebuild_faiss_index

SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
CHART_LIST_RE = re.compile(r"^\s*(?:[-*]\s*|\d+\.\s*)?(.+?):\s*([0-9][0-9\.,]*)\s*$", re.MULTILINE)
PDF_BLOCK_RE = re.compile(r"```json\s*(\{.*?\"pdf_base64\".*?\})\s*```", re.IGNORECASE | re.DOTALL)
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NUMERIC_AGG_RE = re.compile(r"\b(sum|avg)\s*\(\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*\)", re.IGNORECASE)
MAX_CHART_ROWS = 200  # evita carregar milhões de linhas no gráfico
TABLE_PAGE_SIZE = 20
MAX_FILTER_VALUES = 200  # evita carregar muitos valores distintos para filtros
MAX_PDF_ROWS = 200
ALLOWED_DOC_EXTS = {".pdf", ".txt"}
DEFAULT_LLM_CONFIG = {
    "model": getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
    "temperature": getattr(settings, "OPENAI_TEMPERATURE", 0.1),
    "chunk_size": getattr(settings, "RAG_CHUNK_SIZE", 700),
    "chunk_overlap": getattr(settings, "RAG_CHUNK_OVERLAP", 150),
}


def _extract_sql_block(text: str) -> str | None:
    if not text:
        return None
    match = SQL_BLOCK_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _detect_chart_type(query: str, response_text: str) -> str:
    text = f"{query} {response_text}".lower()
    if "pizza" in text or "pie" in text:
        return "pie"
    if "linha" in text or "line" in text:
        return "line"
    return "bar"


def _normalize_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKD", text or "")
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _normalize_identifier(text: str) -> str:
    cleaned = unicodedata.normalize("NFKD", text or "")
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", cleaned)
    return cleaned.lower()


def _cache_key(prefix: str, query: str, session_key: str | None) -> str:
    normalized = _normalize_text(query) or query.strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    session_part = session_key or "anon"
    return f"{prefix}:{session_part}:{digest}"


def _rewrite_numeric_agg(sql: str) -> str | None:
    if not sql:
        return None

    def _repl(match: re.Match) -> str:
        func = match.group(1).upper()
        col = match.group(2)
        return f"{func}(CAST({col} AS NUMERIC))"

    rewritten = NUMERIC_AGG_RE.sub(_repl, sql)
    if rewritten == sql:
        return None
    return rewritten


def _build_agg_expression(agg: str, value_col: str | None) -> str:
    if agg == "count":
        if value_col:
            return f"COUNT({value_col})"
        return "COUNT(*)"
    if not value_col:
        raise ValueError("Coluna numerica obrigatoria para SUM ou AVG.")
    return f"{agg.upper()}(CAST({value_col} AS NUMERIC))"


def _agg_label_pt(agg: str) -> str:
    labels = {"count": "Contagem", "sum": "Soma", "avg": "Média"}
    return labels.get(agg, agg.upper())


def _format_pt_br(value: object) -> str:
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


def _build_chart_data(sql: str, chart_type: str = "bar") -> dict | None:
    # Garante apenas SELECT simples
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
    except Exception as exc:  # noqa: BLE001
        fallback_sql = _rewrite_numeric_agg(sql_clean)
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
            # ignora linhas que não possuem valor numérico na segunda coluna
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


def _build_chart_data_from_rows(rows: list[tuple], cols: list[str], chart_type: str) -> dict | None:
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


def _extract_chart_from_text(text: str) -> dict | None:
    if not text:
        return None
    matches = CHART_LIST_RE.findall(text)
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
        if CHART_LIST_RE.match(line):
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

    chart_type = _detect_chart_type("", text)
    return {
        "labels": labels,
        "values": values,
        "type": chart_type,
        "title": title,
        "truncated": False,
        "max_rows": len(values),
    }


def _apply_runtime_config(config: dict) -> None:
    settings.OPENAI_MODEL = config["model"]
    settings.OPENAI_TEMPERATURE = config["temperature"]
    settings.RAG_CHUNK_SIZE = config["chunk_size"]
    settings.RAG_CHUNK_OVERLAP = config["chunk_overlap"]


def _load_llm_config(request) -> dict:
    config = request.session.get("llm_config") or DEFAULT_LLM_CONFIG.copy()
    config.setdefault("model", DEFAULT_LLM_CONFIG["model"])
    config.setdefault("temperature", DEFAULT_LLM_CONFIG["temperature"])
    config.setdefault("chunk_size", DEFAULT_LLM_CONFIG["chunk_size"])
    config.setdefault("chunk_overlap", DEFAULT_LLM_CONFIG["chunk_overlap"])
    _apply_runtime_config(config)
    return config


def _is_valid_identifier(value: str) -> bool:
    return bool(value and IDENTIFIER_RE.match(value))


def _get_table_columns(table_name: str) -> list[str]:
    if not _is_valid_identifier(table_name):
        return []
    with connections["default"].cursor() as cursor:
        description = connections["default"].introspection.get_table_description(cursor, table_name)
    return [col.name for col in description]


def _get_numeric_columns(table_name: str) -> list[str]:
    if not _is_valid_identifier(table_name):
        return []
    conn = connections["default"]
    if conn.vendor == "postgresql":
        numeric_types = {
            "smallint",
            "integer",
            "bigint",
            "decimal",
            "numeric",
            "real",
            "double precision",
            "smallserial",
            "serial",
            "bigserial",
            "money",
        }
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                """,
                [table_name],
            )
            rows = cursor.fetchall()
        return [row[0] for row in rows if row[1] in numeric_types]
    return _get_table_columns(table_name)


def _get_distinct_values(table_name: str, column_name: str, limit: int = MAX_FILTER_VALUES) -> list[str]:
    values, _ = _get_distinct_values_with_flag(table_name, column_name, limit)
    return values


def _get_distinct_values_with_flag(
    table_name: str, column_name: str, limit: int = MAX_FILTER_VALUES
) -> tuple[list[str], bool]:
    if not _is_valid_identifier(table_name) or not _is_valid_identifier(column_name):
        return [], False
    if table_name not in _get_authorized_tables():
        return [], False
    if column_name not in _get_table_columns(table_name):
        return [], False

    sql = (
        f"SELECT DISTINCT {column_name} FROM {table_name} "
        f"WHERE {column_name} IS NOT NULL ORDER BY {column_name} ASC LIMIT %s"
    )
    with connections["default"].cursor() as cursor:
        cursor.execute(sql, [limit + 1])
        rows = cursor.fetchall()
    truncated = len(rows) > limit
    if truncated:
        rows = rows[:limit]
    return [str(row[0]) for row in rows if row and row[0] is not None], truncated


def _get_authorized_tables() -> list[str]:
    conn = connections["default"]
    if conn.vendor == "postgresql":
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND has_table_privilege(format('%I.%I', table_schema, table_name), 'SELECT')
                ORDER BY table_name
                """
            )
            return [row[0] for row in cursor.fetchall()]
    return conn.introspection.table_names()


def _get_foreign_key_map(table_name: str) -> dict[str, tuple[str, str]]:
    conn = connections["default"]
    if conn.vendor != "postgresql":
        return {}
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = 'public'
              AND tc.table_name = %s
            """,
            [table_name],
        )
        rows = cursor.fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def _get_display_column(table_name: str) -> str | None:
    columns = _get_table_columns(table_name)
    if "nome" in columns:
        return "nome"
    return None


def _build_column_display_map(table_name: str, columns: list[str]) -> dict[str, str]:
    fk_map = _get_foreign_key_map(table_name)
    display_map: dict[str, str] = {}
    used_labels: set[str] = set()
    for col in columns:
        label = col
        ref = fk_map.get(col)
        if ref:
            ref_table, _ref_col = ref
            display_col = _get_display_column(ref_table)
            if display_col and col.endswith("_id"):
                base = ref_table[:-1] if ref_table.endswith("s") else ref_table
                label = f"{display_col}_{base}"
        if label in used_labels:
            label = col
        used_labels.add(label)
        display_map[col] = label
    return display_map


def _resolve_column_input(value: str, reverse_map: dict[str, str]) -> str:
    if not value:
        return value
    return reverse_map.get(value, value)


def _match_identifier(candidate: str, available: list[str]) -> str | None:
    norm = _normalize_identifier(candidate)
    if not norm:
        return None
    for item in available:
        if _normalize_identifier(item) == norm and _is_valid_identifier(item):
            return item
    return None


def _looks_like_chart_request(query: str) -> bool:
    text = _normalize_text(query)
    if not text:
        return False
    return any(token in text for token in ("grafico", "chart", "barra", "barras", "linha", "line", "pizza", "pie"))


def _looks_like_pdf_request(query: str) -> bool:
    text = _normalize_text(query)
    if not text:
        return False
    return any(token in text for token in ("pdf", "relatorio", "formulario"))


def _extract_pdf_payload(text: str) -> dict | None:
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


def _build_pdf_report_from_sql(sql: str, title: str | None = None) -> dict | None:
    sql_clean = sql.strip().rstrip(";")
    if not sql_clean.lower().startswith("select"):
        raise ValueError("Somente SELECT é permitido para gerar PDF.")

    with connections["default"].cursor() as cursor:
        cursor.execute(sql_clean)
        rows = cursor.fetchmany(MAX_PDF_ROWS + 1)
        columns = [col[0] for col in (cursor.description or [])]

    if not rows or not columns:
        return None

    truncated = len(rows) > MAX_PDF_ROWS
    if truncated:
        rows = rows[:MAX_PDF_ROWS]

    spec = {
        "title": title or "Relatorio PDF",
        "columns": columns,
        "rows": [list(row) for row in rows],
    }
    payload_text = render_pdf_report(json.dumps(spec, ensure_ascii=False))
    payload = _extract_pdf_payload(payload_text)
    if not payload or not payload.get("pdf_base64"):
        return None
    return {
        "base64": payload.get("pdf_base64"),
        "file_name": payload.get("file_name") or "relatorio.pdf",
        "title": payload.get("title") or spec["title"],
        "rows_used": payload.get("rows_used") or len(rows),
        "truncated": bool(payload.get("truncated") or truncated),
    }


def _build_report_sql_from_query(query: str) -> tuple[str, str] | None:
    text = _normalize_text(query)
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

    tables = _get_authorized_tables()
    table = None
    table_match = re.search(r"\b(?:tabela|table)\s+([a-z0-9_]+)", text)
    if table_match:
        table = _match_identifier(table_match.group(1), tables)
    if not table and "imoveis" in tables:
        table = "imoveis"
    if not table and tables:
        table = tables[0]
    if not table:
        return None

    columns = _get_table_columns(table)
    if not columns:
        return None

    label_raw = ""
    label_match = re.search(r"\bpor\s+([a-z0-9_]+)\b", text)
    if label_match:
        label_raw = label_match.group(1)
    label_col = _match_identifier(label_raw, columns) if label_raw else None

    value_col = None
    if agg != "count":
        value_match = re.search(r"\b(?:soma|somatorio|sum|media|avg)\s+(?:d[aeo]s?\s+)?([a-z0-9_]+)\b", text)
        value_raw = value_match.group(1) if value_match else ""
        value_col = _match_identifier(value_raw, columns) if value_raw else None
        if not value_col:
            return None
    else:
        value_match = re.search(r"\b(?:quantidade|contagem|count|total)\s+(?:d[aeo]s?\s+)?([a-z0-9_]+)\b", text)
        value_raw = value_match.group(1) if value_match else ""
        value_col = _match_identifier(value_raw, columns) if value_raw else None

    if not label_col:
        return None

    select_expr = _build_agg_expression(agg, value_col)
    sql = (
        f"SELECT {label_col} AS label, {select_expr} AS value "
        f"FROM {table} GROUP BY {label_col} ORDER BY value DESC LIMIT {MAX_PDF_ROWS}"
    )
    title = f"Relatorio: {label_col} x {select_expr}"
    return sql, title


def _build_chart_from_query(query: str) -> dict | None:
    text = _normalize_text(query)
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

    tables = _get_authorized_tables()
    table = _match_identifier(table_raw, tables)
    if not table:
        return None

    columns = _get_table_columns(table)
    label_match = re.search(
        r"\bpor\s+(.+?)(?:\s+(?:da|na)\s+tabela\b|\s+tabela\b|$)",
        text,
    )
    label_raw = (label_match.group(1) if label_match else "").strip()
    label_col = _match_identifier(label_raw, columns) if label_raw else None
    if not label_col:
        return None

    value_col = None
    if agg != "count":
        value_match = re.search(
            r"\b(?:soma|somatorio|sum|media|avg)\s+(?:d[aeo]s?\s+)?(.+?)(?:\s+por\b|$)",
            text,
        )
        value_raw = (value_match.group(1) if value_match else "").strip()
        value_col = _match_identifier(value_raw, columns) if value_raw else None
        if not value_col:
            return None

    select_expr = _build_agg_expression(agg, value_col)

    sql = (
        f"SELECT {label_col} AS label, {select_expr} AS value "
        f"FROM {table} GROUP BY {label_col} ORDER BY value DESC LIMIT {MAX_CHART_ROWS}"
    )
    chart_type = _detect_chart_type(query, "")
    return _build_chart_data(sql, chart_type=chart_type)


def _build_chart_from_description(text: str) -> dict | None:
    if not text:
        return None
    norm = _normalize_text(text)
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

    tables = _get_authorized_tables()
    table = _match_identifier(table_raw, tables)
    if not table:
        return None

    columns = _get_table_columns(table)
    if not columns:
        return None

    paren_candidates = [match.group(1).strip() for match in re.finditer(r"\(([^)]+)\)", text)]
    value_candidate = None
    label_candidate = None

    if agg != "count":
        value_match = re.search(r"(?:soma|sum|m[eé]dia|media|avg)[^()]*\(([^)]+)\)", text, re.IGNORECASE)
        if value_match:
            value_candidate = value_match.group(1).strip()

    label_match = re.search(
        r"(?:agrupad[oa]s?\s+por|por)\s+[^()]*\(([^)]+)\)",
        text,
        re.IGNORECASE,
    )
    if label_match:
        label_candidate = label_match.group(1).strip()

    if not label_candidate and paren_candidates:
        if agg != "count" and len(paren_candidates) >= 2:
            label_candidate = paren_candidates[-1]
        else:
            label_candidate = paren_candidates[0]

    if not value_candidate and agg != "count" and paren_candidates:
        value_candidate = paren_candidates[0]

    label_col = _match_identifier(label_candidate or "", columns)
    value_col = _match_identifier(value_candidate or "", columns) if agg != "count" else None

    if not label_col:
        label_match_norm = re.search(r"\bpor\s+([a-z0-9_]+)\b", norm)
        label_raw = label_match_norm.group(1) if label_match_norm else ""
        label_col = _match_identifier(label_raw, columns) if label_raw else None

    if not label_col:
        return None
    if agg != "count" and not value_col:
        return None

    select_expr = _build_agg_expression(agg, value_col)
    sql = (
        f"SELECT {label_col} AS label, {select_expr} AS value "
        f"FROM {table} GROUP BY {label_col} ORDER BY value DESC LIMIT {MAX_CHART_ROWS}"
    )
    chart_type = _detect_chart_type("", text)
    return _build_chart_data(sql, chart_type=chart_type)


def _build_chart_from_history(history: list[tuple[str, str]], skip_text: str) -> dict | None:
    for role, content in reversed(history):
        if role != "assistant" or not content or content == skip_text:
            continue
        chart_data = _extract_chart_from_text(content)
        if chart_data:
            return chart_data
        try:
            chart_data = _build_chart_from_description(content)
        except Exception:
            continue
        if chart_data:
            return chart_data
    return None


@ensure_csrf_cookie
def index(request):
    response_text = None
    upload_message = None
    config_message = None
    error_message = None
    chart_data = None
    chart_error = None
    pdf_report = None
    query = ""
    query_raw = ""
    docs_dir = Path(getattr(settings, "RAG_DOCS_DIR", settings.BASE_DIR / "documents"))
    docs_dir.mkdir(parents=True, exist_ok=True)
    llm_config = _load_llm_config(request)
    history = request.session.get("chat_history", [])
    display_history = []

    if request.method == "POST":
        action = request.POST.get("action", "ask")
        query = (request.POST.get("query") or "").strip()
        query_raw = query
        if action == "clear":
            history = []
            request.session["chat_history"] = history
            request.session.modified = True
            query = ""
            response_text = None
            error_message = None
            chart_data = None
            chart_error = None
            pdf_report = None
        elif action == "save_config":
            model = (request.POST.get("llm_model") or "").strip()
            temp_raw = request.POST.get("llm_temperature") or ""
            chunk_raw = request.POST.get("rag_chunk_size") or ""
            overlap_raw = request.POST.get("rag_chunk_overlap") or ""
            reindex = request.POST.get("reindex_faiss") == "1"

            errors = []
            if not model:
                errors.append("Informe o modelo de LLM.")
            try:
                temperature = float(temp_raw)
            except ValueError:
                errors.append("Temperatura inválida.")
                temperature = DEFAULT_LLM_CONFIG["temperature"]
            if temperature < 0 or temperature > 2:
                errors.append("Temperatura deve estar entre 0 e 2.")

            try:
                chunk_size = int(chunk_raw)
            except ValueError:
                errors.append("Chunk inválido.")
                chunk_size = DEFAULT_LLM_CONFIG["chunk_size"]
            if chunk_size < 200 or chunk_size > 4000:
                errors.append("Chunk deve estar entre 200 e 4000.")

            try:
                chunk_overlap = int(overlap_raw)
            except ValueError:
                errors.append("Overlap inválido.")
                chunk_overlap = DEFAULT_LLM_CONFIG["chunk_overlap"]
            if chunk_overlap < 0 or chunk_overlap >= chunk_size:
                errors.append("Overlap deve ser menor que o chunk e >= 0.")

            if errors:
                error_message = " ".join(errors)
            else:
                llm_config = {
                    "model": model,
                    "temperature": temperature,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
                request.session["llm_config"] = llm_config
                request.session.modified = True
                _apply_runtime_config(llm_config)
                if reindex:
                    try:
                        rebuild_faiss_index(docs_dir)
                        reset_rules_retriever_cache()
                        config_message = "Configurações aplicadas e índice FAISS reindexado."
                    except Exception as exc:  # noqa: BLE001
                        error_message = f"Configurações salvas, mas falha ao reindexar: {exc}"
                else:
                    config_message = "Configurações aplicadas para esta sessão."
        elif action == "upload":
            uploaded = request.FILES.get("doc_file")
            if not uploaded:
                error_message = "Selecione um arquivo PDF ou TXT para enviar."
            else:
                original_name = get_valid_filename(Path(uploaded.name).name)
                ext = Path(original_name).suffix.lower()
                if ext not in ALLOWED_DOC_EXTS:
                    error_message = "Formato não suportado. Envie apenas PDF ou TXT."
                else:
                    target_path = docs_dir / original_name
                    if target_path.exists():
                        stem = target_path.stem
                        suffix = target_path.suffix
                        counter = 1
                        while target_path.exists():
                            target_path = docs_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    with target_path.open("wb") as handle:
                        for chunk in uploaded.chunks():
                            handle.write(chunk)
                    try:
                        ingest_docs_to_faiss([target_path])
                        reset_rules_retriever_cache()
                        upload_message = f"Arquivo {target_path.name} enviado e indexado."
                    except Exception as exc:  # noqa: BLE001
                        error_message = f"Arquivo enviado, mas falha ao indexar no FAISS: {exc}"
        elif action == "delete_doc":
            doc_name = get_valid_filename(request.POST.get("doc_name") or "")
            if not doc_name:
                error_message = "Arquivo inválido para remoção."
            else:
                target_path = docs_dir / doc_name
                if target_path.exists():
                    try:
                        target_path.unlink()
                        rebuild_faiss_index(docs_dir)
                        reset_rules_retriever_cache()
                        upload_message = f"Arquivo {doc_name} removido e índice atualizado."
                    except Exception as exc:  # noqa: BLE001
                        error_message = f"Erro ao remover arquivo ou atualizar índice: {exc}"
                else:
                    error_message = "Arquivo não encontrado."
        else:
            if query:
                try:
                    cache_key = _cache_key("ask", query_raw, request.session.session_key)
                    cached = cache.get(cache_key)
                    if cached:
                        response_text = cached.get("response_text")
                        chart_data = cached.get("chart_data")
                        chart_error = cached.get("chart_error")
                        pdf_report = cached.get("pdf_report")
                    else:
                        if _looks_like_pdf_request(query_raw):
                            pdf_response = generate_pdf_report(query, history=history)
                            payload = _extract_pdf_payload(pdf_response)
                            if payload and payload.get("pdf_base64"):
                                pdf_report = {
                                    "base64": payload.get("pdf_base64"),
                                    "file_name": payload.get("file_name") or "relatorio.pdf",
                                    "title": payload.get("title") or "Relatorio PDF",
                                    "rows_used": payload.get("rows_used") or 0,
                                    "truncated": bool(payload.get("truncated")),
                                }
                                response_text = "Relatorio PDF gerado."
                            else:
                                sql_block = _extract_sql_block(pdf_response)
                                if sql_block:
                                    pdf_report = _build_pdf_report_from_sql(sql_block)
                                if not pdf_report:
                                    fallback = _build_report_sql_from_query(query_raw)
                                    if fallback:
                                        sql_stmt, title = fallback
                                        pdf_report = _build_pdf_report_from_sql(sql_stmt, title=title)
                                if pdf_report:
                                    response_text = "Relatorio PDF gerado."
                                else:
                                    response_text = pdf_response
                        else:
                            response_text = answer_question(query, history=history)

                            # tenta gerar gráfico quando houver SQL sugerido
                            sql_block = _extract_sql_block(response_text)
                            if sql_block:
                                try:
                                    chart_type = _detect_chart_type(query, response_text)
                                    chart_data = _build_chart_data(sql_block, chart_type=chart_type)
                                except Exception as chart_exc:  # noqa: BLE001
                                    chart_error = f"Erro ao gerar gráfico: {chart_exc}"
                            if not chart_data:
                                chart_data = _extract_chart_from_text(response_text)
                            if not chart_data:
                                try:
                                    chart_data = _build_chart_from_description(response_text)
                                except Exception as chart_exc:  # noqa: BLE001
                                    if not chart_error:
                                        chart_error = f"Erro ao gerar gráfico: {chart_exc}"
                            if not chart_data and _looks_like_chart_request(query_raw):
                                try:
                                    chart_data = _build_chart_from_query(query_raw)
                                    if chart_data:
                                        chart_error = None
                                except Exception as chart_exc:  # noqa: BLE001
                                    if not chart_error:
                                        chart_error = f"Erro ao gerar gráfico: {chart_exc}"
                            if not chart_data and _looks_like_chart_request(query_raw):
                                try:
                                    chart_data = _build_chart_from_history(history, response_text)
                                    if chart_data:
                                        chart_error = None
                                except Exception as chart_exc:  # noqa: BLE001
                                    if not chart_error:
                                        chart_error = f"Erro ao gerar gráfico: {chart_exc}"

                        cache.set(
                            cache_key,
                            {
                                "response_text": response_text,
                                "chart_data": chart_data,
                                "chart_error": chart_error,
                                "pdf_report": pdf_report,
                            },
                            timeout=getattr(settings, "ASK_CACHE_TIMEOUT", 900),
                        )

                    history.append(("user", query))
                    history.append(("assistant", response_text))
                    history = history[-12:]
                    request.session["chat_history"] = history
                    request.session.modified = True
                except Exception as exc:  # noqa: BLE001
                    error_message = str(exc)
            query = ""

    # prepara histórico com enumeração de perguntas
    q_num = 0
    for role, content in history:
        if role == "user":
            q_num += 1
            display_history.append({"role": role, "content": content, "number": q_num})
        else:
            display_history.append({"role": role, "content": content, "number": q_num})
    # inverte para mostrar as últimas no topo
    display_history = list(reversed(display_history))
    available_docs = sorted([p.name for p in docs_dir.glob("*") if p.is_file()])
    available_tables = sorted(_get_authorized_tables())

    return render(
        request,
        "core/index.html",
        {
            "response_text": response_text,
            "upload_message": upload_message,
            "config_message": config_message,
            "error_message": error_message,
            "query": query,
            "openai_ready": bool(getattr(settings, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY")),
            "rules_path": getattr(settings, "CORE_RULES_PATH", None) or os.getenv("CORE_RULES_PATH"),
            "using_postgres": "postgresql" in settings.DATABASES.get("default", {}).get("ENGINE", ""),
            "available_docs": available_docs,
            "available_tables": available_tables,
            "docs_dir": str(docs_dir),
            "history": display_history,
            "chart_data": chart_data,
            "chart_error": chart_error,
            "pdf_report": pdf_report,
            "llm_config": llm_config,
        },
    )


@ensure_csrf_cookie
def charts(request):
    error_message = None
    chart_data = None
    chart_error = None

    tables = _get_authorized_tables()
    table = (request.POST.get("table") or request.GET.get("table") or "").strip()
    if not table and "imoveis" in tables:
        table = "imoveis"
    label_col = (request.POST.get("label_col") or "").strip()
    label_value = (request.POST.get("label_value") or "").strip()
    value_col = (request.POST.get("value_col") or "").strip()
    value_value = (request.POST.get("value_value") or "").strip()
    table_cols = [val.strip() for val in request.POST.getlist("table_cols")]
    table_group_by = (request.POST.get("table_group_by") or "").strip()
    table_group_agg = (request.POST.get("table_group_agg") or "count").strip().lower()
    table_group_value = (request.POST.get("table_group_value") or "").strip()
    filter_cols = [val.strip() for val in request.POST.getlist("filter_col")]
    filter_values = [val.strip() for val in request.POST.getlist("filter_value")]
    filter_ops = [val.strip().lower() for val in request.POST.getlist("filter_op")]
    agg = (request.POST.get("agg") or "count").strip().lower()
    chart_type = (request.POST.get("chart_type") or "bar").strip().lower()
    view_mode = (request.POST.get("view_mode") or request.GET.get("view_mode") or "chart").strip().lower()
    limit_raw = request.POST.get("limit") or "20"
    page_raw = request.POST.get("page") or request.GET.get("page") or "1"

    if view_mode not in {"chart", "table"}:
        view_mode = "chart"

    try:
        limit = int(limit_raw)
    except ValueError:
        limit = 20

    try:
        page = int(page_raw)
    except ValueError:
        page = 1
    page = max(1, page)

    limit = max(1, min(limit, 200))
    columns = _get_table_columns(table) if table else []
    numeric_columns = _get_numeric_columns(table) if table else []
    column_display_map = _build_column_display_map(table, columns) if table else {}
    column_display_reverse = {label: col for col, label in column_display_map.items()}
    display_columns = [column_display_map.get(col, col) for col in columns]
    display_numeric_columns = [column_display_map.get(col, col) for col in numeric_columns]
    label_col = _resolve_column_input(label_col, column_display_reverse)
    value_col = _resolve_column_input(value_col, column_display_reverse)
    table_group_by = _resolve_column_input(table_group_by, column_display_reverse)
    table_group_value = _resolve_column_input(table_group_value, column_display_reverse)
    table_cols = [_resolve_column_input(val, column_display_reverse) for val in table_cols]
    filter_cols = [_resolve_column_input(val, column_display_reverse) for val in filter_cols]
    valid_table_cols = [col for col in table_cols if _is_valid_identifier(col) and col in columns]
    label_values = _get_distinct_values(table, label_col) if table and label_col else []
    value_values = _get_distinct_values(table, value_col) if table and value_col else []
    max_filters = max(len(filter_cols), len(filter_values), len(filter_ops), 1)
    filter_rows = []
    for idx in range(max_filters):
        col = filter_cols[idx] if idx < len(filter_cols) else ""
        val = filter_values[idx] if idx < len(filter_values) else ""
        op = filter_ops[idx] if idx < len(filter_ops) else "and"
        if op not in {"and", "or"}:
            op = "and"
        if idx == 0:
            op = "and"
        values, truncated = _get_distinct_values_with_flag(table, col) if table and col else ([], False)
        filter_rows.append(
            {
                "col": col,
                "display_col": column_display_map.get(col, col),
                "value": val,
                "values": values,
                "op": op,
                "allow_text": truncated,
            }
        )

    table_headers: list[str] = []
    table_rows: list[list[str]] = []
    total_rows = 0
    total_pages = 1

    action = request.POST.get("action") or ""
    if request.method == "POST" and action in {"generate_chart", "download_csv"}:
        if not _is_valid_identifier(table) or table not in tables:
            error_message = "Selecione uma tabela válida."
        elif view_mode == "table":
            if table_group_by:
                if not _is_valid_identifier(table_group_by) or table_group_by not in columns:
                    error_message = "Selecione uma coluna válida para agrupar."
                elif table_group_agg not in {"count", "sum", "avg"}:
                    error_message = "Selecione uma agregação válida para o agrupamento."
                elif table_group_agg == "count" and table_group_value:
                    if not _is_valid_identifier(table_group_value) or table_group_value not in columns:
                        error_message = "Selecione uma coluna válida para COUNT."
                elif table_group_agg != "count" and (
                    not _is_valid_identifier(table_group_value) or table_group_value not in numeric_columns
                ):
                    error_message = "Selecione uma coluna numérica válida para agregação."
            else:
                if not table_cols:
                    error_message = "Selecione ao menos uma coluna para a tabela."
                elif len(valid_table_cols) != len(table_cols):
                    error_message = "Selecione colunas válidas para a tabela."
        else:
            if not _is_valid_identifier(label_col):
                error_message = "Selecione uma coluna válida para o eixo X."
            elif agg not in {"count", "sum", "avg"}:
                error_message = "Selecione uma agregação válida."
            elif agg == "count" and value_col and (not _is_valid_identifier(value_col) or value_col not in columns):
                error_message = "Selecione uma coluna válida para COUNT."
            elif agg != "count" and not _is_valid_identifier(value_col):
                error_message = "Selecione uma coluna numérica válida para agregação."
            elif agg != "count" and numeric_columns and value_col not in numeric_columns:
                error_message = "Selecione uma coluna numérica válida para agregação."

        if not error_message:
            filter_error = None
            for row in filter_rows:
                col = row["col"]
                val = row["value"]
                if not col and not val:
                    continue
                if not col:
                    filter_error = "Selecione uma coluna válida para o filtro."
                    break
                if not _is_valid_identifier(col) or col not in columns:
                    filter_error = "Selecione uma coluna válida para o filtro."
                    break
            if filter_error:
                error_message = filter_error

        if error_message:
            return render(
                request,
                "core/charts.html",
                {
                    "tables": tables,
                    "columns": columns,
                    "selected_table": table,
                    "selected_label": column_display_map.get(label_col, label_col),
                    "selected_label_value": label_value,
                    "selected_value": column_display_map.get(value_col, value_col),
                    "selected_value_value": value_value,
                    "selected_agg": agg,
                    "selected_chart_type": chart_type,
                    "selected_limit": limit,
                    "view_mode": view_mode,
                    "page": page,
                    "total_pages": total_pages,
                    "total_rows": total_rows,
                    "label_values": label_values,
                    "value_values": value_values,
                    "filter_rows": filter_rows,
                    "numeric_columns": numeric_columns,
                    "selected_table_cols": valid_table_cols,
                    "selected_table_group_by": column_display_map.get(table_group_by, table_group_by),
                    "selected_table_group_agg": table_group_agg,
                    "selected_table_group_value": column_display_map.get(table_group_value, table_group_value),
                    "chart_data": chart_data,
                    "table_headers": table_headers,
                    "table_rows": table_rows,
                    "chart_error": chart_error,
                    "error_message": error_message,
                    "column_display_map": column_display_map,
                    "display_columns": display_columns,
                    "display_numeric_columns": display_numeric_columns,
                },
            )

        try:
            params: list[str] = []
            where_clauses = []
            filter_expr = ""
            table_prefix = f"{table}." if _is_valid_identifier(table) else ""
            for row in filter_rows:
                col = row["col"]
                val = row["value"]
                if not col or not val:
                    continue
                clause = f"{table_prefix}{col} = %s"
                if not filter_expr:
                    filter_expr = clause
                else:
                    op = row.get("op", "and").upper()
                    if op not in {"AND", "OR"}:
                        op = "AND"
                    filter_expr = f"{filter_expr} {op} {clause}"
                params.append(val)
            if filter_expr:
                where_clauses.append(f"({filter_expr})")

            with connections["default"].cursor() as cursor:
                if view_mode == "table":
                    if table_group_by:
                        select_expr = _build_agg_expression(table_group_agg, table_group_value)
                        base_sql = f"SELECT {table_group_by} AS label, {select_expr} AS value FROM {table}"
                        if where_clauses:
                            base_sql += " WHERE " + " AND ".join(where_clauses)
                        base_sql += f" GROUP BY {table_group_by}"
                        order_sql = f"{base_sql} ORDER BY value DESC"
                        table_headers = [
                            table_group_by,
                            f"{_agg_label_pt(table_group_agg)}({table_group_value})"
                            if table_group_value
                            else f"{_agg_label_pt(table_group_agg)}(*)",
                        ]

                        if action == "download_csv":
                            cursor.execute(order_sql, params)
                            rows = cursor.fetchall()
                            response = HttpResponse(content_type="text/csv")
                            filename = get_valid_filename(
                                f"{table}_{table_group_by}_{table_group_agg}.csv"
                            ) or "dados.csv"
                            response["Content-Disposition"] = f'attachment; filename="{filename}"'
                            writer = csv.writer(response)
                            writer.writerow(table_headers)
                            for row in rows:
                                writer.writerow([row[0], row[1]])
                            return response

                        count_sql = f"SELECT COUNT(*) FROM ({base_sql}) AS subquery"
                        cursor.execute(count_sql, params)
                        total_rows = int(cursor.fetchone()[0] or 0)
                        total_pages = max(1, (total_rows + TABLE_PAGE_SIZE - 1) // TABLE_PAGE_SIZE)
                        page = min(page, total_pages)
                        offset = (page - 1) * TABLE_PAGE_SIZE
                        cursor.execute(f"{order_sql} LIMIT %s OFFSET %s", params + [TABLE_PAGE_SIZE, offset])
                        rows = cursor.fetchall()
                        table_rows = [[_format_pt_br(row[0]), _format_pt_br(row[1])] for row in rows]
                    else:
                        selected_cols = valid_table_cols
                        fk_map = _get_foreign_key_map(table)
                        select_exprs: list[str] = []
                        select_aliases: list[str] = []
                        table_headers = []
                        join_clauses: list[str] = []
                        for col in selected_cols:
                            ref = fk_map.get(col)
                            if ref:
                                ref_table, ref_col = ref
                                display_col = _get_display_column(ref_table)
                                if display_col:
                                    alias = f"{ref_table}_{col}"
                                    join_clauses.append(
                                        f"LEFT JOIN {ref_table} {alias} ON {table}.{col} = {alias}.{ref_col}"
                                    )
                                    select_exprs.append(f"{alias}.{display_col} AS {col}_nome")
                                    select_aliases.append(f"{col}_nome")
                                    if col.endswith("_id"):
                                        table_headers.append(col[:-3])
                                    else:
                                        table_headers.append(col)
                                    continue
                            select_exprs.append(f"{table}.{col}")
                            select_aliases.append(col)
                            table_headers.append(col)

                        base_sql = f"SELECT {', '.join(select_exprs)} FROM {table}"
                        if join_clauses:
                            base_sql += " " + " ".join(join_clauses)
                        if where_clauses:
                            base_sql += " WHERE " + " AND ".join(where_clauses)
                        order_sql = f"{base_sql} ORDER BY 1"

                        if action == "download_csv":
                            cursor.execute(order_sql, params)
                            rows = cursor.fetchall()
                            response = HttpResponse(content_type="text/csv")
                            filename = get_valid_filename(f"{table}_tabela.csv") or "dados.csv"
                            response["Content-Disposition"] = f'attachment; filename="{filename}"'
                            writer = csv.writer(response)
                            writer.writerow(table_headers)
                            for row in rows:
                                writer.writerow(list(row))
                            return response

                        count_sql = f"SELECT COUNT(*) FROM ({base_sql}) AS subquery"
                        cursor.execute(count_sql, params)
                        total_rows = int(cursor.fetchone()[0] or 0)
                        total_pages = max(1, (total_rows + TABLE_PAGE_SIZE - 1) // TABLE_PAGE_SIZE)
                        page = min(page, total_pages)
                        offset = (page - 1) * TABLE_PAGE_SIZE
                        cursor.execute(f"{order_sql} LIMIT %s OFFSET %s", params + [TABLE_PAGE_SIZE, offset])
                        rows = cursor.fetchall()
                        table_rows = [[_format_pt_br(cell) for cell in row] for row in rows]
                else:
                    select_expr = _build_agg_expression(agg, value_col)
                    base_sql = f"SELECT {label_col} AS label, {select_expr} AS value FROM {table}"
                    if label_value:
                        where_clauses.append(f"{label_col} = %s")
                        params.append(label_value)
                    if value_value and _is_valid_identifier(value_col):
                        where_clauses.append(f"{value_col} = %s")
                        params.append(value_value)
                    if where_clauses:
                        base_sql += " WHERE " + " AND ".join(where_clauses)
                    base_sql += f" GROUP BY {label_col}"
                    order_sql = f"{base_sql} ORDER BY value DESC"

                    cursor.execute(f"{order_sql} LIMIT %s", params + [limit])
                    rows = cursor.fetchall()
                    cols = [col[0] for col in (cursor.description or [])]
                    chart_data = _build_chart_data_from_rows(rows, cols, chart_type)
                    if chart_data:
                        chart_data["title"] = f"{_agg_label_pt(agg)} por {label_col}"

                    table_rows = [[_format_pt_br(row[0]), _format_pt_br(row[1])] for row in rows]
                    total_rows = len(table_rows)
                    total_pages = 1
                    value_header = (
                        f"{_agg_label_pt(agg)}({value_col})" if value_col else f"{_agg_label_pt(agg)}(*)"
                    )
                    table_headers = [label_col, value_header]

                    if action == "download_csv":
                        response = HttpResponse(content_type="text/csv")
                        filename = get_valid_filename(f"{table}_{label_col}_{agg}.csv") or "dados.csv"
                        response["Content-Disposition"] = f'attachment; filename="{filename}"'
                        writer = csv.writer(response)
                        writer.writerow(table_headers)
                        for row in rows:
                            writer.writerow([row[0], row[1]])
                        return response
        except Exception as exc:  # noqa: BLE001
            chart_error = f"Erro ao gerar gráfico: {exc}"

    return render(
        request,
        "core/charts.html",
        {
            "tables": tables,
            "columns": columns,
            "selected_table": table,
            "selected_label": column_display_map.get(label_col, label_col),
            "selected_label_value": label_value,
            "selected_value": column_display_map.get(value_col, value_col),
            "selected_value_value": value_value,
            "selected_table_cols": valid_table_cols,
            "selected_agg": agg,
            "selected_chart_type": chart_type,
            "selected_limit": limit,
            "view_mode": view_mode,
            "page": page,
            "total_pages": total_pages,
            "total_rows": total_rows,
            "label_values": label_values,
            "value_values": value_values,
            "filter_rows": filter_rows,
            "numeric_columns": numeric_columns,
            "chart_data": chart_data,
            "table_headers": table_headers,
            "table_rows": table_rows,
            "selected_table_group_by": column_display_map.get(table_group_by, table_group_by),
            "selected_table_group_agg": table_group_agg,
            "selected_table_group_value": column_display_map.get(table_group_value, table_group_value),
            "chart_error": chart_error,
            "error_message": error_message,
            "column_display_map": column_display_map,
            "display_columns": display_columns,
            "display_numeric_columns": display_numeric_columns,
        },
    )
