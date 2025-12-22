import os
import re
from pathlib import Path

from django.conf import settings
from django.db import connections
from django.shortcuts import render
from django.utils.text import get_valid_filename
from django.views.decorators.csrf import ensure_csrf_cookie

from .langchain_agents import answer_question, reset_rules_retriever_cache
from .rag_index import ingest_docs_to_faiss, rebuild_faiss_index

SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
CHART_LIST_RE = re.compile(r"^\s*\d+\.\s*(.+?):\s*([0-9\.,]+)", re.MULTILINE)
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_CHART_ROWS = 200  # evita carregar milhões de linhas no gráfico
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


def _build_chart_data(sql: str, chart_type: str = "bar") -> dict | None:
    # Garante apenas SELECT simples
    sql_clean = sql.strip().rstrip(";")
    if not sql_clean.lower().startswith("select"):
        raise ValueError("Somente SELECT é permitido para gerar gráficos.")

    with connections["default"].cursor() as cursor:
        cursor.execute(sql_clean)
        rows = cursor.fetchmany(MAX_CHART_ROWS + 1)
        cols = [col[0] for col in (cursor.description or [])]

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


@ensure_csrf_cookie
def index(request):
    response_text = None
    upload_message = None
    config_message = None
    error_message = None
    chart_data = None
    chart_error = None
    query = ""
    docs_dir = Path(getattr(settings, "RAG_DOCS_DIR", settings.BASE_DIR / "documents"))
    docs_dir.mkdir(parents=True, exist_ok=True)
    llm_config = _load_llm_config(request)
    history = request.session.get("chat_history", [])
    display_history = []

    if request.method == "POST":
        action = request.POST.get("action", "ask")
        query = (request.POST.get("query") or "").strip()
        if action == "clear":
            history = []
            request.session["chat_history"] = history
            request.session.modified = True
            query = ""
            response_text = None
            error_message = None
            chart_data = None
            chart_error = None
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
                    response_text = answer_question(query, history=history)
                    history.append(("user", query))
                    history.append(("assistant", response_text))
                    history = history[-12:]
                    request.session["chat_history"] = history
                    request.session.modified = True

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
            "docs_dir": str(docs_dir),
            "history": display_history,
            "chart_data": chart_data,
            "chart_error": chart_error,
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
    label_col = (request.POST.get("label_col") or "").strip()
    value_col = (request.POST.get("value_col") or "").strip()
    agg = (request.POST.get("agg") or "count").strip().lower()
    chart_type = (request.POST.get("chart_type") or "bar").strip().lower()
    limit_raw = request.POST.get("limit") or "20"

    try:
        limit = int(limit_raw)
    except ValueError:
        limit = 20

    limit = max(1, min(limit, 200))
    columns = _get_table_columns(table) if table else []

    if request.method == "POST" and request.POST.get("action") == "generate_chart":
        if not _is_valid_identifier(table) or table not in tables:
            error_message = "Selecione uma tabela válida."
        elif not _is_valid_identifier(label_col):
            error_message = "Selecione uma coluna válida para o eixo X."
        elif agg not in {"count", "sum", "avg"}:
            error_message = "Selecione uma agregação válida."
        elif agg != "count" and not _is_valid_identifier(value_col):
            error_message = "Selecione uma coluna numérica para agregação."
        else:
            if agg == "count":
                select_expr = "COUNT(*)"
            else:
                select_expr = f"{agg.upper()}({value_col})"

            sql = (
                f"SELECT {label_col} AS label, {select_expr} AS value "
                f"FROM {table} GROUP BY {label_col} ORDER BY value DESC LIMIT {limit}"
            )
            try:
                with connections["default"].cursor() as cursor:
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                labels = [str(row[0]) for row in rows]
                values = [float(row[1]) for row in rows]
                chart_data = {
                    "labels": labels,
                    "values": values,
                    "type": chart_type,
                    "title": f"{agg.upper()} por {label_col}",
                    "truncated": False,
                    "max_rows": limit,
                }
            except Exception as exc:  # noqa: BLE001
                chart_error = f"Erro ao gerar gráfico: {exc}"

    return render(
        request,
        "core/charts.html",
        {
            "tables": tables,
            "columns": columns,
            "selected_table": table,
            "selected_label": label_col,
            "selected_value": value_col,
            "selected_agg": agg,
            "selected_chart_type": chart_type,
            "selected_limit": limit,
            "chart_data": chart_data,
            "chart_error": chart_error,
            "error_message": error_message,
        },
    )
