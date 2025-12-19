import os
import re
from pathlib import Path

from django.conf import settings
from django.db import connections
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from .langchain_agents import answer_question

SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
MAX_CHART_ROWS = 200  # evita carregar milhões de linhas no gráfico


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


@ensure_csrf_cookie
def index(request):
    response_text = None
    error_message = None
    chart_data = None
    chart_error = None
    query = ""
    docs_dir = Path(getattr(settings, "RAG_DOCS_DIR", settings.BASE_DIR / "documents"))
    docs_dir.mkdir(parents=True, exist_ok=True)
    available_docs = sorted([p.name for p in docs_dir.glob("*") if p.is_file()])
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
            display_history.append({"role": role, "content": content, "number": None})
    # inverte para mostrar as últimas no topo
    display_history = list(reversed(display_history))

    return render(
        request,
        "core/index.html",
        {
            "response_text": response_text,
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
        },
    )
