from __future__ import annotations

import os
from pathlib import Path

from django.conf import settings
from django.shortcuts import render
from django.utils.text import get_valid_filename
from django.views.decorators.csrf import ensure_csrf_cookie

from ..agents import AnalyticsOrchestrator
from ..rag_index import ingest_docs_to_faiss, rebuild_faiss_index
from ..services.sql_introspection import get_authorized_tables, get_table_columns


DEFAULT_LLM_CONFIG = {
    "model": getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
    "temperature": getattr(settings, "OPENAI_TEMPERATURE", 0.1),
    "chunk_size": getattr(settings, "RAG_CHUNK_SIZE", 700),
    "chunk_overlap": getattr(settings, "RAG_CHUNK_OVERLAP", 150),
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


def _set_last_sql(request, sql: str | None) -> None:
    if sql:
        request.session["last_sql"] = sql
        request.session.modified = True


def _get_last_sql(request) -> str | None:
    return request.session.get("last_sql")


def _set_last_table(request, table: str | None) -> None:
    if table:
        request.session["last_table"] = table
        request.session.modified = True


def _get_last_table(request) -> str | None:
    return request.session.get("last_table")


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
            request.session.pop("last_sql", None)
            request.session.pop("last_table", None)
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
                        config_message = "Configurações aplicadas e índice FAISS reindexado."
                    except Exception as exc:
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
                if ext not in {".pdf", ".txt"}:
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
                        upload_message = f"Arquivo {target_path.name} enviado e indexado."
                    except Exception as exc:
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
                        upload_message = f"Arquivo {doc_name} removido e índice atualizado."
                    except Exception as exc:
                        error_message = f"Erro ao remover arquivo ou atualizar índice: {exc}"
                else:
                    error_message = "Arquivo não encontrado."

        else:
            if query:
                try:
                    orchestrator = AnalyticsOrchestrator()
                    payload = orchestrator.ask(
                        query,
                        history=history,
                        session_key=request.session.session_key,
                        last_sql=_get_last_sql(request),
                        last_table=_get_last_table(request),
                    )

                    response_text = payload.get("response_text")
                    chart_data = payload.get("chart_data")
                    chart_error = payload.get("chart_error")
                    pdf_report = payload.get("pdf_report")

                    _set_last_sql(request, payload.get("last_sql"))
                    _set_last_table(request, payload.get("last_table"))

                    history.append(("user", query))
                    history.append(("assistant", response_text or ""))
                    history = history[-12:]
                    request.session["chat_history"] = history
                    request.session.modified = True

                except Exception as exc:
                    error_message = str(exc)

            query = ""

    q_num = 0
    for role, content in history:
        if role == "user":
            q_num += 1
            display_history.append({"role": role, "content": content, "number": q_num})
        else:
            display_history.append({"role": role, "content": content, "number": q_num})

    display_history = list(reversed(display_history))
    available_docs = sorted([p.name for p in docs_dir.glob("*") if p.is_file()])
    available_tables = sorted(get_authorized_tables())
    table_columns_map: dict[str, list[str]] = {}
    for tbl in available_tables:
        try:
            table_columns_map[tbl] = get_table_columns(tbl)
        except Exception:
            table_columns_map[tbl] = []

    doc_previews: dict[str, str] = {}
    for doc in available_docs:
        preview = ""
        doc_path = docs_dir / doc
        if doc_path.exists() and doc_path.is_file():
            if doc_path.suffix.lower() == ".txt":
                try:
                    with doc_path.open("r", encoding="utf-8", errors="ignore") as f:
                        preview = f.read(1200)
                except Exception:
                    preview = "Não foi possível ler o conteúdo deste documento."
            else:
                preview = "Pré-visualização indisponível para este formato."
        doc_previews[doc] = preview

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
            "table_columns_map": table_columns_map,
            "doc_previews": doc_previews,
            "docs_dir": str(docs_dir),
            "history": display_history,
            "chart_data": chart_data,
            "chart_error": chart_error,
            "pdf_report": pdf_report,
            "llm_config": llm_config,
        },
    )
