from __future__ import annotations

import hashlib
import re
from typing import Any

from django.conf import settings
from django.core.cache import cache
from django.db import connections

from .conversation_state import ConversationState
from .semantic_interpreter import interpret
from .query_planner import plan as build_plan
from .sql_executor import (
    CountMovimentacoesPorDiaExecutor,
    CountVendasPorDiaExecutor,
    ListProdutosPorDiaExecutor,
    SumProdutosVendidosPorDiaExecutor,
    TopProdutosPorDiaExecutor,
    TopProdutosPorPeriodoExecutor,
    TotalVendasFinanceiroPorDiaExecutor,
    execute,
)
from .coherence_validator import validate
from .response_generator import generate

from ..services.chart_builder import (
    build_chart_data,
    build_chart_from_description,
    build_chart_from_history,
    build_chart_from_query,
    detect_chart_type,
    extract_chart_from_text,
    looks_like_chart_request,
)
from ..services.export_service import (
    build_pdf_report_from_sql,
    extract_pdf_payload,
    extract_sql_block,
    looks_like_pdf_request,
)
from ..services.sql_introspection import get_authorized_tables
from ..services.table_builder import (
    build_report_sql_from_query,
    normalize_text,
    parse_br_date,
)


# ============================================================
# Exceptions
# ============================================================

class UnsupportedQuestionError(Exception):
    """Pergunta fora do escopo analítico suportado."""


# ============================================================
# Cache helpers
# ============================================================

def _cache_key(prefix: str, query: str, session_key: str | None, history: list | None) -> str:
    hist_digest = hashlib.sha256(str(history or []).encode("utf-8")).hexdigest()[:16]
    base = f"{prefix}|{session_key or 'anon'}|{normalize_text(query)}|h={hist_digest}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


# ============================================================
# Fallback EXTREMAMENTE RESTRITO (somente contagem simples)
# ============================================================

def _fallback_answer_vendas_por_dia(query: str) -> tuple[str | None, str | None]:
    qn = normalize_text(query)

    # ❌ Nunca interceptar perguntas financeiras
    if any(token in qn for token in ("reais", "r$", "valor", "soma", "somatorio")):
        return None, None
    if "total" in qn and any(token in qn for token in ("reais", "r$", "valor")):
        return None, None

    if "venda" not in qn and "vendas" not in qn:
        return None, None

    dia = parse_br_date(query)
    if not dia:
        return None, None

    if "movimentacao_estoque" not in get_authorized_tables():
        return None, None

    sql = """
        SELECT COUNT(*)::int
        FROM movimentacao_estoque
        WHERE tipo_movimento = 'saida'
          AND DATE(data_movimento) = %s
    """.strip()

    with connections["default"].cursor() as cursor:
        cursor.execute(sql, [dia])
        total = cursor.fetchone()[0] or 0

    return f"Foram realizadas {total} vendas no dia {dia}.", sql


# ============================================================
# State builder (historico incremental para manter contexto)
# ============================================================

def _build_state(
    history: list[tuple[str, str]] | None,
    current: dict,
) -> ConversationState:
    state = ConversationState()

    history = history or []
    for idx, (role, content) in enumerate(history):
        if role != "user" or not content:
            continue

        past = interpret(content, history[:idx])
        state.update(
            {
                "raw_question": past.get("raw_question"),
                "intent": past.get("intent"),
                "objeto": past.get("objeto"),
                "tipo_movimento": past.get("tipo_movimento"),
                "data": past.get("data"),
                "metric": past.get("metric"),
                "rank_limit": past.get("rank_limit"),
                "date_anchor": past.get("date_anchor"),
                "rank_position": past.get("rank_position"),
                "date_start": past.get("date_start"),
                "date_end": past.get("date_end"),
                "periodo": past.get("periodo"),
            }
        )

    state.update(current)
    return state



# ============================================================
# Executor selector (EXPLÍCITO)
# ============================================================

def _select_executor(semantic: dict) -> object | None:
    intent = semantic.get("intent")
    objeto = semantic.get("objeto")
    tipo_movimento = semantic.get("tipo_movimento")
    data = semantic.get("data")
    metric = semantic.get("metric")
    date_start = semantic.get("date_start")
    date_end = semantic.get("date_end")

    # CONTAGEM DE MOVIMENTAÇÕES
    if intent == "count" and objeto == "movimentacao" and tipo_movimento in {"entrada", "saida"} and data:
        return CountMovimentacoesPorDiaExecutor()

    # LISTAGEM DE PRODUTOS (VENDA / ENTRADA)
    if intent == "list" and objeto == "produto" and tipo_movimento in {"entrada", "saida"} and data:
        return ListProdutosPorDiaExecutor()

    # CONTAGEM DE VENDAS
    if intent == "count" and objeto == "venda" and data:
        return CountVendasPorDiaExecutor()

    # TOTAL FINANCEIRO
    if intent == "sum" and objeto == "venda" and data:
        return TotalVendasFinanceiroPorDiaExecutor()

    if intent == "sum" and objeto == "produto" and metric == "quantidade" and data:
        return SumProdutosVendidosPorDiaExecutor()

    if intent == "top" and objeto == "produto" and date_start and date_end:
        return TopProdutosPorPeriodoExecutor()

    if intent == "top" and objeto == "produto":
        return TopProdutosPorDiaExecutor()

    return None


# ============================================================
# Pipeline determinístico (NÚCLEO)
# ============================================================

class AnalyticsPipeline:
    def run(self, question: str, history: list[tuple[str, str]] | None = None) -> str:
        semantic_raw = interpret(question, history)
        state = _build_state(history, semantic_raw)

        semantic = state.to_dict()

        if semantic.get("needs_confirmation"):
            return semantic.get("confirmation") or "Qual ano?"

        if semantic.get("unsupported"):
            raise UnsupportedQuestionError(
                "Essa pergunta não pode ser respondida por critérios analíticos."
            )

        # ✅ executor escolhido com estado CONSOLIDADO
        executor = _select_executor(semantic)
        if not executor:
            raise UnsupportedQuestionError(
                "Pergunta fora do escopo analítico suportado."
            )

        # plano lógico agora é apenas transporte
        query_plan = build_plan(semantic, state)
        query_plan["executor"] = executor

        result = execute(query_plan)
        result = validate(result, semantic)
        return generate(result, semantic)



def run_agents_pipeline(question: str, history: list[tuple[str, str]] | None = None) -> str:
    try:
        return AnalyticsPipeline().run(question, history=history)
    except UnsupportedQuestionError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        return f"Erro ao processar pergunta: {exc}"


# ============================================================
# Orchestrator de UI (chat / gráfico / PDF)
# ============================================================

class AnalyticsOrchestrator:
    def ask(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        session_key: str | None = None,
        last_sql: str | None = None,
        last_table: str | None = None,
    ) -> dict[str, Any]:

        query_raw = (question or "").strip()
        history = history or []

        if not query_raw:
            return {
                "response_text": "",
                "chart_data": None,
                "chart_error": None,
                "pdf_report": None,
                "last_sql": last_sql,
                "last_table": last_table,
            }

        ck = _cache_key("ask", query_raw, session_key, history)
        cached = cache.get(ck)
        if cached:
            return cached

        response_text = None
        chart_data = None
        chart_error = None
        pdf_report = None
        current_last_sql = last_sql
        current_last_table = last_table

        # -------------------------
        # PDF
        # -------------------------
        if looks_like_pdf_request(query_raw):
            response_text = run_agents_pipeline(query_raw, history=history)

            if "fora do escopo" in response_text.lower() or "nao pode ser respondida" in response_text.lower():
                payload = {
                    "response_text": response_text,
                    "chart_data": None,
                    "chart_error": None,
                    "pdf_report": None,
                    "last_sql": current_last_sql,
                    "last_table": current_last_table,
                }
                cache.set(ck, payload, timeout=getattr(settings, "ASK_CACHE_TIMEOUT", 900))
                return payload

            payload = extract_pdf_payload(response_text)
            if payload and payload.get("pdf_base64"):
                pdf_report = {
                    "base64": payload["pdf_base64"],
                    "file_name": payload.get("file_name") or "relatorio.pdf",
                    "title": payload.get("title") or "Relatorio PDF",
                    "rows_used": payload.get("rows_used") or 0,
                    "truncated": bool(payload.get("truncated")),
                }
                response_text = "Relatório PDF gerado."

        # -------------------------
        # Pergunta normal
        # -------------------------
        else:
            fb_text, fb_sql = _fallback_answer_vendas_por_dia(query_raw)
            if fb_text:
                response_text = fb_text
                current_last_sql = fb_sql
                current_last_table = "movimentacao_estoque"
            else:
                response_text = run_agents_pipeline(query_raw, history=history)

            # Bloqueio → NÃO gerar SQL, gráfico ou PDF
            if response_text and (
                "fora do escopo" in response_text.lower()
                or "nao pode ser respondida" in response_text.lower()
            ):
                payload = {
                    "response_text": response_text,
                    "chart_data": None,
                    "chart_error": None,
                    "pdf_report": None,
                    "last_sql": current_last_sql,
                    "last_table": current_last_table,
                }
                cache.set(ck, payload, timeout=getattr(settings, "ASK_CACHE_TIMEOUT", 900))
                return payload

            sql_block = extract_sql_block(response_text or "")
            if sql_block:
                current_last_sql = sql_block
                current_last_table = _try_extract_table_from_sql(sql_block)

            # Gráficos (somente se houver SQL)
            if sql_block:
                try:
                    chart_type = detect_chart_type(query_raw, response_text or "")
                    chart_data = build_chart_data(sql_block, chart_type=chart_type)
                except Exception as exc:  # noqa: BLE001
                    chart_error = f"Erro ao gerar gráfico: {exc}"

            if not chart_data:
                chart_data = extract_chart_from_text(response_text or "")

            if not chart_data:
                try:
                    chart_data = build_chart_from_description(response_text or "")
                except Exception:
                    pass

            if not chart_data and looks_like_chart_request(query_raw):
                try:
                    chart_data = build_chart_from_query(query_raw)
                except Exception:
                    pass

            if not chart_data and looks_like_chart_request(query_raw):
                try:
                    chart_data = build_chart_from_history(history, response_text or "")
                except Exception:
                    pass

        payload = {
            "response_text": response_text,
            "chart_data": chart_data,
            "chart_error": chart_error,
            "pdf_report": pdf_report,
            "last_sql": current_last_sql,
            "last_table": current_last_table,
        }

        cache.set(ck, payload, timeout=getattr(settings, "ASK_CACHE_TIMEOUT", 900))
        return payload


# ============================================================
# Utils
# ============================================================

def _try_extract_table_from_sql(sql: str) -> str | None:
    if not sql:
        return None
    m = re.search(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", sql, re.IGNORECASE)
    return m.group(1) if m else None

def reset_rules_retriever_cache() -> None:
    """
    Compatibilidade com versões antigas.
    Atualmente não há cache semântico ou retriever a ser limpo.
    """
    return None
