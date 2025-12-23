"""
Helpers to run LangChain agents against Postgres and a rules document (RAG).
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus

from django.conf import settings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.utilities import SQLDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sqlalchemy.exc import SAWarning

warnings.filterwarnings("ignore", message="Did not recognize type 'geometry'", category=SAWarning)


def get_openai_api_key() -> str:
    api_key = getattr(settings, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Chave OpenAI não configurada.")
    return api_key


def get_chat_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=get_openai_api_key(),
        model=getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
        temperature=getattr(settings, "OPENAI_TEMPERATURE", 0.1),
    )


def _postgres_uri_from_settings() -> str:
    db_cfg = settings.DATABASES.get("default", {})
    engine = db_cfg.get("ENGINE", "")
    if "postgresql" not in engine:
        raise RuntimeError("Configure o banco default como Postgres para usar o agente SQL.")

    name = db_cfg.get("NAME")
    user = db_cfg.get("USER")
    password = db_cfg.get("PASSWORD")
    host = db_cfg.get("HOST") or "localhost"
    port = db_cfg.get("PORT") or "5432"

    missing = [key for key, value in [("NAME", name), ("USER", user), ("PASSWORD", password)] if not value]
    if missing:
        raise RuntimeError(f"Campos ausentes no DATABASES['default']: {', '.join(missing)}.")

    return f"postgresql+psycopg2://{user}:{quote_plus(password)}@{host}:{port}/{name}"


def build_sql_agent():
    db = SQLDatabase.from_uri(_postgres_uri_from_settings())
    llm = get_chat_llm()
    return create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=False)


def ask_database(question: str) -> str:
    if not question:
        raise ValueError("A pergunta para o banco de dados não pode ser vazia.")

    print("[agent] consultando banco de dados com pergunta:", question)
    agent = build_sql_agent()
    result: Dict[str, Any] = agent.invoke({"input": question})
    if isinstance(result, dict):
        output = result.get("output") or result.get("result")
        if output:
            return str(output)
    # fallback se não veio nada utilizável
    return "Não foi possível responder com base no banco de dados."


@lru_cache(maxsize=1)
def _build_rules_retriever():
    rules_path_raw = getattr(settings, "CORE_RULES_PATH", None) or os.getenv("CORE_RULES_PATH")
    if not rules_path_raw:
        raise RuntimeError("Defina CORE_RULES_PATH com o caminho do documento de regras para o RAG.")

    rules_path = Path(rules_path_raw)
    if not rules_path.exists():
        raise FileNotFoundError(f"Arquivo de regras não encontrado em {rules_path}.")

    embeddings = OpenAIEmbeddings(api_key=get_openai_api_key())
    index_path = Path(getattr(settings, "RAG_INDEX_PATH", Path(settings.BASE_DIR) / "documents" / "faiss_index"))
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists():
        try:
            vs = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
            return vs.as_retriever(search_kwargs={"k": 4})
        except Exception as exc:  # noqa: BLE001
            print("[rag] falha ao carregar índice FAISS salvo, reconstruindo:", exc)

    # Ajuste para PDF ou texto simples
    if rules_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(rules_path))
    else:
        loader = TextLoader(str(rules_path), encoding="utf-8")

    docs = loader.load()
    chunk_size = getattr(settings, "RAG_CHUNK_SIZE", 700)
    chunk_overlap = getattr(settings, "RAG_CHUNK_OVERLAP", 150)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(splits, embedding=embeddings)
    # Persiste para uso nos próximos carregamentos
    vector_store.save_local(str(index_path))
    return vector_store.as_retriever(search_kwargs={"k": 4})


def reset_rules_retriever_cache() -> None:
    _build_rules_retriever.cache_clear()


def build_rules_agent():
    """Agent autônomo que consulta o documento de regras via ferramenta de busca."""
    retriever = _build_rules_retriever()
    llm = get_chat_llm()

    def search_rules(query: str) -> str:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
        else:
            docs = retriever.invoke(query)
        if not docs:
            return "Nenhum trecho relevante encontrado."
        return "\n\n".join(f"- {doc.page_content}" for doc in docs)

    rules_tool = Tool(
        name="buscar_regras",
        func=search_rules,
        description="Busca trechos relevantes no documento de regras para responder perguntas.",
    )

    system_prompt = (
        "Você é um agente autônomo que responde estritamente com base no documento de regras local. "
        "Não use conhecimento externo ou da internet. Quando precisar de contexto, chame a ferramenta "
        "'buscar_regras'. Explique de forma curta e cite os trechos encontrados."
    )

    return create_agent(
        model=llm,
        tools=[rules_tool],
        system_prompt=system_prompt,
    )


def ask_rules_document(question: str) -> str:
    if not question:
        raise ValueError("A pergunta para o documento de regras não pode ser vazia.")

    print("[agent] consultando documento de regras com pergunta:", question)
    agent = build_rules_agent()
    result = agent.invoke({"messages": [("user", question)]})
    if isinstance(result, dict) and "messages" in result:
        messages = result.get("messages") or []
        if messages:
            # O último AIMessage contém a resposta final.
            final_msg = messages[-1]
            try:
                content = final_msg.content  # type: ignore[assignment]
            except Exception:
                content = str(final_msg)
            if content:
                return content
    # fallback se nada encontrado
    return "Não foi possível responder com base no documento."


def build_code_agent():
    """
    Agente especializado em gerar código SQL ou Python.
    Usa ferramentas para buscar regras e consultar o banco antes de propor código.
    """
    llm = get_chat_llm()

    db_tool = Tool(
        name="consultar_banco",
        func=ask_database,
        description="Use para verificar esquemas, colunas ou contagens no Postgres antes de propor SQL.",
    )
    rules_tool = Tool(
        name="consultar_regras",
        func=ask_rules_document,
        description="Use para entender regras do Manual antes de gerar código.",
    )

    system_prompt = (
        "Você é um assistente de código. Gere SQL ou Python conforme solicitado. "
        "Use as ferramentas para obter contexto (regras ou detalhes do banco) antes de propor código. "
        "Quando criar SQL, prefira consultas de leitura ou criação de tabelas/visões conforme solicitado, "
        "não faça alterações destrutivas. Responda com blocos de código bem formatados."
    )

    return create_agent(
        model=llm,
        tools=[db_tool, rules_tool],
        system_prompt=system_prompt,
    )


def generate_code(question: str, history: List[Tuple[str, str]] | None = None) -> str:
    print("[agent] gerador de código recebendo pergunta:", question)
    agent = build_code_agent()
    msgs = history[:] if history else []
    msgs.append(("user", question))
    result = agent.invoke({"messages": msgs})
    if isinstance(result, dict) and "messages" in result:
        messages = result.get("messages") or []
        if messages:
            final_msg = messages[-1]
            try:
                content = final_msg.content  # type: ignore[assignment]
            except Exception:
                content = str(final_msg)
            if content:
                return content
    return "Não foi possível gerar o código solicitado."


def build_chart_agent():
    """
    Agente para sugerir gráficos e dashboards.
    Usa o banco para obter métricas e o documento de regras para entender contexto ou definições.
    """
    llm = get_chat_llm()

    db_tool = Tool(
        name="consultar_banco",
        func=ask_database,
        description="Busque valores ou agregações reais no Postgres antes de propor um gráfico.",
    )
    rules_tool = Tool(
        name="consultar_regras",
        func=ask_rules_document,
        description="Use para entender definições ou regras de negócio que impactam o gráfico.",
    )

    system_prompt = (
        "Você é um consultor de visualização de dados. Proponha gráficos e dashboards acionáveis. "
        "Sempre que precisar de números reais, chame 'consultar_banco' para gerar a consulta. "
        "Use 'consultar_regras' quando precisar de definições ou contexto de negócio. "
        "Sempre escreva consultas de leitura agregadas (COUNT, SUM, AVG etc.) com GROUP BY, "
        "sem SELECT *, e inclua LIMIT 200 ou menos para evitar carregar grandes volumes. "
        "Inclua sempre a consulta principal em um bloco ```sql``` para permitir a renderizacao do grafico. "
        "Nao responda sem esse bloco; se houver duvida, ainda assim proponha uma consulta tentativa. "
        "Responda em passos curtos: objetivo, consultas SQL sugeridas (apenas leitura), "
        "campos para eixo/legenda, tipo de gráfico recomendado e anotações de uso."
    )

    return create_agent(
        model=llm,
        tools=[db_tool, rules_tool],
        system_prompt=system_prompt,
    )


def generate_charts(question: str, history: List[Tuple[str, str]] | None = None) -> str:
    print("[agent] gerador de gráficos recebendo pergunta:", question)
    agent = build_chart_agent()
    msgs = history[:] if history else []
    msgs.append(("user", question))
    result = agent.invoke({"messages": msgs})
    if isinstance(result, dict) and "messages" in result:
        messages = result.get("messages") or []
        if messages:
            final_msg = messages[-1]
            try:
                content = final_msg.content  # type: ignore[assignment]
            except Exception:
                content = str(final_msg)
            if content:
                return content
    return "Não foi possível sugerir gráficos com as fontes locais."


def _extract_series(data: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[Any], list[float]]:
    x_vals: list[Any] = []
    y_vals: list[float] = []
    for idx, row in enumerate(data):
        if x_key not in row or y_key not in row:
            raise ValueError(f"Linha {idx} precisa conter '{x_key}' e '{y_key}'.")
        x_vals.append(row[x_key])
        try:
            y_vals.append(float(row[y_key]))
        except Exception:
            raise ValueError(f"Valor inválido para '{y_key}' na linha {idx}.")
    return x_vals, y_vals


def render_chart(chart_spec_json: str) -> str:
    """
    Renderiza um gráfico com matplotlib a partir de uma especificação JSON.
    Campos esperados: chart_type, data, x, y, title (opcional), group_by (opcional), output_format (opcional).
    Retorna a imagem em base64 e não salva arquivos em disco.
    """
    try:
        spec = json.loads(chart_spec_json)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise ValueError("Especificação inválida: envie um JSON válido.") from exc

    chart_type = (spec.get("chart_type") or "").lower()
    if chart_type not in {"line", "bar", "pie", "area", "scatter"}:
        raise ValueError("Tipo de gráfico inválido. Use line, bar, pie, area ou scatter.")

    data = spec.get("data") or []
    if not isinstance(data, list) or not data:
        raise ValueError("data deve ser uma lista não vazia de objetos.")

    x_key = spec.get("x")
    y_key = spec.get("y")
    if not x_key or not y_key:
        raise ValueError("Campos 'x' e 'y' são obrigatórios.")

    title = spec.get("title") or "Gráfico"
    group_by = spec.get("group_by")
    output_format = (spec.get("output_format") or "png").lower()
    if output_format not in {"png", "svg"}:
        raise ValueError("Formato inválido. Use png ou svg.")

    fig, ax = plt.subplots(figsize=(8, 5))
    if group_by:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in data:
            group_val = str(row.get(group_by, "grupo"))
            grouped.setdefault(group_val, []).append(row)
        for label, rows in grouped.items():
            x_vals, y_vals = _extract_series(rows, x_key, y_key)
            if chart_type == "bar":
                ax.bar(x_vals, y_vals, label=label, alpha=0.85)
            elif chart_type == "line":
                ax.plot(x_vals, y_vals, marker="o", label=label)
            elif chart_type == "area":
                ax.fill_between(x_vals, y_vals, alpha=0.35, label=label)
                ax.plot(x_vals, y_vals, linewidth=1, color="black", alpha=0.5)
            elif chart_type == "scatter":
                ax.scatter(x_vals, y_vals, label=label)
            elif chart_type == "pie":
                labels = [f"{label} | {val}" for val in x_vals]
                ax.pie(y_vals, labels=labels, autopct="%1.1f%%")
    else:
        x_vals, y_vals = _extract_series(data, x_key, y_key)
        if chart_type == "bar":
            ax.bar(x_vals, y_vals, alpha=0.85)
        elif chart_type == "line":
            ax.plot(x_vals, y_vals, marker="o")
        elif chart_type == "area":
            ax.fill_between(x_vals, y_vals, alpha=0.35)
            ax.plot(x_vals, y_vals, linewidth=1, color="black", alpha=0.5)
        elif chart_type == "scatter":
            ax.scatter(x_vals, y_vals)
        elif chart_type == "pie":
            ax.pie(y_vals, labels=[str(v) for v in x_vals], autopct="%1.1f%%")

    ax.set_title(title)
    if chart_type != "pie":
        ax.set_xlabel(str(x_key))
        ax.set_ylabel(str(y_key))
        ax.grid(True, linestyle="--", alpha=0.2)
        if group_by:
            ax.legend(title=str(group_by))

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format=output_format, bbox_inches="tight")
    plt.close(fig)

    return json.dumps(
        {
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "chart_type": chart_type,
            "rows_used": len(data),
            "output_format": output_format,
        },
        ensure_ascii=False,
    )


def render_pdf_report(report_spec_json: str) -> str:
    """
    Gera um relatorio PDF a partir de dados prontos.
    Campos esperados: title (opcional), columns (opcional), rows (lista de objetos ou listas).
    Retorna a imagem PDF em base64 e nao salva arquivos em disco.
    """
    try:
        spec = json.loads(report_spec_json)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise ValueError("Especificacao invalida: envie um JSON valido.") from exc

    title = spec.get("title") or "Relatorio"
    columns = spec.get("columns") or []
    rows = spec.get("rows") or []
    if not isinstance(rows, list):
        raise ValueError("rows deve ser uma lista.")

    if rows and isinstance(rows[0], dict) and not columns:
        columns = list(rows[0].keys())
    if not columns:
        columns = ["dados"]

    max_rows = 200
    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]

    table_rows: list[list[str]] = []
    if rows and isinstance(rows[0], dict):
        for row in rows:
            table_rows.append([str(row.get(col, "")) for col in columns])
    elif rows:
        for row in rows:
            if isinstance(row, (list, tuple)):
                table_rows.append([str(val) for val in row])
            else:
                table_rows.append([str(row)])
    else:
        table_rows = []

    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=14, fontweight="bold")

    if table_rows:
        table = ax.table(
            cellText=table_rows,
            colLabels=columns,
            loc="upper center",
            bbox=[0.05, 0.08, 0.9, 0.84],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
    else:
        ax.text(0.5, 0.5, "Sem dados para o relatorio.", ha="center", va="center")

    if truncated:
        ax.text(0.5, 0.04, f"Dados truncados para {max_rows} linhas.", ha="center", va="bottom", fontsize=9)

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="pdf", bbox_inches="tight")
    plt.close(fig)

    file_name = f"relatorio_{int(time.time())}.pdf"
    return json.dumps(
        {
            "pdf_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "file_name": file_name,
            "title": title,
            "rows_used": len(table_rows),
            "truncated": truncated,
        },
        ensure_ascii=False,
    )


def build_render_agent():
    """
    Agente responsável por renderizar gráficos com matplotlib.
    Espera receber uma especificação JSON com dados já calculados.
    """
    llm = get_chat_llm()

    render_tool = Tool(
        name="renderizar_grafico",
        func=render_chart,
        description=(
            "Renderiza gráficos com matplotlib. Envie um JSON com: "
            "chart_type (line|bar|pie|area|scatter), data (lista de objetos), x, y, "
            "title (opcional), group_by (opcional), output_format (png|svg). "
            "Retorna a imagem em base64 (nao salva em disco)."
        ),
    )

    system_prompt = (
        "Você renderiza gráficos com base em dados prontos. Não invente dados. "
        "Converta a solicitação do usuário em uma chamada para renderizar_grafico, "
        "enviando o JSON completo com chart_type, data, x, y e opcionais."
    )

    return create_agent(
        model=llm,
        tools=[render_tool],
        system_prompt=system_prompt,
    )


def build_report_agent():
    """
    Agente responsavel por gerar relatorios em PDF com dados do banco.
    """
    llm = get_chat_llm()

    db_tool = Tool(
        name="consultar_banco",
        func=ask_database,
        description="Use para obter dados do Postgres antes de montar o relatorio.",
    )
    rules_tool = Tool(
        name="consultar_regras",
        func=ask_rules_document,
        description="Use para entender regras ou contexto necessario ao relatorio.",
    )
    render_tool = Tool(
        name="renderizar_relatorio_pdf",
        func=render_pdf_report,
        description=(
            "Gera relatorio em PDF com base64. Envie JSON com: "
            "title (opcional), columns (opcional), rows (lista de objetos ou listas)."
        ),
    )

    system_prompt = (
        "Voce gera relatorios PDF com dados reais. Sempre que precisar de dados, "
        "use 'consultar_banco'. Ao final, chame 'renderizar_relatorio_pdf' com um JSON "
        "contendo title, columns e rows. Limite o volume a no maximo 200 linhas. "
        "Responda somente com o JSON retornado pela ferramenta, de preferencia em bloco ```json```."
    )

    return create_agent(
        model=llm,
        tools=[db_tool, rules_tool, render_tool],
        system_prompt=system_prompt,
    )


def generate_pdf_report(question: str, history: List[Tuple[str, str]] | None = None) -> str:
    print("[agent] gerador de relatorios PDF recebendo pergunta:", question)
    agent = build_report_agent()
    msgs = history[:] if history else []
    msgs.append(("user", question))
    result = agent.invoke({"messages": msgs})
    if isinstance(result, dict) and "messages" in result:
        messages = result.get("messages") or []
        if messages:
            final_msg = messages[-1]
            try:
                content = final_msg.content  # type: ignore[assignment]
            except Exception:
                content = str(final_msg)
            if content:
                return content
    return "Nao foi possivel gerar o relatorio PDF."


def render_chart_request(question: str, history: List[Tuple[str, str]] | None = None) -> str:
    print("[agent] renderização de gráfico recebendo pergunta:", question)
    agent = build_render_agent()
    msgs = history[:] if history else []
    msgs.append(("user", question))
    result = agent.invoke({"messages": msgs})
    if isinstance(result, dict) and "messages" in result:
        messages = result.get("messages") or []
        if messages:
            final_msg = messages[-1]
            try:
                content = final_msg.content  # type: ignore[assignment]
            except Exception:
                content = str(final_msg)
            if content:
                return content
    return "Não foi possível renderizar o gráfico."


def build_manager_agent():
    """Agente que decide automaticamente entre banco e documento."""
    llm = get_chat_llm()

    db_tool = Tool(
        name="consultar_banco",
        func=ask_database,
        description="Use para perguntas que dependem de dados ou números do banco de dados Postgres.",
    )

    rules_tool = Tool(
        name="consultar_regras",
        func=ask_rules_document,
        description="Use para perguntas sobre regras, procedimentos ou conteúdo textual do documento de regras.",
    )

    code_tool = Tool(
        name="gerar_codigo",
        func=generate_code,
        description="Use para gerar código SQL ou Python com base nas regras e no esquema do banco.",
    )

    chart_tool = Tool(
        name="gerar_graficos",
        func=generate_charts,
        description="Use para propor gráficos e dashboards; consulte o banco para métricas reais.",
    )
    render_tool = Tool(
        name="renderizar_grafico",
        func=render_chart_request,
        description="Use quando o usuário fornecer dados e pedir a renderização do gráfico.",
    )
    report_tool = Tool(
        name="gerar_relatorio_pdf",
        func=generate_pdf_report,
        description="Use para gerar relatorios em PDF com dados do Postgres.",
    )

    system_prompt = (
        "Você é um agente autônomo que decide a melhor fonte de dados.\n"
        "- Use 'consultar_banco' para perguntas sobre registros, contagens ou valores no Postgres.\n"
        "- Use 'consultar_regras' para perguntas sobre regras, procedimentos ou texto do documento.\n"
        "- Use 'gerar_codigo' quando o usuário pedir código SQL ou Python.\n"
        "- Use 'gerar_graficos' quando o usuário pedir gráficos, dashboards ou visualizações.\n"
        "- Para pedidos de grafico ou visualizacao, nunca responda direto: sempre use 'gerar_graficos'.\n"
        "- Use 'gerar_relatorio_pdf' quando o usuario pedir relatorio ou formulario em PDF.\n"
        "- Use 'renderizar_grafico' quando receber dados prontos e precisar gerar a visualização.\n"
        "Responda apenas com base nas ferramentas locais; não use conhecimento externo."
    )

    return create_agent(
        model=llm,
        tools=[db_tool, rules_tool, code_tool, chart_tool, render_tool, report_tool],
        system_prompt=system_prompt,
    )


def answer_question(question: str, history: list[tuple[str, str]] | None = None) -> str:
    print("[agent] orquestrador recebendo pergunta:", question)
    agent = build_manager_agent()
    msgs = history[:] if history else []
    msgs.append(("user", question))
    result = agent.invoke({"messages": msgs})
    if isinstance(result, dict) and "messages" in result:
        messages = result.get("messages") or []
        if messages:
            final_msg = messages[-1]
            try:
                content = final_msg.content  # type: ignore[assignment]
            except Exception:
                content = str(final_msg)
            if content:
                return content
    # fallback se nada retornou
    return "Não foi possível responder com as fontes locais."
