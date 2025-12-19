"""
Helpers to run LangChain agents against Postgres and a rules document (RAG).
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus

from django.conf import settings

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
        temperature=0.1,
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    splits = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(splits, embedding=embeddings)
    # Persiste para uso nos próximos carregamentos
    vector_store.save_local(str(index_path))
    return vector_store.as_retriever(search_kwargs={"k": 4})


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

    system_prompt = (
        "Você é um agente autônomo que decide a melhor fonte de dados.\n"
        "- Use 'consultar_banco' para perguntas sobre registros, contagens ou valores no Postgres.\n"
        "- Use 'consultar_regras' para perguntas sobre regras, procedimentos ou texto do documento.\n"
        "- Use 'gerar_codigo' quando o usuário pedir código SQL ou Python.\n"
        "Responda apenas com base nas ferramentas locais; não use conhecimento externo."
    )

    return create_agent(
        model=llm,
        tools=[db_tool, rules_tool, code_tool],
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
