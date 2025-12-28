from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

from django.conf import settings

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _get_openai_api_key() -> str:
    api_key = getattr(settings, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Chave OpenAI não configurada.")
    return api_key


def _get_rag_index_path() -> Path:
    # Observação: FAISS.save_local cria uma pasta. Mantemos isso.
    return Path(getattr(settings, "RAG_INDEX_PATH", Path(settings.BASE_DIR) / "documents" / "faiss_index"))


def _get_chunk_settings() -> tuple[int, int]:
    chunk_size = int(getattr(settings, "RAG_CHUNK_SIZE", 700))
    chunk_overlap = int(getattr(settings, "RAG_CHUNK_OVERLAP", 150))
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)
    return chunk_size, chunk_overlap


def _load_documents(paths: List[Path]) -> list:
    docs = []
    for path in paths:
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())
    return docs


def ingest_docs_to_faiss(paths: List[Path]) -> Path:
    if not paths:
        raise ValueError("Nenhum arquivo informado para indexação.")

    embeddings = OpenAIEmbeddings(api_key=_get_openai_api_key())
    index_path = _get_rag_index_path()
    index_path.parent.mkdir(parents=True, exist_ok=True)

    docs = _load_documents(paths)
    chunk_size, chunk_overlap = _get_chunk_settings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    if index_path.exists():
        vs = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
        vs.add_documents(splits)
    else:
        vs = FAISS.from_documents(splits, embedding=embeddings)

    vs.save_local(str(index_path))
    return index_path


def rebuild_faiss_index(docs_dir: Path) -> Path | None:
    paths = [p for p in docs_dir.glob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}]
    index_path = _get_rag_index_path()

    if not paths:
        if index_path.exists():
            shutil.rmtree(index_path, ignore_errors=True)
        return None

    embeddings = OpenAIEmbeddings(api_key=_get_openai_api_key())
    docs = _load_documents(paths)
    chunk_size, chunk_overlap = _get_chunk_settings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    vs = FAISS.from_documents(splits, embedding=embeddings)
    vs.save_local(str(index_path))
    return index_path
