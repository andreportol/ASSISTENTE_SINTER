import os
from pathlib import Path
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from .langchain_agents import answer_question


@ensure_csrf_cookie
def index(request):
    response_text = None
    error_message = None
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
        else:
            if query:
                try:
                    response_text = answer_question(query, history=history)
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
        },
    )
