from __future__ import annotations

import re
import unicodedata
from calendar import monthrange
from datetime import date, datetime, timedelta

from django.utils.timezone import now

DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
TOP_RE = re.compile(r"\btop\s+(\d{1,3})\b")
PROD_TOP_RE = re.compile(r"\b(\d{1,3})\s+produt")
ORDINAL_RE = re.compile(r"\b(\d{1,2})\s*(?:º|o|ª|a)\b")
YEAR_RE = re.compile(r"\b(20\d{2})\b")

# Palavras que tornam a pergunta subjetiva / não mensurável
SUBJETIVOS = ("bonito", "feio", "melhor", "pior", "legal", "ruim")
ORDINAIS = {
    "primeiro": 1,
    "primeira": 1,
    "segundo": 2,
    "segunda": 2,
    "terceiro": 3,
    "terceira": 3,
    "quarto": 4,
    "quarta": 4,
    "quinto": 5,
    "quinta": 5,
    "sexto": 6,
    "sexta": 6,
    "setimo": 7,
    "setima": 7,
    "oitavo": 8,
    "oitava": 8,
    "nono": 9,
    "nona": 9,
    "decimo": 10,
    "decima": 10,
}
MONTHS = {
    "janeiro": 1,
    "fevereiro": 2,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _normalize(text: str) -> str:
    cleaned = unicodedata.normalize("NFKD", text or "")
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return cleaned.lower()


def _parse_date(raw: str, normalized: str) -> str | None:
    if "ontem" in normalized:
        return (now().date() - timedelta(days=1)).strftime("%Y-%m-%d")

    if "hoje" in normalized:
        return now().date().strftime("%Y-%m-%d")

    match = DATE_RE.search(raw or "")
    if not match:
        return None
    day, month, year = match.groups()
    try:
        return datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _last_year_from_history(history: list[tuple[str, str]] | None) -> int | None:
    if not history:
        return None
    for role, content in reversed(history):
        if role != "user" or not content:
            continue
        match = YEAR_RE.search(content)
        if match:
            return int(match.group(1))
    return None


def _last_month_from_history(history: list[tuple[str, str]] | None) -> tuple[str, int] | None:
    if not history:
        return None
    for role, content in reversed(history):
        if role != "user" or not content:
            continue
        norm = _normalize(content)
        for month_key, month_num in MONTHS.items():
            if re.search(rf"\b{month_key}\b", norm):
                return month_key, month_num
    return None


def _last_assistant_prompted_year(history: list[tuple[str, str]] | None) -> bool:
    if not history:
        return False
    for role, content in reversed(history):
        if not content:
            continue
        if role == "assistant":
            return "qual ano" in _normalize(content)
        if role == "user":
            return False
    return False


def _is_year_only_response(normalized: str) -> bool:
    if not YEAR_RE.search(normalized):
        return False
    cleaned = YEAR_RE.sub("", normalized)
    cleaned = re.sub(r"\b(ano|em|de|do|da|no|na)\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
    return cleaned == ""


def _period_mentioned(normalized: str) -> bool:
    if re.search(r"\bmes\b", normalized):
        return True
    if re.search(r"\bano\b", normalized):
        return True
    if re.search(r"\bem\s+20\d{2}\b", normalized):
        return True
    for month_key in MONTHS:
        if re.search(rf"\b{month_key}\b", normalized):
            return True
    return False


def _parse_period(
    normalized: str,
    history: list[tuple[str, str]] | None,
    prefer_month_from_history: bool = False,
) -> tuple[str, str, str] | None:
    year_match = YEAR_RE.search(normalized)
    year = int(year_match.group(1)) if year_match else _last_year_from_history(history)
    if not year:
        return None

    explicit_year_period = bool(
        re.search(r"\bano\b", normalized) or re.search(rf"\bem\s+{year}\b", normalized)
    )

    for month_key, month_num in MONTHS.items():
        pattern = rf"(?:mes\s+de\s+)?{month_key}(?:\s+de\s+{year})?"
        if re.search(pattern, normalized):
            start = date(year, month_num, 1)
            end = date(year, month_num, monthrange(year, month_num)[1])
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "mes"

    month_from_history = _last_month_from_history(history)
    if month_from_history and (prefer_month_from_history or not explicit_year_period):
        _, month_num = month_from_history
        start = date(year, month_num, 1)
        end = date(year, month_num, monthrange(year, month_num)[1])
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "mes"

    if explicit_year_period:
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "ano"

    return None


# ------------------------------------------------------------
# Interpreter
# ------------------------------------------------------------
def interpret(question: str, history: list[tuple[str, str]] | None = None) -> dict:
    raw = (question or "").strip()
    norm = _normalize(raw)

    # 1️⃣ BLOQUEIO DURO — subjetivo
    if any(word in norm for word in SUBJETIVOS):
        return {
            "raw_question": raw,
            "unsupported": True,
            "reason": "Pergunta subjetiva não mensurável",
        }

    semantic: dict = {
        "raw_question": raw,
        "intent": None,
        "objeto": None,
        "tipo_movimento": None,
        "data": None,
        "metric": None,
        "rank_limit": None,
        "date_anchor": None,
        "rank_position": None,
        "date_start": None,
        "date_end": None,
        "periodo": None,
        "needs_confirmation": None,
        "confirmation": None,
        "confirmation_response": None,
    }

    # 2️⃣ DATA
    year_only_response = _is_year_only_response(norm)
    prefer_month_from_history = _last_assistant_prompted_year(history) and year_only_response
    period_mentioned = _period_mentioned(norm)
    if not period_mentioned and prefer_month_from_history and YEAR_RE.search(norm):
        period_mentioned = True
    if not period_mentioned:
        semantic["data"] = _parse_date(raw, norm)
    if semantic["data"] is None and any(token in norm for token in ("no dia", "nesse dia", "neste dia")):
        semantic["date_anchor"] = "dia"
    if semantic["data"] is None and "no total" in norm:
        semantic["date_anchor"] = "total"

    if semantic["data"] is None and period_mentioned:
        period = _parse_period(norm, history, prefer_month_from_history)
        if not period:
            semantic["needs_confirmation"] = True
            semantic["confirmation"] = "Qual ano?"
        else:
            semantic["date_start"], semantic["date_end"], semantic["periodo"] = period
            semantic["date_anchor"] = semantic["periodo"]
    if prefer_month_from_history:
        semantic["confirmation_response"] = True

    # 3️⃣ INTENT (ordem importa)
    is_top = False
    if (
        "top" in norm
        or "mais vendido" in norm
        or "mais vendidos" in norm
        or "mais vendida" in norm
        or "mais vendidas" in norm
        or "maior faturamento" in norm
        or "mais lucrativo" in norm
        or "mais lucrativa" in norm
    ):
        is_top = True

    rank_position = None
    for word, value in ORDINAIS.items():
        if word in norm:
            rank_position = value
            break
    if rank_position is None:
        match = ORDINAL_RE.search(raw)
        if match:
            rank_position = int(match.group(1))

    if is_top:
        semantic["intent"] = "top"
        semantic["rank_limit"] = 1
        top_match = TOP_RE.search(norm)
        if top_match:
            semantic["rank_limit"] = int(top_match.group(1))
        else:
            prod_match = PROD_TOP_RE.search(norm)
            if prod_match:
                semantic["rank_limit"] = int(prod_match.group(1))

        if any(token in norm for token in ("faturamento", "lucrativo", "lucro", "valor")):
            semantic["metric"] = "valor"
        else:
            semantic["metric"] = "quantidade"
        if (
            semantic["data"] is None
            and semantic["date_start"] is None
            and semantic["date_anchor"] is None
            and not period_mentioned
        ):
            semantic["date_anchor"] = "context"
        if rank_position is None and any(token in norm for token in ("depois do", "depois da", "apos o", "apos a")):
            rank_position = 2
        if rank_position is not None:
            semantic["rank_position"] = rank_position
            if semantic["rank_limit"] is None or semantic["rank_limit"] < rank_position:
                semantic["rank_limit"] = rank_position

    elif any(token in norm for token in ("quais", "listar", "liste", "nomes")):
        semantic["intent"] = "list"

    elif (
        "total" in norm
        and any(token in norm for token in ("reais", "r$", "valor"))
    ) or any(token in norm for token in ("soma", "somatorio")):
        semantic["intent"] = "sum"

    elif any(
        token in norm
        for token in ("quantas", "quantos", "quantidade", "numero", "contagem", "count")
    ):
        semantic["intent"] = "count"

    # 4️⃣ OBJETO (literal primeiro)
    if "produto" in norm or "produtos" in norm:
        semantic["objeto"] = "produto"

    elif "movimentacao" in norm or "movimentacoes" in norm:
        semantic["objeto"] = "movimentacao"

    elif "venda" in norm or "vendas" in norm or "vendid" in norm:
        semantic["objeto"] = "venda"

    if semantic["intent"] == "top":
        semantic["objeto"] = "produto"
        if semantic["tipo_movimento"] is None:
            semantic["tipo_movimento"] = "saida"

    # 5️⃣ TIPO DE MOVIMENTO — técnico explícito
    if "entrada" in norm or "entradas" in norm:
        semantic["tipo_movimento"] = "entrada"

    elif "saida" in norm or "saidas" in norm:
        semantic["tipo_movimento"] = "saida"

    # 6️⃣ REGRA DE DOMÍNIO FUNDAMENTAL
    # vendido / vendidos / venda ⇒ saída
    if semantic["tipo_movimento"] is None:
        if any(token in norm for token in ("vendido", "vendidos", "venda", "vendas")):
            semantic["tipo_movimento"] = "saida"

    # 7️⃣ REGRA DE DOMÍNIO — quantidade de produtos vendidos
    if (
        semantic["intent"] == "count"
        and semantic["objeto"] == "produto"
        and any(token in norm for token in ("vendido", "vendidos", "venda", "vendas"))
    ):
        semantic["intent"] = "sum"
        semantic["metric"] = "quantidade"
        if semantic["tipo_movimento"] is None:
            semantic["tipo_movimento"] = "saida"

    if semantic["intent"] == "sum" and semantic["metric"] is None:
        if any(token in norm for token in ("reais", "r$", "valor", "faturamento", "lucro")):
            semantic["metric"] = "valor"
        elif "produto" in norm or "produtos" in norm:
            semantic["metric"] = "quantidade"

    # 8️⃣ REGRA DE DOMÍNIO — agregação correta
    # "Quantas saídas de produtos" = contagem de movimentações
    if (
        semantic["intent"] == "count"
        and semantic["objeto"] == "produto"
        and semantic["tipo_movimento"] in {"entrada", "saida"}
    ):
        semantic["objeto"] = "movimentacao"

    return semantic
