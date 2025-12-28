from __future__ import annotations

from decimal import Decimal


def validate(result: dict, semantic: dict) -> dict:
    if result.get("error"):
        return result

    intent = semantic.get("intent")
    if intent == "list":
        items = result.get("items")
        if not isinstance(items, list):
            return {"error": "Resposta incoerente para listagem."}
        return result

    if intent == "count":
        value = result.get("value")
        if isinstance(value, bool) or value is None:
            return {"error": "Resposta incoerente para contagem."}
        try:
            result["value"] = int(value)
        except Exception:
            return {"error": "Resposta incoerente para contagem."}
        return result

    if intent == "sum":
        value = result.get("value")
        if value is None:
            result["value"] = Decimal("0")
            return result
        if not isinstance(value, (int, float, Decimal)):
            return {"error": "Resposta incoerente para soma."}
        return result

    if intent == "top":
        items = result.get("items")
        if not isinstance(items, list):
            return {"error": "Resposta incoerente para ranking."}
        if not items:
            return {"error": "Ranking vazio."}
        rank_limit = semantic.get("rank_limit")
        rank_position = semantic.get("rank_position")
        if rank_limit is not None:
            try:
                limit = int(rank_limit)
                if len(items) > limit:
                    return {"error": "Ranking excede o limite solicitado."}
            except Exception:
                return {"error": "Limite inválido para ranking."}
        if rank_position is not None:
            try:
                pos = int(rank_position)
                if pos <= 0 or len(items) < pos:
                    return {"error": "Ranking insuficiente para a posição solicitada."}
            except Exception:
                return {"error": "Posição inválida para ranking."}
        return result

    return {"error": "Intencao nao suportada."}
