from __future__ import annotations

from .conversation_state import ConversationState


def plan(semantic: dict, state: ConversationState) -> dict:
    merged = state.to_dict()
    for key in (
        "intent",
        "objeto",
        "tipo_movimento",
        "data",
        "metric",
        "rank_limit",
        "rank_position",
        "date_start",
        "date_end",
        "periodo",
    ):
        value = semantic.get(key)
        if value is not None:
            merged[key] = value
    return {
        "intent": merged.get("intent"),
        "objeto": merged.get("objeto"),
        "tipo_movimento": merged.get("tipo_movimento"),
        "data": merged.get("data"),
        "metric": merged.get("metric"),
        "rank_limit": merged.get("rank_limit"),
        "rank_position": merged.get("rank_position"),
        "date_start": merged.get("date_start"),
        "date_end": merged.get("date_end"),
        "periodo": merged.get("periodo"),
    }
