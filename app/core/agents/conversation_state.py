from __future__ import annotations

from typing import Any, Dict

# Campos que representam "intenção semântica"
# Eles são resetados quando uma nova pergunta válida chega
RESET_FIELDS = {
    "intent",
    "objeto",
    "tipo_movimento",
    "data",
    "metric",
    "rank_limit",
    "date_anchor",
    "rank_position",
    "date_start",
    "date_end",
    "periodo",
    "needs_confirmation",
    "confirmation",
    "confirmation_response",
}


class ConversationState:
    """
    Mantém estado semântico mínimo entre perguntas consecutivas.

    Regras fundamentais:
    - Perguntas inválidas (unsupported) limpam o estado completamente.
    - Nova pergunta válida reseta apenas o núcleo semântico (RESET_FIELDS).
    - Campos ausentes na nova pergunta podem herdar valores anteriores,
      desde que não tenham sido explicitamente resetados.
    """

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}

    def update(self, new_state: dict) -> dict:
        """
        Atualiza o estado da conversa com base na nova interpretação semântica.
        """

        # 🚫 BLOQUEIO DURO
        # Pergunta inválida ou subjetiva → estado não pode contaminar próximas perguntas
        if new_state.get("unsupported"):
            self.state = {
                "raw_question": new_state.get("raw_question"),
                "unsupported": True,
                "reason": new_state.get("reason"),
            }
            return dict(self.state)

        # 🧹 NOVA PERGUNTA VÁLIDA
        # Se há uma pergunta nova, limpamos o núcleo semântico
        confirmation_response = bool(new_state.get("confirmation_response"))
        keep_date = bool(
            new_state.get("date_anchor") in {"dia", "context"} and self.state.get("data")
        )
        keep_period = bool(
            new_state.get("date_anchor") in {"mes", "ano", "context"}
            and self.state.get("date_start")
            and self.state.get("date_end")
        )
        if new_state.get("raw_question") and not confirmation_response:
            for field in RESET_FIELDS:
                if keep_date and field == "data":
                    continue
                if keep_period and field in {"date_start", "date_end", "periodo"}:
                    continue
                self.state.pop(field, None)
        elif confirmation_response:
            self.state.pop("needs_confirmation", None)
            self.state.pop("confirmation", None)

        # 🔁 MERGE CONTROLADO
        # Apenas valores explícitos (não None) sobrescrevem o estado
        for key, value in new_state.items():
            if value is not None:
                self.state[key] = value

        return dict(self.state)

    def to_dict(self) -> dict:
        """
        Retorna uma cópia imutável do estado atual.
        """
        return dict(self.state)
