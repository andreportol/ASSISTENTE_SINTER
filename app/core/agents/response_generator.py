from __future__ import annotations

from datetime import datetime
from decimal import Decimal

MONTH_NAMES = {
    1: "janeiro",
    2: "fevereiro",
    3: "marco",
    4: "abril",
    5: "maio",
    6: "junho",
    7: "julho",
    8: "agosto",
    9: "setembro",
    10: "outubro",
    11: "novembro",
    12: "dezembro",
}


def _format_currency(value: Decimal | float | int) -> str:
    dec = value if isinstance(value, Decimal) else Decimal(str(value))
    dec = dec.quantize(Decimal("0.01"))
    formatted = f"{dec:,.2f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")

def _format_quantity(value: object) -> str:
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, (float, Decimal)):
        num = float(value)
        if num.is_integer():
            return str(int(num))
        formatted = f"{num:,.2f}"
        return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return str(value)

def _build_period_label(date_start: str | None, periodo: str | None) -> str | None:
    if not date_start or not periodo:
        return None
    try:
        parsed = datetime.strptime(date_start, "%Y-%m-%d").date()
    except ValueError:
        return None
    if periodo == "mes":
        month = MONTH_NAMES.get(parsed.month)
        if not month:
            return None
        return f"no mes de {month} de {parsed.year}"
    if periodo == "ano":
        return f"no ano de {parsed.year}"
    return None


def generate(result: dict, semantic: dict) -> str:
    if result.get("error"):
        return result["error"]

    intent = semantic.get("intent")
    objeto = semantic.get("objeto")
    tipo_movimento = semantic.get("tipo_movimento")
    data = semantic.get("data")

    if intent == "list":
        items = result.get("items") or []
        if not items:
            if tipo_movimento:
                return f"Nenhum produto encontrado para {tipo_movimento} no dia {data}."
            return "Nenhum produto encontrado."

        nomes = ", ".join(str(item) for item in items)
        if tipo_movimento and data:
            return f"Produtos com {tipo_movimento} no dia {data}: {nomes}."
        return f"Produtos: {nomes}."

    if intent == "count":
        total = result.get("value", 0)
        if objeto == "venda":
            return f"Foram realizadas {total} vendas no dia {data}."
        if tipo_movimento:
            label = "entradas" if tipo_movimento == "entrada" else "saidas"
            return f"Foram registradas {total} {label} no dia {data}."
        return f"Total encontrado: {total}."

    if intent == "sum":
        metric = semantic.get("metric")
        if metric == "quantidade":
            total = _format_quantity(result.get("value", 0))
            if data:
                return f"Foram vendidos {total} produtos no dia {data}."
            return f"Foram vendidos {total} produtos."

        valor = _format_currency(result.get("value", Decimal("0")))
        if data:
            return f"O total de vendas no dia {data} foi de R$ {valor}."
        return f"O total de vendas foi de R$ {valor}."

    if intent == "top":
        items = result.get("items") or []
        metric = semantic.get("metric") or "quantidade"
        rank_limit = semantic.get("rank_limit") or len(items)
        rank_position = semantic.get("rank_position")
        period_label = _build_period_label(semantic.get("date_start"), semantic.get("periodo"))
        if period_label:
            date_label = period_label
        elif data:
            date_label = f"no dia {data}"
        else:
            date_label = "no total"

        if not items:
            return "Nenhum produto encontrado para o ranking."

        if rank_position:
            pos = int(rank_position)
            item = items[pos - 1] if len(items) >= pos else None
            if not item:
                return "Ranking insuficiente para a posição solicitada."
            if metric == "valor":
                total_fmt = _format_currency(item.get("total", 0))
                return (
                    f"O {pos}º produto com maior faturamento {date_label} foi "
                    f"{item.get('produto')}, com R$ {total_fmt}."
                )
            total_fmt = _format_quantity(item.get("total", 0))
            return (
                f"O {pos}º produto mais vendido {date_label} foi {item.get('produto')}, "
                f"com {total_fmt} unidades."
            )

        if metric == "valor":
            if rank_limit == 1:
                top = items[0]
                total_fmt = _format_currency(top.get("total", 0))
                return (
                    f"O produto com maior faturamento {date_label} foi "
                    f"{top.get('produto')}, com R$ {total_fmt}."
                )

            header = f"Os {rank_limit} produtos com maior faturamento {date_label} foram:"
            lines = []
            for idx, item in enumerate(items, start=1):
                total_fmt = _format_currency(item.get("total", 0))
                lines.append(f"{idx}. {item.get('produto')} — R$ {total_fmt}")
            return header + "\n" + "\n".join(lines)

        if rank_limit == 1:
            top = items[0]
            total_fmt = _format_quantity(top.get("total", 0))
            return (
                f"O produto mais vendido {date_label} foi {top.get('produto')}, "
                f"com {total_fmt} unidades."
            )

        header = f"Os {rank_limit} produtos mais vendidos {date_label} foram:"
        lines = []
        for idx, item in enumerate(items, start=1):
            total_fmt = _format_quantity(item.get("total", 0))
            lines.append(f"{idx}. {item.get('produto')} — {total_fmt} unidades")
        return header + "\n" + "\n".join(lines)

    return "Consulta nao suportada."
