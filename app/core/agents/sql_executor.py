from __future__ import annotations

from decimal import Decimal
from typing import Any

from django.db import connections

from ..services.sql_introspection import get_table_columns


# ============================================================
# EXECUTORS
# ============================================================

class CountMovimentacoesPorDiaExecutor:
    """Conta movimentações (entrada ou saída) em um dia."""

    def execute(self, plan: dict) -> dict[str, Any]:
        data = plan.get("data")
        tipo_movimento = plan.get("tipo_movimento")

        if not data or tipo_movimento not in {"entrada", "saida"}:
            raise ValueError("Dados insuficientes para contagem de movimentações.")

        sql = """
            SELECT COUNT(*)::int
            FROM movimentacao_estoque
            WHERE tipo_movimento = %s
              AND DATE(data_movimento) = %s
        """.strip()

        with connections["default"].cursor() as cursor:
            cursor.execute(sql, [tipo_movimento, data])
            total = cursor.fetchone()[0] or 0

        return {"value": int(total)}


class CountVendasPorDiaExecutor:
    """Conta vendas (alias de saída)."""

    def execute(self, plan: dict) -> dict[str, Any]:
        data = plan.get("data")
        if not data:
            raise ValueError("Data obrigatória para contagem de vendas.")

        sql = """
            SELECT COUNT(*)::int
            FROM movimentacao_estoque
            WHERE tipo_movimento = 'saida'
              AND DATE(data_movimento) = %s
        """.strip()

        with connections["default"].cursor() as cursor:
            cursor.execute(sql, [data])
            total = cursor.fetchone()[0] or 0

        return {"value": int(total)}


class ListProdutosPorDiaExecutor:
    """Lista produtos movimentados (entrada ou saída) em um dia."""

    def execute(self, plan: dict) -> dict[str, Any]:
        data = plan.get("data")
        tipo_movimento = plan.get("tipo_movimento")

        if not data or tipo_movimento not in {"entrada", "saida"}:
            raise ValueError("Dados insuficientes para listar produtos.")

        cols = get_table_columns("produtos")
        if "nome" not in cols:
            raise ValueError("Coluna 'nome' não encontrada em produtos.")

        sql = """
            SELECT DISTINCT p.nome
            FROM movimentacao_estoque m
            JOIN produtos p ON p.id = m.produto_id
            WHERE m.tipo_movimento = %s
              AND DATE(m.data_movimento) = %s
              AND m.quantidade > 0
            ORDER BY p.nome
        """.strip()

        with connections["default"].cursor() as cursor:
            cursor.execute(sql, [tipo_movimento, data])
            rows = cursor.fetchall()

        return {
            "items": [row[0] for row in rows if row and row[0] is not None]
        }


class TotalVendasFinanceiroPorDiaExecutor:
    """Soma financeira das vendas em um dia."""

    def execute(self, plan: dict) -> dict[str, Any]:
        data = plan.get("data")
        if not data:
            raise ValueError("Data obrigatória para total financeiro de vendas.")

        cols = get_table_columns("produtos")
        if "preco_unitario" in cols:
            price_col = "preco_unitario"
        elif "preco_venda" in cols:
            price_col = "preco_venda"
        else:
            raise ValueError("Coluna de preço não encontrada em produtos.")

        sql = f"""
            SELECT COALESCE(SUM(m.quantidade * p.{price_col}), 0)
            FROM movimentacao_estoque m
            JOIN produtos p ON p.id = m.produto_id
            WHERE m.tipo_movimento = 'saida'
              AND DATE(m.data_movimento) = %s
        """.strip()

        with connections["default"].cursor() as cursor:
            cursor.execute(sql, [data])
            total = cursor.fetchone()[0]

        return {"value": Decimal(total)}


class SumProdutosVendidosPorDiaExecutor:
    """Soma a quantidade de produtos vendidos em um dia."""

    def execute(self, plan: dict) -> dict[str, Any]:
        data = plan.get("data")
        if not data:
            raise ValueError("Data obrigatória para soma de produtos vendidos.")

        sql = """
            SELECT COALESCE(SUM(m.quantidade), 0)
            FROM movimentacao_estoque m
            WHERE m.tipo_movimento = 'saida'
              AND DATE(m.data_movimento) = %s
        """.strip()

        with connections["default"].cursor() as cursor:
            cursor.execute(sql, [data])
            total = cursor.fetchone()[0] or 0

        return {"value": Decimal(total)}


class TopProdutosPorDiaExecutor:
    """Ranking de produtos por quantidade ou valor em um dia."""

    def execute(self, plan: dict) -> dict[str, Any]:
        data = plan.get("data")
        metric = plan.get("metric") or "quantidade"
        rank_limit = plan.get("rank_limit") or 1
        rank_position = plan.get("rank_position")

        try:
            limit = int(rank_limit)
        except Exception:
            raise ValueError("Limite inválido para ranking.")

        if limit <= 0:
            raise ValueError("Limite inválido para ranking.")

        if metric not in {"quantidade", "valor"}:
            raise ValueError("Métrica inválida para ranking.")

        if rank_position is not None:
            try:
                pos = int(rank_position)
            except Exception:
                raise ValueError("Posição inválida para ranking.")
            if pos <= 0:
                raise ValueError("Posição inválida para ranking.")
            if limit < pos:
                limit = pos

        if metric == "valor":
            cols = get_table_columns("produtos")
            if "preco_unitario" in cols:
                price_col = "preco_unitario"
            elif "preco_venda" in cols:
                price_col = "preco_venda"
            else:
                raise ValueError("Coluna de preço não encontrada em produtos.")

            select_expr = f"SUM(m.quantidade * p.{price_col})"
        else:
            select_expr = "SUM(m.quantidade)"

        where_clauses = ["m.tipo_movimento = 'saida'"]
        params: list[Any] = []
        if data:
            where_clauses.append("DATE(m.data_movimento) = %s")
            params.append(data)

        sql = f"""
            SELECT p.nome, {select_expr} AS total
            FROM movimentacao_estoque m
            JOIN produtos p ON p.id = m.produto_id
            WHERE {" AND ".join(where_clauses)}
            GROUP BY p.nome
            ORDER BY total DESC
            LIMIT %s
        """.strip()

        with connections["default"].cursor() as cursor:
            cursor.execute(sql, params + [limit])
            rows = cursor.fetchall()

        items = []
        for row in rows:
            if not row:
                continue
            items.append({"produto": row[0], "total": row[1]})

        return {
            "items": items,
            "metric": metric,
            "rank_limit": limit,
            "rank_position": rank_position,
        }


class TopProdutosPorPeriodoExecutor:
    """Ranking de produtos por quantidade ou valor em um período."""

    def execute(self, plan: dict) -> dict[str, Any]:
        start = plan.get("date_start")
        end = plan.get("date_end")
        metric = plan.get("metric") or "quantidade"
        rank_limit = plan.get("rank_limit") or 1
        rank_position = plan.get("rank_position")

        if not start or not end:
            raise ValueError("Período inválido para ranking.")

        try:
            limit = int(rank_limit)
        except Exception:
            raise ValueError("Limite inválido para ranking.")

        if limit <= 0:
            raise ValueError("Limite inválido para ranking.")

        if metric not in {"quantidade", "valor"}:
            raise ValueError("Métrica inválida para ranking.")

        if rank_position is not None:
            try:
                pos = int(rank_position)
            except Exception:
                raise ValueError("Posição inválida para ranking.")
            if pos <= 0:
                raise ValueError("Posição inválida para ranking.")
            if limit < pos:
                limit = pos

        if metric == "valor":
            cols = get_table_columns("produtos")
            if "preco_unitario" in cols:
                price_col = "preco_unitario"
            elif "preco_venda" in cols:
                price_col = "preco_venda"
            else:
                raise ValueError("Coluna de preço não encontrada em produtos.")

            select_expr = f"SUM(m.quantidade * p.{price_col})"
        else:
            select_expr = "SUM(m.quantidade)"

        sql = f"""
            SELECT p.nome, {select_expr} AS total
            FROM movimentacao_estoque m
            JOIN produtos p ON p.id = m.produto_id
            WHERE m.tipo_movimento = 'saida'
              AND DATE(m.data_movimento) BETWEEN %s AND %s
            GROUP BY p.nome
            ORDER BY total DESC
            LIMIT %s
        """.strip()

        with connections["default"].cursor() as cursor:
            cursor.execute(sql, [start, end, limit])
            rows = cursor.fetchall()

        items = []
        for row in rows:
            if not row:
                continue
            items.append({"produto": row[0], "total": row[1]})

        return {
            "items": items,
            "metric": metric,
            "rank_limit": limit,
            "rank_position": rank_position,
        }


# ============================================================
# EXECUTION DISPATCH
# ============================================================

def execute(plan: dict) -> dict[str, Any]:
    executor = plan.get("executor")
    if not executor:
        raise RuntimeError("Executor não definido para a consulta.")

    return executor.execute(plan)
