from __future__ import annotations

import csv

from django.http import HttpResponse
from django.shortcuts import render
from django.utils.text import get_valid_filename
from django.views.decorators.csrf import ensure_csrf_cookie

from ..services.chart_builder import build_chart_data_from_rows
from ..services.sql_introspection import (
    MAX_FILTER_VALUES,
    build_column_display_map,
    get_authorized_tables,
    get_distinct_values,
    get_distinct_values_with_flag,
    get_display_column,
    get_foreign_key_map,
    get_numeric_columns,
    get_table_columns,
    is_valid_identifier,
)
from ..services.table_builder import (
    agg_label_pt,
    build_agg_expression,
    format_pt_br,
    resolve_column_input,
)


TABLE_PAGE_SIZE = 20


@ensure_csrf_cookie
def charts(request):
    error_message = None
    chart_data = None
    chart_error = None

    tables = get_authorized_tables()
    raw_view_mode_select = request.POST.get("view_mode_select")
    view_mode = (
        raw_view_mode_select
        or request.POST.get("view_mode")
        or request.GET.get("view_mode")
        or "chart"
    ).strip().lower()

    table = (request.POST.get("table") or request.GET.get("table") or "").strip()
    if not table and "imoveis" in tables:
        table = "imoveis"

    label_col = (request.POST.get("label_col") or "").strip()
    label_value = (request.POST.get("label_value") or "").strip()
    value_col = (request.POST.get("value_col") or "").strip()
    value_value = (request.POST.get("value_value") or "").strip()

    table_cols = [val.strip() for val in request.POST.getlist("table_cols")]
    table_group_by = (request.POST.get("table_group_by") or "").strip()
    table_group_agg = (request.POST.get("table_group_agg") or "count").strip().lower()
    table_group_value = (request.POST.get("table_group_value") or "").strip()

    filter_cols = [val.strip() for val in request.POST.getlist("filter_col")]
    filter_values = [val.strip() for val in request.POST.getlist("filter_value")]
    filter_ops = [val.strip().lower() for val in request.POST.getlist("filter_op")]

    agg = (request.POST.get("agg") or "count").strip().lower()
    chart_type = (request.POST.get("chart_type") or "bar").strip().lower()

    limit_raw = request.POST.get("limit") or "20"
    page_raw = request.POST.get("page") or request.GET.get("page") or "1"

    # Se o usuário apenas alternar entre gráfico/tabela, limpamos todos os campos
    if request.method == "POST" and raw_view_mode_select is not None:
        table = ""
        label_col = ""
        label_value = ""
        value_col = ""
        value_value = ""
        table_cols = []
        table_group_by = ""
        table_group_value = ""
        table_group_agg = "count"
        filter_cols = []
        filter_values = []
        filter_ops = ["and"]
        agg = "count"
        chart_type = "bar"
        limit_raw = "20"
        page_raw = "1"

    if view_mode not in {"chart", "table"}:
        view_mode = "chart"

    try:
        limit = int(limit_raw)
    except ValueError:
        limit = 20

    try:
        page = int(page_raw)
    except ValueError:
        page = 1
    page = max(1, page)
    limit = max(1, min(limit, 200))

    columns = get_table_columns(table) if table else []
    numeric_columns = get_numeric_columns(table) if table else []

    column_display_map = build_column_display_map(table, columns) if table else {}
    column_display_reverse = {label: col for col, label in column_display_map.items()}
    display_columns = [column_display_map.get(col, col) for col in columns]
    display_numeric_columns = [column_display_map.get(col, col) for col in numeric_columns]

    label_col = resolve_column_input(label_col, column_display_reverse)
    value_col = resolve_column_input(value_col, column_display_reverse)
    table_group_by = resolve_column_input(table_group_by, column_display_reverse)
    table_group_value = resolve_column_input(table_group_value, column_display_reverse)
    table_cols = [resolve_column_input(val, column_display_reverse) for val in table_cols]
    filter_cols = [resolve_column_input(val, column_display_reverse) for val in filter_cols]

    valid_table_cols = [col for col in table_cols if is_valid_identifier(col) and col in columns]
    label_values = get_distinct_values(table, label_col) if table and label_col else []
    value_values = get_distinct_values(table, value_col) if table and value_col else []

    max_filters = max(len(filter_cols), len(filter_values), len(filter_ops), 1)
    filter_rows = []
    for idx in range(max_filters):
        col = filter_cols[idx] if idx < len(filter_cols) else ""
        val = filter_values[idx] if idx < len(filter_values) else ""
        op = filter_ops[idx] if idx < len(filter_ops) else "and"
        if op not in {"and", "or"}:
            op = "and"
        if idx == 0:
            op = "and"
        values, truncated = get_distinct_values_with_flag(table, col) if table and col else ([], False)
        filter_rows.append(
            {
                "col": col,
                "display_col": column_display_map.get(col, col),
                "value": val,
                "values": values,
                "op": op,
                "allow_text": truncated,
            }
        )

    table_headers: list[str] = []
    table_rows: list[list[str]] = []
    total_rows = 0
    total_pages = 1

    action = request.POST.get("action") or ""
    if request.method == "POST" and action in {"generate_chart", "download_csv"}:
        if not is_valid_identifier(table) or table not in tables:
            error_message = "Selecione uma tabela válida."
        elif view_mode == "table":
            if table_group_by:
                if not is_valid_identifier(table_group_by) or table_group_by not in columns:
                    error_message = "Selecione uma coluna válida para agrupar."
                elif table_group_agg not in {"count", "sum", "avg"}:
                    error_message = "Selecione uma agregação válida para o agrupamento."
                elif table_group_agg == "count" and table_group_value:
                    if not is_valid_identifier(table_group_value) or table_group_value not in columns:
                        error_message = "Selecione uma coluna válida para COUNT."
                elif table_group_agg != "count" and (
                    not is_valid_identifier(table_group_value) or table_group_value not in numeric_columns
                ):
                    error_message = "Selecione uma coluna numérica válida para agregação."
            else:
                if not table_cols:
                    error_message = "Selecione ao menos uma coluna para a tabela."
                elif len(valid_table_cols) != len(table_cols):
                    error_message = "Selecione colunas válidas para a tabela."
        else:
            if not is_valid_identifier(label_col):
                error_message = "Selecione uma coluna válida para o eixo X."
            elif agg not in {"count", "sum", "avg"}:
                error_message = "Selecione uma agregação válida."
            elif agg == "count" and value_col and (not is_valid_identifier(value_col) or value_col not in columns):
                error_message = "Selecione uma coluna válida para COUNT."
            elif agg != "count" and not is_valid_identifier(value_col):
                error_message = "Selecione uma coluna numérica válida para agregação."
            elif agg != "count" and numeric_columns and value_col not in numeric_columns:
                error_message = "Selecione uma coluna numérica válida para agregação."

        if not error_message:
            filter_error = None
            for row in filter_rows:
                col = row["col"]
                val = row["value"]
                if not col and not val:
                    continue
                if not col:
                    filter_error = "Selecione uma coluna válida para o filtro."
                    break
                if not is_valid_identifier(col) or col not in columns:
                    filter_error = "Selecione uma coluna válida para o filtro."
                    break
            if filter_error:
                error_message = filter_error

        if error_message:
            return render(
                request,
                "core/charts.html",
                {
                    "tables": tables,
                    "columns": columns,
                    "selected_table": table,
                    "selected_label": column_display_map.get(label_col, label_col),
                    "selected_label_value": label_value,
                    "selected_value": column_display_map.get(value_col, value_col),
                    "selected_value_value": value_value,
                    "selected_agg": agg,
                    "selected_chart_type": chart_type,
                    "selected_limit": limit,
                    "view_mode": view_mode,
                    "page": page,
                    "total_pages": total_pages,
                    "total_rows": total_rows,
                    "label_values": label_values,
                    "value_values": value_values,
                    "filter_rows": filter_rows,
                    "numeric_columns": numeric_columns,
                    "selected_table_cols": valid_table_cols,
                    "selected_table_group_by": column_display_map.get(table_group_by, table_group_by),
                    "selected_table_group_agg": table_group_agg,
                    "selected_table_group_value": column_display_map.get(table_group_value, table_group_value),
                    "chart_data": chart_data,
                    "table_headers": table_headers,
                    "table_rows": table_rows,
                    "chart_error": chart_error,
                    "error_message": error_message,
                    "column_display_map": column_display_map,
                    "display_columns": display_columns,
                    "display_numeric_columns": display_numeric_columns,
                },
            )

        try:
            params: list[str] = []
            where_clauses = []
            filter_expr = ""
            table_prefix = f"{table}." if is_valid_identifier(table) else ""

            for row in filter_rows:
                col = row["col"]
                val = row["value"]
                if not col or not val:
                    continue
                clause = f"{table_prefix}{col} = %s"
                if not filter_expr:
                    filter_expr = clause
                else:
                    op = row.get("op", "and").upper()
                    if op not in {"AND", "OR"}:
                        op = "AND"
                    filter_expr = f"{filter_expr} {op} {clause}"
                params.append(val)

            if filter_expr:
                where_clauses.append(f"({filter_expr})")

            from django.db import connections

            with connections["default"].cursor() as cursor:
                if view_mode == "table":
                    if table_group_by:
                        select_expr = build_agg_expression(table_group_agg, table_group_value)
                        base_sql = f"SELECT {table_group_by} AS label, {select_expr} AS value FROM {table}"
                        if where_clauses:
                            base_sql += " WHERE " + " AND ".join(where_clauses)
                        base_sql += f" GROUP BY {table_group_by}"
                        order_sql = f"{base_sql} ORDER BY value DESC"
                        table_headers = [
                            table_group_by,
                            f"{agg_label_pt(table_group_agg)}({table_group_value})"
                            if table_group_value
                            else f"{agg_label_pt(table_group_agg)}(*)",
                        ]

                        if action == "download_csv":
                            cursor.execute(order_sql, params)
                            rows = cursor.fetchall()
                            response = HttpResponse(content_type="text/csv")
                            filename = get_valid_filename(f"{table}_{table_group_by}_{table_group_agg}.csv") or "dados.csv"
                            response["Content-Disposition"] = f'attachment; filename="{filename}"'
                            writer = csv.writer(response)
                            writer.writerow(table_headers)
                            for row in rows:
                                writer.writerow([row[0], row[1]])
                            return response

                        count_sql = f"SELECT COUNT(*) FROM ({base_sql}) AS subquery"
                        cursor.execute(count_sql, params)
                        total_rows = int(cursor.fetchone()[0] or 0)
                        total_pages = max(1, (total_rows + TABLE_PAGE_SIZE - 1) // TABLE_PAGE_SIZE)
                        page = min(page, total_pages)
                        offset = (page - 1) * TABLE_PAGE_SIZE
                        cursor.execute(f"{order_sql} LIMIT %s OFFSET %s", params + [TABLE_PAGE_SIZE, offset])
                        rows = cursor.fetchall()
                        table_rows = [[format_pt_br(row[0]), format_pt_br(row[1])] for row in rows]

                    else:
                        selected_cols = valid_table_cols
                        fk_map = get_foreign_key_map(table)
                        select_exprs: list[str] = []
                        table_headers = []
                        join_clauses: list[str] = []

                        for col in selected_cols:
                            ref = fk_map.get(col)
                            if ref:
                                ref_table, ref_col = ref
                                display_col = get_display_column(ref_table)
                                if display_col:
                                    alias = f"{ref_table}_{col}"
                                    join_clauses.append(
                                        f"LEFT JOIN {ref_table} {alias} ON {table}.{col} = {alias}.{ref_col}"
                                    )
                                    select_exprs.append(f"{alias}.{display_col} AS {col}_nome")
                                    if col.endswith("_id"):
                                        table_headers.append(col[:-3])
                                    else:
                                        table_headers.append(col)
                                    continue

                            select_exprs.append(f"{table}.{col}")
                            table_headers.append(col)

                        base_sql = f"SELECT {', '.join(select_exprs)} FROM {table}"
                        if join_clauses:
                            base_sql += " " + " ".join(join_clauses)
                        if where_clauses:
                            base_sql += " WHERE " + " AND ".join(where_clauses)
                        order_sql = f"{base_sql} ORDER BY 1"

                        if action == "download_csv":
                            cursor.execute(order_sql, params)
                            rows = cursor.fetchall()
                            response = HttpResponse(content_type="text/csv")
                            filename = get_valid_filename(f"{table}_tabela.csv") or "dados.csv"
                            response["Content-Disposition"] = f'attachment; filename="{filename}"'
                            writer = csv.writer(response)
                            writer.writerow(table_headers)
                            for row in rows:
                                writer.writerow(list(row))
                            return response

                        count_sql = f"SELECT COUNT(*) FROM ({base_sql}) AS subquery"
                        cursor.execute(count_sql, params)
                        total_rows = int(cursor.fetchone()[0] or 0)
                        total_pages = max(1, (total_rows + TABLE_PAGE_SIZE - 1) // TABLE_PAGE_SIZE)
                        page = min(page, total_pages)
                        offset = (page - 1) * TABLE_PAGE_SIZE
                        cursor.execute(f"{order_sql} LIMIT %s OFFSET %s", params + [TABLE_PAGE_SIZE, offset])
                        rows = cursor.fetchall()
                        table_rows = [[format_pt_br(cell) for cell in row] for row in rows]

                else:
                    select_expr = build_agg_expression(agg, value_col)
                    base_sql = f"SELECT {label_col} AS label, {select_expr} AS value FROM {table}"

                    if label_value:
                        where_clauses.append(f"{label_col} = %s")
                        params.append(label_value)
                    if value_value and is_valid_identifier(value_col):
                        where_clauses.append(f"{value_col} = %s")
                        params.append(value_value)

                    if where_clauses:
                        base_sql += " WHERE " + " AND ".join(where_clauses)
                    base_sql += f" GROUP BY {label_col}"
                    order_sql = f"{base_sql} ORDER BY value DESC"

                    cursor.execute(f"{order_sql} LIMIT %s", params + [limit])
                    rows = cursor.fetchall()
                    cols = [col[0] for col in (cursor.description or [])]
                    chart_data = build_chart_data_from_rows(rows, cols, chart_type)
                    if chart_data:
                        chart_data["title"] = f"{agg_label_pt(agg)} por {label_col}"

                    table_rows = [[format_pt_br(row[0]), format_pt_br(row[1])] for row in rows]
                    total_rows = len(table_rows)
                    total_pages = 1
                    value_header = f"{agg_label_pt(agg)}({value_col})" if value_col else f"{agg_label_pt(agg)}(*)"
                    table_headers = [label_col, value_header]

                    if action == "download_csv":
                        response = HttpResponse(content_type="text/csv")
                        filename = get_valid_filename(f"{table}_{label_col}_{agg}.csv") or "dados.csv"
                        response["Content-Disposition"] = f'attachment; filename="{filename}"'
                        writer = csv.writer(response)
                        writer.writerow(table_headers)
                        for row in rows:
                            writer.writerow([row[0], row[1]])
                        return response

        except Exception as exc:
            chart_error = f"Erro ao gerar gráfico: {exc}"

    return render(
        request,
        "core/charts.html",
        {
            "tables": tables,
            "columns": columns,
            "selected_table": table,
            "selected_label": column_display_map.get(label_col, label_col),
            "selected_label_value": label_value,
            "selected_value": column_display_map.get(value_col, value_col),
            "selected_value_value": value_value,
            "selected_table_cols": valid_table_cols,
            "selected_agg": agg,
            "selected_chart_type": chart_type,
            "selected_limit": limit,
            "view_mode": view_mode,
            "page": page,
            "total_pages": total_pages,
            "total_rows": total_rows,
            "label_values": label_values,
            "value_values": value_values,
            "filter_rows": filter_rows,
            "numeric_columns": numeric_columns,
            "chart_data": chart_data,
            "table_headers": table_headers,
            "table_rows": table_rows,
            "selected_table_group_by": column_display_map.get(table_group_by, table_group_by),
            "selected_table_group_agg": table_group_agg,
            "selected_table_group_value": column_display_map.get(table_group_value, table_group_value),
            "chart_error": chart_error,
            "error_message": error_message,
            "column_display_map": column_display_map,
            "display_columns": display_columns,
            "display_numeric_columns": display_numeric_columns,
        },
    )
