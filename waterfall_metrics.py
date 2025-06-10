
"""
Script #1 — Waterfall metrics for Teradata check‑columns
--------------------------------------------------------

This module exposes a single public function:

    run_waterfall_metrics(
        engine,
        schema_name: str,
        table_name: str,
        identifier_cols: List[str],
        id_col_combinations: List[List[str]],
        logger: logging.Logger
    ) -> List[Dict[str, Any]]

All heavy lifting (deduplication and metrics) is executed inside Teradata
via SQL generated on‑the‑fly; Python only orchestrates.

The function returns, for each identifier‑combo supplied, a dict like:
{
    "identifier_combo": ["customer_id"],
    "total_rows": 123456,
    "metrics": [
        { "check_name": "email_BA_1",
          "check_order": 1,
          "first_zero": 117,
          "only_zero":  84,
          "any_zero":  212,
          "running_first_zero": 117,
          "remaining": 123339 },
        ...
    ]
}

Dependencies: sqlalchemy >= 2.0
"""

from __future__ import annotations

import logging
from textwrap import indent
from typing import List, Sequence, Dict, Any, Tuple

from sqlalchemy import text, inspect
from sqlalchemy.engine import Connection, Engine, Result


def _comma(seq: Sequence[str]) -> str:
    """Return a comma‑separated string."""
    return ", ".join(seq)


def _cast01(col: str) -> str:
    """Cast numeric 0/1 column to CHAR(1) for concatenation."""
    return f"CAST({col} AS CHAR(1))"


def _build_leading_run_expr(check_cols: List[str]) -> str:
    """SQL that yields the length of the leading run of 1s across all checks."""
    concat_expr = "||".join(_cast01(c) for c in check_cols) + "||'0'"
    return f"POSITION('0' IN {concat_expr}) - 1"


def _template_meta(col_name: str) -> Tuple[str, str, int]:
    """Split <channel>_<template>_<n> into components."""
    parts = col_name.split("_")
    increment = int(parts[-1])
    template_id = parts[-2]
    channel = "_".join(parts[:-2])
    return channel, template_id, increment


def run_waterfall_metrics(
    engine: Engine | Connection,
    schema_name: str,
    table_name: str,
    identifier_cols: List[str],
    id_col_combinations: List[List[str]],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Compute all requested waterfall metrics entirely in Teradata."""

    insp = inspect(engine)
    cols_all = insp.get_columns(table_name, schema=schema_name)
    col_order = [c["name"] for c in sorted(cols_all, key=lambda x: x["ordinal_position"])]

    check_cols = [c for c in col_order if c not in identifier_cols]
    if not check_cols:
        raise ValueError("No check columns detected – verify identifier_cols argument.")

    logger.info("Discovered %d check columns", len(check_cols))

    meta = {c: _template_meta(c) for c in check_cols}
    results: List[Dict[str, Any]] = []

    for combo in id_col_combinations:
        logger.info("Processing identifier combo %s", combo)

        # Deduplication CTE
        count_ones_expr = " + ".join(check_cols)
        leading_run_expr = _build_leading_run_expr(check_cols)
        partition_by = _comma(combo)

        dedup_cte = f"""WITH dedup AS (
            SELECT
                { _comma(col_order) },
                ROW_NUMBER() OVER (
                    PARTITION BY {partition_by}
                    ORDER BY {count_ones_expr} DESC,
                             {leading_run_expr} DESC
                ) AS rn
            FROM {schema_name}.{table_name}
        )"""

        # Create secondary index on identifier cols (idempotent)
        try:
            with engine.begin() as conn:
                conn.execute(text(
                    f"CREATE INDEX ({_comma(combo)}) ON {schema_name}.{table_name}"
                ))
        except Exception:
            pass  # likely exists

        metric_selects = []
        total_rows_sql = "(SELECT COUNT(*) FROM dedup WHERE rn = 1)"

        for order_idx, col_j in enumerate(check_cols, start=1):
            channel_j, templ_j, _ = meta[col_j]
            scope_cols = [
                c for c in check_cols
                if meta[c][0] == channel_j and (meta[c][1] == "BA" or meta[c][1] == templ_j)
            ]
            prior_cols = scope_cols[: scope_cols.index(col_j)]
            cond_m1 = f"{col_j}=0" + (" AND " + " AND ".join(f"{c}=1" for c in prior_cols) if prior_cols else "")
            other_in_scope = [c for c in scope_cols if c != col_j]
            cond_m2 = f"{col_j}=0" + (" AND " + " AND ".join(f"{c}=1" for c in other_in_scope) if other_in_scope else "")
            cond_m3 = f"{col_j}=0"

            channel_templates = [meta[c][1] for c in check_cols if meta[c][0] == channel_j and meta[c][1] != "BA"]
            dep_clause = ""
            if templ_j != "BA" and channel_templates.index(templ_j) > 0:
                prev_templs = channel_templates[: channel_templates.index(templ_j)]
                prev_cols = [c for c in check_cols if meta[c][0] == channel_j and meta[c][1] in prev_templs]
                prev_zero_filter = " OR ".join(f"{c}=0" for c in prev_cols)
                dep_clause = f" AND ({prev_zero_filter})"

            metric_selects.append(f"""
            SELECT
                '{col_j}' AS check_name,
                {order_idx} AS check_order,
                SUM(CASE WHEN {cond_m1}{dep_clause} THEN 1 ELSE 0 END) AS first_zero,
                SUM(CASE WHEN {cond_m2}{dep_clause} THEN 1 ELSE 0 END) AS only_zero,
                SUM(CASE WHEN {cond_m3}{dep_clause} THEN 1 ELSE 0 END) AS any_zero
            FROM dedup
            WHERE rn = 1""")

        metrics_union = "\nUNION ALL\n".join(metric_selects)
        final_sql = f"""{dedup_cte},
        metrics_raw AS (
            {metrics_union}
        )
        SELECT
            m.*,
            SUM(first_zero) OVER (ORDER BY check_order) AS running_first_zero,
            ({total_rows_sql}) - SUM(first_zero) OVER (ORDER BY check_order) AS remaining
        FROM metrics_raw m
        ORDER BY check_order"""

        with engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(text(final_sql)).fetchall()]

        results.append({
            "identifier_combo": combo,
            "total_rows": rows[0]["running_first_zero"] + rows[0]["remaining"] if rows else 0,
            "metrics": rows
        })

    return results
