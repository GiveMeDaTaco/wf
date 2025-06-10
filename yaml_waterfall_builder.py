
"""
Script #2 — YAML‑driven Teradata waterfall builder
--------------------------------------------------

Public function:

    build_waterfall_tables(engine, yaml_path: str | Path, logger) -> List[str]

* Validates YAML per the specification.
* Generates CREATE TABLE AS … statements for each channel (base first).
* Adds secondary indexes on the identifier (count) columns.
* Executes each DDL via the supplied SQLAlchemy engine.
* Returns the list of executed SQL strings (helpful for audit/tests).
"""

from __future__ import annotations
import re, logging, yaml
from pathlib import Path
from typing import Dict, Any, List
from sqlalchemy import Engine, text

RE_OUTPUT = re.compile(r"^[A-Za-z_][\w\$#]*\.[A-Za-z_][\w\$#]*$")

def _comma(cols: List[str]) -> str:
    return ", ".join(cols)

def _is_sql_identifier(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][\w\$#]*$", name))

def _validate_yaml(doc: Dict[str, Any]):
    if "settings" not in doc or "tables" not in doc:
        raise ValueError("YAML needs top‑level 'settings' and 'tables'.")
    channels = [k for k in doc if k not in ("settings", "tables")]
    if not channels or channels[0] != "base":
        raise ValueError("First channel must be 'base'.")
    for ch in channels:
        if not _is_sql_identifier(ch):
            raise ValueError(f"Illegal channel name '{ch}'.")
        out = doc[ch].get("output")
        if not (out and RE_OUTPUT.match(out)):
            raise ValueError(f"Channel '{ch}': output must be 'schema.table'.")
        for chk in doc[ch].get("checks", []):
            if not chk.get("teradata_logic"):
                raise ValueError(f"{ch}: every check needs 'teradata_logic'.")
            if ch != "base" and not chk.get("template_id"):
                raise ValueError(f"{ch}: non‑base checks need 'template_id'.")
            if not chk.get("description", "").strip():
                raise ValueError(f"{ch}: description cannot be empty.")
    if sum(1 for t in doc["tables"].values() if t["join_type"].upper() == "FROM") != 1:
        raise ValueError("Exactly one table must have join_type 'FROM'.")

def build_waterfall_tables(engine: Engine, yaml_path: str | Path, logger: logging.Logger) -> List[str]:
    doc = yaml.safe_load(Path(yaml_path).read_text())
    _validate_yaml(doc)

    settings = doc["settings"]
    count_columns: List[str] = settings["count_columns"]
    offer_code: str = settings["offer_code"]
    ddl_executed: List[str] = []

    # Build the FROM / JOIN block once
    tables_clause: List[str] = []
    for tbl_key, info in doc["tables"].items():
        jt = info["join_type"].upper()
        alias = info["alias"]
        if jt == "FROM":
            tables_clause.append(f"FROM {tbl_key} {alias}")
        else:
            join_logic = info["join_logic"]
            tables_clause.append(f"{jt} {tbl_key} {alias} ON {join_logic}")
    from_sql = "\n    ".join(tables_clause)

    channels = [k for k in doc if k not in ("settings", "tables")]

    for ch in channels:
        info = doc[ch]
        out_schema, _ = info["output"].split(".")
        select_cols = _comma(count_columns)
        select_checks: List[str] = []
        templ_counter: Dict[str, int] = {}

        for chk in info["checks"]:
            templ = chk.get("template_id", "BA")
            templ_counter[templ] = templ_counter.get(templ, 0) + 1
            alias = f"{ch}_{templ}_{templ_counter[templ]}"
            logic = chk["teradata_logic"]
            select_checks.append(f"CASE WHEN {logic} THEN 1 ELSE 0 END AS {alias}")

        select_list = ",\n               ".join([select_cols, *select_checks])

        if ch == "base":
            from_part = from_sql
        else:
            base_tbl = f"user_work.{offer_code}_base_waterfall"
            # build list of base check aliases
            tmp_ctr, base_aliases = {}, []
            for chk in doc["base"]["checks"]:
                templ = chk.get("template_id", "BA")
                tmp_ctr[templ] = tmp_ctr.get(templ, 0) + 1
                base_aliases.append(f"base_{templ}_{tmp_ctr[templ]}")
            wf_subquery = (
                "SELECT " + _comma([c.split(".")[-1] for c in count_columns]) +
                f" FROM {base_tbl} WHERE " +
                " AND ".join(f"{c}=1" for c in base_aliases)
            )
            first_alias = doc["tables"][next(iter(doc["tables"]))]["alias"]
            join_keys = " AND ".join(
                f"{first_alias}.{c.split('.')[-1]} = base_wf.{c.split('.')[-1]}"
                for c in count_columns
            )
            from_part = (
                f"FROM (\n        {wf_subquery}\n    ) base_wf\n    " +
                from_sql.replace("FROM ", f"INNER JOIN (") +
                f" ON {join_keys}\n    )"
            )

        ddl = f"""CREATE MULTISET TABLE {out_schema}.{offer_code}_{ch}_waterfall AS (
            SELECT
               {select_list}
            {from_part}
        ) WITH DATA""".strip()

        idx_stmt = f"CREATE INDEX ({_comma([c.split('.')[-1] for c in count_columns])}) ON {out_schema}.{offer_code}_{ch}_waterfall"

        with engine.begin() as conn:
            conn.execute(text(ddl))
            conn.execute(text(idx_stmt))

        logger.info("Created table %s.%s", out_schema, f"{offer_code}_{ch}_waterfall")
        ddl_executed.extend([ddl, idx_stmt])

    return ddl_executed
