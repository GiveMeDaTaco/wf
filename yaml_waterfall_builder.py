# yaml_waterfall_builder.py  (patched)
# ---------------------------------------------------------------
from __future__ import annotations
import re, yaml, logging
from pathlib import Path
from typing import Any, Dict, List
from sqlalchemy import Engine, text

RE_OUTPUT = re.compile(r"^[A-Za-z_][\w$#]*\.[A-Za-z_][\w$#]*$")

# ── helpers ────────────────────────────────────────────────────
_comma   = lambda xs: ", ".join(xs)
_is_id   = lambda s: re.match(r"^[A-Za-z_][\w$#]*$", s)
_bare    = lambda s: s.split(".")[-1]          # strip a. → customer_id

def _validate(doc: Dict[str, Any]) -> None:
    if "settings" not in doc or "tables" not in doc:
        raise ValueError("YAML must have top‑level 'settings' and 'tables'.")
    chans = [k for k in doc if k not in ("settings", "tables")]
    if not chans or chans[0] != "base":
        raise ValueError("First channel must be literally 'base'.")
    for ch in chans:
        if not _is_id(ch):
            raise ValueError(f"Illegal channel name {ch}")
        if not RE_OUTPUT.match(doc[ch].get("output", "")):
            raise ValueError(f"{ch}: output must be schema.table")
        for ck in doc[ch].get("checks", []):
            if not ck.get("teradata_logic"):
                raise ValueError(f"{ch}: every check needs teradata_logic")
            if ch != "base" and not ck.get("template_id"):
                raise ValueError(f"{ch}: non‑base check missing template_id")
            if not ck.get("description", "").strip():
                raise ValueError(f"{ch}: description cannot be empty")
    if sum(j["join_type"].upper() == "FROM" for j in doc["tables"].values()) != 1:
        raise ValueError("Exactly one table must have join_type FROM")

# ── main builder ───────────────────────────────────────────────
def build_waterfall_tables(engine: Engine,
                           yaml_path: str | Path,
                           logger: logging.Logger) -> List[str]:
    doc = yaml.safe_load(Path(yaml_path).read_text())
    _validate(doc)

    settings      = doc["settings"]
    id_cols       = settings["count_columns"]        # may include aliases
    offer_code    = settings["offer_code"]
    ddl_ran: List[str] = []

    # ---- 1.  keep table order exactly as in YAML -----------------
    tables_items  = list(doc["tables"].items())
    base_tbl_key, base_tbl_info = next((k, v) for k, v in tables_items
                                       if v["join_type"].upper() == "FROM")
    first_alias   = base_tbl_info["alias"]            # e.g. a

    # ---- 2.  pre‑compute base‑check aliases ----------------------
    base_chk_aliases, per_tmpl_ctr = [], {}
    for ck in doc["base"]["checks"]:
        tmpl = ck.get("template_id", "BA")
        per_tmpl_ctr[tmpl] = per_tmpl_ctr.get(tmpl, 0) + 1
        base_chk_aliases.append(f"base_{tmpl}_{per_tmpl_ctr[tmpl]}")

    # ---- 3.  iterate over channels (base first) ------------------
    for ch in [k for k in doc if k not in ("settings", "tables")]:
        info            = doc[ch]
        out_schema, _   = info["output"].split(".")
        # ---- select list ----------------------------------------
        sel_cols   = _comma(id_cols)
        tmpl_ctr   = {}
        sel_checks = []
        for ck in info["checks"]:
            tmpl = ck.get("template_id", "BA")
            tmpl_ctr[tmpl] = tmpl_ctr.get(tmpl, 0) + 1
            alias = f"{ch}_{tmpl}_{tmpl_ctr[tmpl]}"
            sel_checks.append(
                f"CASE WHEN {ck['teradata_logic']} THEN 1 ELSE 0 END AS {alias}"
            )
        select_list = ",\n               ".join([sel_cols, *sel_checks])

        # ---- FROM/JOIN block ------------------------------------
        lines = [f"FROM {base_tbl_key} {first_alias}"]
        if ch != "base":                                     # add the filter join
            id_names  = [_bare(c) for c in id_cols]
            base_wf   = (
                "SELECT " + _comma(id_names) +
                f" FROM user_work.{offer_code}_base_waterfall " +
                "WHERE " + " AND ".join(f"{c}=1" for c in base_chk_aliases)
            )
            join_keys = " AND ".join(
                f"{first_alias}.{name} = base_wf.{name}" for name in id_names
            )
            lines.append(f"INNER JOIN ({base_wf}) base_wf ON {join_keys}")

        # remaining tables (skip the FROM table already used)
        for key, meta in tables_items:
            if key == base_tbl_key:           # already emitted
                continue
            jt   = meta["join_type"].upper()
            ali  = meta["alias"]
            oncl = meta["join_logic"]
            lines.append(f"{jt} {key} {ali} ON {oncl}")

        from_block = "\n    ".join(lines)

        # ---- full DDL -------------------------------------------
        ddl = f"""CREATE MULTISET TABLE {out_schema}.{offer_code}_{ch}_waterfall AS (
            SELECT
               {select_list}
            {from_block}
        ) WITH DATA"""
        idx = f"CREATE INDEX ({_comma([_bare(c) for c in id_cols])}) " \
              f"ON {out_schema}.{offer_code}_{ch}_waterfall"

        with engine.begin() as cx:
            cx.execute(text(ddl))
            cx.execute(text(idx))
        logger.info("Created %s.%s",
                    out_schema, f"{offer_code}_{ch}_waterfall")
        ddl_ran += [ddl, idx]

    return ddl_ran
