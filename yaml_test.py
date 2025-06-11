from pathlib import Path
import logging, types, sys

# ---- stub sqlalchemy so `text()` & `Engine` exist ----
sqlalchemy_stub = types.ModuleType("sqlalchemy")
sqlalchemy_stub.text = lambda s: s
class DummyEngine:                       # prints every SQL executed
    def begin(self):
        class _Ctx:
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def execute(self, sql): print("⮕", sql, "\n")
        return _Ctx()
sys.modules["sqlalchemy"] = sqlalchemy_stub

# ---- example YAML -----------------------------------
Path("sample.yaml").write_text("""
settings:
  count_columns: [a.customer_id, a.site_id]
  offer_code: OFF123

base:
  output: user_work.base_waterfall
  checks:
    - teradata_logic: "a.signup_date >= DATE '2025-01-01'"
      description: "Signed‑up in 2025"
      template_id: BA
    - teradata_logic: "a.email_opt_in = 1"
      description: "Opt‑in"
      template_id: BA

email:
  output: user_work.email_waterfall
  checks:
    - teradata_logic: "b.email_sent_flag = 1"
      description: "Email sent"
      template_id: A
    - teradata_logic: "b.email_open_flag = 1"
      description: "Email opened"
      template_id: A

tables:
  mktg.customer a:
    join_type: FROM
    join_logic: ""
    alias: a
  mktg.email b:
    join_type: LEFT JOIN
    join_logic: "a.customer_id = b.customer_id"
    alias: b
""")

import yaml_waterfall_builder   # (patched code above)
logger = logging.getLogger("demo")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

yaml_waterfall_builder.build_waterfall_tables(
    engine=DummyEngine(),
    yaml_path="sample.yaml",
    logger=logger
)
