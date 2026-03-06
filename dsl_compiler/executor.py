# dsl_compiler/executor.py
from __future__ import annotations

from typing import Any, Dict
from sqlalchemy.engine import Engine
from sqlalchemy import text as sqla_text
import datetime
from decimal import Decimal

from .exceptions import DatabaseExecutionError

def _jsonify(v: Any) -> Any:
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v

class Executor:
    def __init__(self, engine: Engine):
        self.engine = engine

    def execute(self, sql: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with self.engine.begin() as conn:
                # If SQL already contains psycopg2-style %(name)s binds, pass through
                if "%(" in sql:
                    res = conn.exec_driver_sql(sql, params)
                else:
                    res = conn.execute(sqla_text(sql), params)

                if not res.returns_rows:
                    return {"rows": [], "row_count": 0, "columns": []}

                mappings = res.mappings().all()
                rows = [{k: _jsonify(v) for k, v in dict(r).items()} for r in mappings]
                cols = list(res.keys())

                return {"rows": rows, "row_count": len(rows), "columns": cols}

        except Exception as e:
            raise DatabaseExecutionError(str(e), sql=sql, original=e) from e