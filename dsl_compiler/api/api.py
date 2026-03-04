# dsl_compiler/api.py
import yaml
from sqlalchemy.engine import Engine
from ..compiler import Compiler
from ..executor import Executor

def execute_query_plan(*, engine: Engine, schema_path: str, query_plan: dict) -> dict:
    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f) or {}

    compiler = Compiler(schema)
    sql, params = compiler.compile(query_plan)

    executor = Executor(engine)
    result = executor.execute(sql, params)

    result["sql"] = sql
    result["params"] = params
    return result