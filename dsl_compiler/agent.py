# dsl_compiler/agent.py
import yaml
from sqlalchemy.engine import Engine
from .compiler import Compiler
from .executor import Executor
from .llm_integration import get_llm_client

class QueryAgent:
    """Orchestrates the process of converting a question to a database query and response."""

    def __init__(self, schema_path: str, engine: Engine, llm_config: dict):
        with open(schema_path, 'r') as f:
            self.schema = yaml.safe_load(f)
        
        self.llm_client = get_llm_client(llm_config)
        self.compiler = Compiler(self.schema)
        self.executor = Executor(engine)

    def _get_system_prompt(self) -> str:
        """Generates the system prompt with schema context for the LLM."""
        schema_str = yaml.dump(self.schema)
        return f"""
You are a database query assistant. Your only job is to take a user's question
and convert it into a JSON object that follows a specific DSL format.
You must call the `process_dsl_query` tool with the generated JSON.

Here is the database schema you are working with:
{schema_str}

Here is an example of the DSL format (QueryPlan):
{{
  "dataset": "dataset_name",
  "metrics": [{{ "agg": "COUNT", "field": "*", "alias": "count" }}],
  "dimensions": [{{ "field": "column_name" }}],
  "filters": [{{ "field": "column_name", "op": "operator", "value": "some_value" }}],
  "limit": 100
}}

Based on the user's question, generate the appropriate JSON.
"""

    def ask(self, question: str) -> str:
        """
        Takes a user's question, generates DSL, compiles to SQL,
        executes it, and returns a natural language response.
        """
        system_prompt = self._get_system_prompt()
        
        try:
            # 1. Generate DSL using the LLM
            dsl_json = self.llm_client.generate_dsl(system_prompt, question)

            # 2. Compile DSL to SQL
            sql_query, params = self.compiler.compile(dsl_json)

            # 3. Execute SQL
            results = self.executor.execute(sql_query, params)
            if "error" in results:
                return f"Error executing query: {results['error']['message']}"

            # 4. Interpret results with LLM
            final_response = self.llm_client.interpret_results(question, results['rows'])
            
            return final_response
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
