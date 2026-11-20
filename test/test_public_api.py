import copy
import unittest
from pathlib import Path

from groundedql import compile_query_plan, plan_query
from groundedql.exceptions import QueryPlanError

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = str(ROOT / "config" / "schema.yaml")


class TestPublicCompilationAPI(unittest.TestCase):
    def test_compiles_without_database(self):
        plan = {
            "version": "1.0",
            "dataset": "customers",
            "dimensions": [{"field": "country"}],
            "filters": [
                {
                    "field": "country",
                    "op": "=",
                    "value": "Ghana",
                }
            ],
        }

        result = compile_query_plan(
            schema_path=SCHEMA_PATH,
            query_plan=plan,
        )

        self.assertIn("sql", result)
        self.assertIn("params", result)
        self.assertIn("SELECT", result["sql"].upper())
        self.assertIn("customers", result["sql"].lower())
        self.assertNotIn("Ghana", result["sql"])
        self.assertIn("Ghana", result["params"].values())

    def test_rejects_invalid_dataset(self):
        plan = {
            "version": "1.0",
            "dataset": "nonexistent_table",
            "dimensions": [{"field": "country"}],
        }

        with self.assertRaises(QueryPlanError):
            compile_query_plan(
                schema_path=SCHEMA_PATH,
                query_plan=plan,
            )

    def test_does_not_modify_original_plan(self):
        plan = {
            "version": "1.0",
            "dataset": "customers",
            "dimensions": [{"field": "country"}],
            "meta": {
                "source": "test",
            },
        }

        original = copy.deepcopy(plan)

        compile_query_plan(
            schema_path=SCHEMA_PATH,
            query_plan=plan,
        )

        self.assertEqual(plan, original)


class TestPublicPlanningAPI(unittest.TestCase):
    def test_plan_query_without_database(self):
        from unittest.mock import patch

        expected_plan = {
            "version": "1.0",
            "dataset": "customers",
            "dimensions": [{"field": "country"}],
        }

        with patch("groundedql.planner.QueryPlanPlanner") as mock_planner:
            mock_planner.return_value.plan_with_retry.return_value = expected_plan

            result = plan_query(
                question="Show customers by country",
                schema_path=SCHEMA_PATH,
                llm=object(),
            )

            self.assertEqual(result, expected_plan)

    def test_planning_and_compilation(self):
        from unittest.mock import patch

        expected_plan = {
            "version": "1.0",
            "dataset": "customers",
            "dimensions": [{"field": "country"}],
            "filters": [
                {
                    "field": "country",
                    "op": "=",
                    "value": "Ghana",
                }
            ],
        }

        with patch("groundedql.planner.QueryPlanPlanner") as mock_planner:
            mock_planner.return_value.plan_with_retry.return_value = expected_plan

            plan = plan_query(
                question="Show customers in Ghana",
                schema_path=SCHEMA_PATH,
                llm=object(),
            )

        result = compile_query_plan(
            schema_path=SCHEMA_PATH,
            query_plan=plan,
        )

        self.assertIn("SELECT", result["sql"].upper())
        self.assertNotIn("Ghana", result["sql"])
        self.assertIn("Ghana", result["params"].values())


class TestCombinedPublicAPI(unittest.TestCase):
    def test_plan_and_compile_without_database(self):
        from unittest.mock import patch

        from groundedql import plan_and_compile_query

        expected_plan = {
            "version": "1.0",
            "dataset": "customers",
            "dimensions": [{"field": "country"}],
            "filters": [
                {
                    "field": "country",
                    "op": "=",
                    "value": "Ghana",
                }
            ],
        }

        with patch("groundedql.planner.QueryPlanPlanner") as mock_planner:
            mock_planner.return_value.plan_with_retry.return_value = expected_plan

            result = plan_and_compile_query(
                question="Show customers in Ghana",
                schema_path=SCHEMA_PATH,
                llm=object(),
            )

        self.assertEqual(result["query_plan"], expected_plan)
        self.assertIn("SELECT", result["sql"].upper())
        self.assertNotIn("Ghana", result["sql"])
        self.assertIn("Ghana", result["params"].values())


if __name__ == "__main__":
    unittest.main()
