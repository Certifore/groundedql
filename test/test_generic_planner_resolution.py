"""Focused regression tests for generic schema and relational resolution."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intentql.compiler import Compiler
from intentql.evidence_planner import build_evidence_plan
from intentql.intent_normalize import normalize_intent
from intentql.join_planner import auto_inject_joins


def _linked_schema() -> dict:
    return {
        "version": 1,
        "dialect": "postgres",
        "tables": [
            {
                "name": "facts",
                "db_table": "facts",
                "columns": [
                    {"name": "fact_id", "db_column": "fact_id", "type": "bigint"},
                    {"name": "account_id", "db_column": "account_id", "type": "bigint"},
                    {"name": "location_id", "db_column": "location_id", "type": "bigint"},
                ],
                "primary_id": "fact_id",
            },
            {
                "name": "accounts",
                "db_table": "accounts",
                "columns": [{"name": "account_id", "db_column": "account_id", "type": "bigint"}],
                "primary_id": "account_id",
            },
            {
                "name": "locations",
                "db_table": "locations",
                "columns": [
                    {"name": "location_id", "db_column": "location_id", "type": "bigint"},
                    {"name": "region", "db_column": "region", "type": "text"},
                ],
                "primary_id": "location_id",
            },
        ],
        "links": [
            {
                "name": "facts_to_accounts",
                "from_table": "facts",
                "to_table": "accounts",
                "join_type": "left",
                "on": [{"left": "facts.account_id", "op": "=", "right": "accounts.account_id"}],
            },
            {
                "name": "facts_to_locations",
                "from_table": "facts",
                "to_table": "locations",
                "join_type": "left",
                "on": [{"left": "facts.location_id", "op": "=", "right": "locations.location_id"}],
            },
        ],
    }


def test_invalid_qualified_hint_rebinds_to_unique_linked_owner() -> None:
    intent = {
        "dataset": "facts",
        "aggregation": "list",
        "filters": [],
        "group_by": ["accounts.region"],
        "output_columns": ["accounts.region"],
        "sort_column": "accounts.region",
    }
    normalized = normalize_intent(intent, _linked_schema())
    assert normalized["group_by"] == ["locations.region"]
    assert normalized["output_columns"] == ["locations.region"]
    assert normalized["sort_column"] == "locations.region"


def test_relative_average_builds_typed_scoped_subquery() -> None:
    schema = {
        "version": 1,
        "dialect": "postgres",
        "tables": [
            {
                "name": "records",
                "db_table": "records",
                "columns": [
                    {"name": "record_id", "db_column": "record_id", "type": "bigint"},
                    {"name": "severity", "db_column": "severity", "type": "bigint"},
                    {"name": "cohort", "db_column": "cohort", "type": "text"},
                    {"name": "score", "db_column": "score", "type": "float"},
                ],
                "primary_id": "record_id",
            }
        ],
        "links": [],
    }
    question = (
        "What number of records with severity level 2 and cohort of only A "
        "have score 20% higher than average?"
    )
    plan = build_evidence_plan(question, schema, value_index={"records": {"cohort": ["A"]}})
    assert plan is not None
    assert plan["select"][0]["expr"]["func"] in {"count", "count_distinct"}
    where_text = repr(plan["where"])
    assert "scalar_subquery" in where_text
    assert "records.score" in where_text
    assert "records.cohort" in where_text
    assert "records.severity" in where_text


def test_distinct_on_compiles_as_relational_operator() -> None:
    schema = _linked_schema()
    plan = {
        "version": "1.0",
        "dataset": "locations",
        "select": [{"expr": {"col": "locations.region"}, "alias": "region"}],
        "distinct_on": [{"col": "locations.location_id"}],
        "order_by": [{"by": {"col": "locations.location_id"}, "dir": "asc"}],
        "limit": None,
        "offset": 0,
    }
    sql, _params = Compiler(schema).compile(plan)
    assert "DISTINCT ON" in sql
    assert "ORDER BY" in sql


def test_evidence_lists_and_aliases_stay_scoped() -> None:
    schema = {
        "version": 1,
        "dialect": "postgres",
        "tables": [
            {
                "name": "entities",
                "db_table": "entities",
                "columns": [
                    {"name": "entity_id", "db_column": "entity_id", "type": "bigint"},
                    {"name": "status", "db_column": "status", "type": "text"},
                ],
                "primary_id": "entity_id",
            },
            {
                "name": "measurements",
                "db_table": "measurements",
                "columns": [
                    {"name": "entity_id", "db_column": "entity_id", "type": "bigint"},
                    {"name": "marker", "db_column": "marker", "type": "text"},
                    {"name": "other_marker", "db_column": "other_marker", "type": "text"},
                ],
            },
        ],
        "links": [
            {
                "name": "measurements_to_entities",
                "from_table": "measurements",
                "to_table": "entities",
                "join_type": "left",
                "on": [{"left": "measurements.entity_id", "op": "=", "right": "entities.entity_id"}],
            }
        ],
    }
    question = (
        "How many entities have a normal marker and are active?\n\n"
        "Evidence: normal marker refers to marker IN('-', '+-'); "
        "'-' means 'negative'; '+-' means '0'; active refers to status = 'active';"
    )
    value_index = {
        "measurements": {
            "marker": ["negative", "0"],
            "other_marker": ["negative", "0"],
        },
        "entities": {"status": ["active"]},
    }
    plan = build_evidence_plan(question, schema, value_index=value_index)
    assert plan is not None
    where = repr(plan["where"])
    assert "measurements.marker" in where
    assert "measurements.other_marker" not in where
    assert "['negative', '0']" in where


def test_ranked_row_sum_uses_role_specific_join() -> None:
    schema = {
        "version": 1,
        "dialect": "postgres",
        "tables": [
            {
                "name": "events",
                "db_table": "events",
                "columns": [
                    {"name": "home_group_id", "db_column": "home_group_id", "type": "bigint"},
                    {"name": "away_group_id", "db_column": "away_group_id", "type": "bigint"},
                    {"name": "home_score", "db_column": "home_score", "type": "bigint"},
                    {"name": "away_score", "db_column": "away_score", "type": "bigint"},
                ],
            },
            {
                "name": "groups",
                "db_table": "groups",
                "columns": [
                    {"name": "group_id", "db_column": "group_id", "type": "bigint"},
                    {"name": "name", "db_column": "name", "type": "text"},
                ],
                "primary_id": "group_id",
            },
        ],
        "links": [
            {
                "name": "events_to_groups_home_group_id",
                "from_table": "events",
                "to_table": "groups",
                "join_type": "left",
                "on": [{"left": "events.home_group_id", "op": "=", "right": "groups.group_id"}],
            },
            {
                "name": "events_to_groups_away_group_id",
                "from_table": "events",
                "to_table": "groups",
                "join_type": "left",
                "on": [{"left": "events.away_group_id", "op": "=", "right": "groups.group_id"}],
            },
        ],
    }
    question = (
        "Which away group scored the most?\n\n"
        "Evidence: Final result should return the Groups.name; "
        "away group refers to away_group_id; scored the most refers to MAX(SUM(home_score, away_score));"
    )
    plan = build_evidence_plan(question, schema)
    assert plan is not None
    assert plan["group_by"] == [{"col": "groups.name"}]
    assert plan.get("joins") == [{"link": "events_to_groups_away_group_id"}], plan


def test_null_related_attribute_uses_inner_join_override() -> None:
    schema = _linked_schema()
    question = (
        "How many facts have no region value?\n\n"
        "Evidence: no region value refers to locations.region IS NULL;"
    )
    plan = build_evidence_plan(question, schema)
    assert plan is not None
    assert plan.get("joins") == [{"link": "facts_to_locations", "type": "inner"}], plan
    body = {key: value for key, value in plan.items() if key != "meta"}
    sql, _params = Compiler(schema).compile(auto_inject_joins(body, schema))
    assert "LEFT OUTER JOIN facts" not in sql
    assert "JOIN facts" in sql


def test_direct_join_beats_textually_relevant_detour() -> None:
    schema = {
        "version": 1,
        "dialect": "postgres",
        "tables": [
            {
                "name": "events",
                "db_table": "events",
                "columns": [{"name": "event_id", "db_column": "event_id", "type": "bigint"}],
                "primary_id": "event_id",
            },
            {
                "name": "budgets",
                "db_table": "budgets",
                "columns": [
                    {"name": "budget_id", "db_column": "budget_id", "type": "bigint"},
                    {"name": "event_id", "db_column": "event_id", "type": "bigint"},
                    {"name": "amount", "db_column": "amount", "type": "float"},
                ],
                "primary_id": "budget_id",
            },
            {
                "name": "expenses",
                "db_table": "expenses",
                "columns": [
                    {"name": "budget_id", "db_column": "budget_id", "type": "bigint"},
                    {"name": "member_id", "db_column": "member_id", "type": "bigint"},
                ],
            },
            {
                "name": "members",
                "db_table": "members",
                "columns": [
                    {"name": "member_id", "db_column": "member_id", "type": "bigint"},
                    {"name": "event_id", "db_column": "event_id", "type": "bigint"},
                ],
                "primary_id": "member_id",
            },
        ],
        "links": [
            {
                "name": "budgets_to_events",
                "from_table": "budgets",
                "to_table": "events",
                "join_type": "left",
                "on": [{"left": "budgets.event_id", "op": "=", "right": "events.event_id"}],
            },
            {
                "name": "expenses_to_budgets",
                "from_table": "expenses",
                "to_table": "budgets",
                "join_type": "left",
                "on": [{"left": "expenses.budget_id", "op": "=", "right": "budgets.budget_id"}],
            },
            {
                "name": "expenses_to_members",
                "from_table": "expenses",
                "to_table": "members",
                "join_type": "left",
                "on": [{"left": "expenses.member_id", "op": "=", "right": "members.member_id"}],
            },
            {
                "name": "members_to_events",
                "from_table": "members",
                "to_table": "events",
                "join_type": "left",
                "on": [{"left": "members.event_id", "op": "=", "right": "events.event_id"}],
            },
        ],
    }
    question = (
        "Which event has the highest budget amount?\n\n"
        "Evidence: event refers to events.event_id; highest amount refers to MAX(budgets.amount);"
    )
    plan = build_evidence_plan(question, schema)
    assert plan is not None
    body = {key: value for key, value in plan.items() if key != "meta"}
    joined = auto_inject_joins(body, schema)
    assert joined.get("joins") == [{"link": "budgets_to_events"}], joined


def test_human_entity_count_prefers_detail_table() -> None:
    schema = {
        "version": 1,
        "dialect": "postgres",
        "tables": [
            {
                "name": "sessions",
                "db_table": "sessions",
                "columns": [
                    {"name": "session_id", "db_column": "session_id", "type": "bigint"},
                    {"name": "name", "db_column": "name", "type": "text"},
                ],
                "primary_id": "session_id",
            },
            {
                "name": "people",
                "db_table": "people",
                "columns": [
                    {"name": "person_id", "db_column": "person_id", "type": "bigint"},
                    {"name": "first_name", "db_column": "first_name", "type": "text"},
                    {"name": "last_name", "db_column": "last_name", "type": "text"},
                    {"name": "size", "db_column": "size", "type": "text"},
                ],
                "primary_id": "person_id",
            },
            {
                "name": "attendance",
                "db_table": "attendance",
                "columns": [
                    {"name": "session_id", "db_column": "session_id", "type": "bigint"},
                    {"name": "person_id", "db_column": "person_id", "type": "bigint"},
                ],
            },
        ],
        "links": [
            {
                "name": "attendance_to_sessions",
                "from_table": "attendance",
                "to_table": "sessions",
                "join_type": "left",
                "on": [{"left": "attendance.session_id", "op": "=", "right": "sessions.session_id"}],
            },
            {
                "name": "attendance_to_people",
                "from_table": "attendance",
                "to_table": "people",
                "join_type": "left",
                "on": [{"left": "attendance.person_id", "op": "=", "right": "people.person_id"}],
            },
        ],
    }
    question = (
        "How many students attended the session named Launch and have size Medium?\n\n"
        "Evidence: session named Launch refers to sessions.name = 'Launch'; size Medium refers to people.size = 'Medium';"
    )
    plan = build_evidence_plan(question, schema)
    assert plan is not None
    assert "people.person_id" in repr(plan["select"]), plan


def test_quoted_date_variants_are_deduplicated() -> None:
    schema = {
        "version": 1,
        "dialect": "postgres",
        "tables": [
            {
                "name": "records",
                "db_table": "records",
                "columns": [
                    {"name": "record_id", "db_column": "record_id", "type": "bigint"},
                    {"name": "record_date", "db_column": "record_date", "type": "date"},
                ],
                "primary_id": "record_id",
            }
        ],
        "links": [],
    }
    question = "How many records on 2019/8/20?\n\nEvidence: on 2019/8/20 refers to record_date = '2019-8-20';"
    plan = build_evidence_plan(question, schema)
    assert plan is not None
    assert repr(plan["where"]).count("records.record_date") == 1, plan


def test_count_threshold_groups_the_counted_entity() -> None:
    schema = {
        "version": 1,
        "dialect": "postgres",
        "tables": [
            {
                "name": "sessions",
                "db_table": "sessions",
                "columns": [
                    {"name": "session_id", "db_column": "session_id", "type": "bigint"},
                    {"name": "kind", "db_column": "kind", "type": "text"},
                ],
                "primary_id": "session_id",
            },
            {
                "name": "attendance",
                "db_table": "attendance",
                "columns": [
                    {"name": "session_id", "db_column": "session_id", "type": "bigint"},
                    {"name": "person_id", "db_column": "person_id", "type": "bigint"},
                ],
            },
            {
                "name": "people",
                "db_table": "people",
                "columns": [
                    {"name": "person_id", "db_column": "person_id", "type": "bigint"},
                    {"name": "first_name", "db_column": "first_name", "type": "text"},
                ],
                "primary_id": "person_id",
            },
        ],
        "links": [
            {
                "name": "attendance_to_sessions",
                "from_table": "attendance",
                "to_table": "sessions",
                "join_type": "left",
                "on": [{"left": "attendance.session_id", "op": "=", "right": "sessions.session_id"}],
            },
            {
                "name": "attendance_to_people",
                "from_table": "attendance",
                "to_table": "people",
                "join_type": "left",
                "on": [{"left": "attendance.person_id", "op": "=", "right": "people.person_id"}],
            },
        ],
    }
    question = (
        "Among sessions attended by more than 10 people, how many are workshops?\n\n"
        "Evidence: workshops refers to sessions.kind = 'Workshop'; "
        "attended by more than 10 people refers to COUNT(session_id) > 10;"
    )
    plan = build_evidence_plan(question, schema)
    assert plan is not None
    assert plan["group_by"] == [{"col": "sessions.session_id"}], plan
    assert "sessions.session_id" in repr(plan["select"]), plan


def _entity_metric_schema() -> dict:
    return {
        "version": 1,
        "dialect": "postgres",
        "tables": [
            {
                "name": "entities",
                "db_table": "entities",
                "columns": [
                    {"name": "entity_id", "db_column": "entity_id", "type": "bigint"},
                    {"name": "entity_name", "db_column": "entity_name", "type": "text"},
                    {"name": "birthday", "db_column": "birthday", "type": "text"},
                    {"name": "height", "db_column": "height", "type": "bigint"},
                ],
                "primary_id": "entity_id",
            },
            {
                "name": "entity_metrics",
                "db_table": "entity_metrics",
                "columns": [
                    {"name": "metric_id", "db_column": "metric_id", "type": "bigint"},
                    {"name": "entity_id", "db_column": "entity_id", "type": "bigint"},
                    {"name": "date", "db_column": "date", "type": "text"},
                    {"name": "buildscore", "db_column": "buildscore", "type": "bigint"},
                    {"name": "accuracy", "db_column": "accuracy", "type": "bigint"},
                    {"name": "rating", "db_column": "rating", "type": "bigint"},
                    {"name": "status", "db_column": "status", "type": "text"},
                ],
                "primary_id": "metric_id",
            },
        ],
        "links": [
            {
                "name": "entity_metrics_to_entities",
                "from_table": "entity_metrics",
                "to_table": "entities",
                "join_type": "left",
                "on": [{"left": "entity_metrics.entity_id", "op": "=", "right": "entities.entity_id"}],
            }
        ],
    }


def test_compact_identifier_ranked_metric_values() -> None:
    question = (
        "What are the build score values of the top 4 entities with the highest build score?\n\n"
        "Evidence: build score refers to buildScore; highest build score refers to MAX(buildScore);"
    )
    plan = build_evidence_plan(question, _entity_metric_schema())
    assert plan is not None
    assert plan["limit"] == 4, plan
    assert plan["select"][0]["expr"] == {"col": "entity_metrics.buildscore"}, plan
    assert plan["order_by"][0]["dir"] == "desc", plan


def test_ranked_formula_groups_entity_identity() -> None:
    question = (
        "List the top 10 entities' names in descending order of average accuracy.\n\n"
        "Evidence: average accuracy = DIVIDE(SUM(accuracy), COUNT(entity_id));"
    )
    plan = build_evidence_plan(question, _entity_metric_schema())
    assert plan is not None
    assert plan["limit"] == 10, plan
    assert {"col": "entities.entity_id"} in plan["group_by"], plan
    assert "entity_metrics.accuracy" in repr(plan["order_by"]), plan


def test_metric_time_range_uses_metric_table_date() -> None:
    question = (
        "From 2010 to 2015, what was the average rating of entities higher than 170?\n\n"
        "Evidence: average rating = SUM(rating) / COUNT(metric_id); higher than 170 refers to height > 170;"
    )
    plan = build_evidence_plan(question, _entity_metric_schema())
    assert plan is not None
    assert "'func': 'sum'" in repr(plan["select"]), plan
    assert "'func': 'count'" in repr(plan["select"]), plan
    assert "entity_metrics.date" in repr(plan["where"]), plan
    assert "entities.birthday" not in repr(plan["where"]), plan


def test_conditional_average_formula_uses_metric_values() -> None:
    question = (
        "What is the difference of average accuracy between Alpha One and Beta Two?\n\n"
        "Evidence: difference = SUBTRACT(AVG(accuracy WHERE entity_name = 'Alpha One'), "
        "AVG(accuracy WHERE entity_name = 'Beta Two'));"
    )
    plan = build_evidence_plan(question, _entity_metric_schema())
    assert plan is not None
    assert "entity_metrics.accuracy" in repr(plan["select"]), plan
    assert "entities.entity_name" in repr(plan["select"]), plan


def test_explicit_sum_count_ratio_is_preserved() -> None:
    question = (
        "From 2010 to 2015, what was the average rating of entities higher than 170?\n\n"
        "Evidence: average rating = SUM(rating) / COUNT(metric_id); higher than 170 refers to height > 170;"
    )
    plan = build_evidence_plan(question, _entity_metric_schema())
    assert plan is not None
    select_text = repr(plan["select"])
    assert "'func': 'sum'" in select_text, plan
    assert "'func': 'count'" in select_text, plan


def test_comparative_names_do_not_become_conflicting_equalities() -> None:
    question = (
        "Which of these entities is older, Alpha One or Beta Two?\n\n"
        "Evidence: The larger the birthday value, the younger the entity is, and vice versa;"
    )
    value_index = {"entities": {"entity_name": ["Alpha One", "Beta Two"]}}
    plan = build_evidence_plan(question, _entity_metric_schema(), value_index=value_index)
    assert plan is not None
    where = repr(plan["where"])
    assert "'op': 'in'" in where, plan
    assert "'op': '='" not in where, plan


def main() -> None:
    test_invalid_qualified_hint_rebinds_to_unique_linked_owner()
    test_relative_average_builds_typed_scoped_subquery()
    test_distinct_on_compiles_as_relational_operator()
    test_evidence_lists_and_aliases_stay_scoped()
    test_ranked_row_sum_uses_role_specific_join()
    test_null_related_attribute_uses_inner_join_override()
    test_direct_join_beats_textually_relevant_detour()
    test_human_entity_count_prefers_detail_table()
    test_quoted_date_variants_are_deduplicated()
    test_count_threshold_groups_the_counted_entity()
    test_compact_identifier_ranked_metric_values()
    test_ranked_formula_groups_entity_identity()
    test_metric_time_range_uses_metric_table_date()
    test_conditional_average_formula_uses_metric_values()
    test_explicit_sum_count_ratio_is_preserved()
    test_comparative_names_do_not_become_conflicting_equalities()
    print("ok: generic planner resolution")


if __name__ == "__main__":
    main()
