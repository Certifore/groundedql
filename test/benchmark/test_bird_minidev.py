from __future__ import annotations

import json
import tempfile
from pathlib import Path

from bird_minidev import load_examples, normalize_rows, rows_match, schema_path_for


def test_load_examples() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mini_dev_postgresql.json"
        path.write_text(
            json.dumps([
                {
                    "db_id": "financial",
                    "question": "How many accounts are there?",
                    "evidence": "Use accounts.",
                    "SQL": "SELECT COUNT(*) FROM accounts",
                }
            ]),
            encoding="utf-8",
        )
        examples = load_examples(path)

    assert len(examples) == 1
    assert examples[0].db_id == "financial"
    assert examples[0].sql.lower().startswith("select")


def test_schema_path_for() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "financial").mkdir()
        expected = root / "financial" / "schema.yaml"
        expected.write_text("tables: []\n", encoding="utf-8")
        assert schema_path_for("financial", root) == expected


def test_rows_match_ignores_order_by_default() -> None:
    left = [(2, "b"), (1, "a")]
    right = [(1, "a"), (2, "b")]
    assert rows_match(left, right)
    assert normalize_rows(left, ordered=True) != normalize_rows(right, ordered=True)


def main() -> None:
    test_load_examples()
    test_schema_path_for()
    test_rows_match_ignores_order_by_default()
    print("ok: BIRD Mini-Dev benchmark helpers")


if __name__ == "__main__":
    main()
