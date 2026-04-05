# IntentQL roadmap

This document describes how we plan to evolve IntentQL toward **broad coverage of natural-language questions that are answerable from an allowlisted database schema**—with deterministic, safe SQL and high consistency.

We do **not** claim every English sentence maps perfectly every time; we aim for **expressive completeness** of the intermediate representation (relative to SQL over the schema) and **high, measurable** quality on real questions.

---

## Phase 1 — Core question families (trust first)

Fix the failure modes users hit first when moving beyond “how many X per Y”:

| Area | Direction |
|------|-----------|
| **Detail / lookup** | Extract ID-like tokens from the question (`WO12345`, UUIDs) into equality filters on the right column; small `limit`; avoid full-table “list” plans with no selective filter. |
| **Time ranges** | Resolve phrases (“last 3 years”, “last quarter”, “last year”) into a **single** consistent mapping to `primary_date` and relative-date sentinels; clarify when two date columns could apply. |
| **Trends** | **Time bucketing** (`date_trunc` by month/year), not raw timestamp in `group_by`. |
| **Ratios / percentages** | Two metrics (numerator + denominator) or compound execution; division in the plan or merged result. |

**Deliverable:** Tests + few-shot examples per family so rephrasings stay stable.

**Status (implemented in library):** `QueryIntent` gains `time_bucket` and `aggregation: ratio`; `TimeRange` includes `last_2_years`, `last_3_years`, `last_6_months`. Normalization injects WO-/ID-like tokens onto `primary_id`, infers monthly/yearly buckets for trend phrasing, and may coerce to `list` for detail questions. `build_plan_from_intent` uses `=` filters on IDs, emits `date_trunc` via dimension `time_bucket`, and builds percentage plans as two scalar subqueries + `pct`. Regression coverage: `compile_intent_phase1_*` entries in `test/regression_test/test_qs.json`, executed via `python test/test_main.py lint` (or default suite).

---

## Phase 2 — Grow the IR toward “SQL over schema”

Extend `QueryIntent` / `QueryPlan` and the compiler so that, for tables and columns in `schema.yaml`, the plan can approach **what you could hand-write in Postgres**: joins, aggregates, CTEs, subqueries, filtered aggregates, window functions where needed, etc.—always through the **allowlist**, never raw LLM-generated SQL.

**Deliverable:** A checklist: “Can this SQL-over-schema query be expressed as a plan?” approaches yes for your benchmark queries.

---

## Phase 3 — Multi-step natural language

- **Decomposition:** Compare A vs B, “X as % of Y”, period-over-period → multiple sub-intents or plans, then merge for the final answer.
- **Clarification:** When the question is underspecified, return a structured **clarify** response instead of executing a wrong query.

**Deliverable:** Fewer confident wrong answers; explicit handling of ambiguity.

---

## Phase 4 — Quality loop (LLM + validation)

- **Intent memory**, **value index**, and **deterministic normalization** updated for each new family.
- **Semantic lint** rules (e.g. detail queries must have selective filters).
- **Regression suites** per domain (multi-building style tests as a template).

**Deliverable:** Coverage and consistency are measurable; CI catches regressions.

---

## Phase 5 — Product and documentation

- Public **cookbook**: lookup, trend, ratio, period-over-period, etc.
- Clear **limits**: ambiguous asks, narrative-only questions, concepts not in the schema.

---

## Definition of “good enough” for a release series

1. **Expressiveness:** The team can map essentially any SQL-over-`schema.yaml` query to a supported plan shape (or a small number of composed plans).
2. **Quality:** Benchmarks on **real** target-app questions meet a high bar; failures are mostly clarification or out-of-schema, not silent wrong SQL.

---

## Related

- [Documentation](https://certifore.github.io/intentql_docs) (user guides and API)
- Issues: [github.com/Certifore/intentql/issues](https://github.com/Certifore/intentql/issues)
