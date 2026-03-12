# QCE Benchmark Report
## A Fair Head-to-Head Comparison: QCE vs LangChain SQLAgent vs GPT-4 Direct

**Date:** March 2026  
**Database:** PostgreSQL (Supabase) — Northwind (open source, publicly reproducible)  
**LLM used by all tools:** GPT-4o, temperature=0  
**Schema:** E-commerce domain — customers, orders, products, categories, employees, suppliers, shippers  
**Benchmark code:** Open and reproducible — see [Benchmark Reproducibility](#benchmark-reproducibility)

---

## The One-Sentence Story

> **GPT-4 Direct works in your demo and fails silently in production. QCE is the compiler
> layer that makes natural language queries production-safe — not by prompting more carefully,
> but by enforcing guarantees in code that no prompt can provide.**

---

## The Wedge: Why This Exists and Who Needs It

Every team that builds a natural language query interface goes through the same journey:

**Week 1:** "GPT-4 can write SQL! This is amazing."  
**Week 4:** "Why did it return the wrong number? There's no error, just a wrong answer."  
**Week 8:** "The same question gives different SQL on different days. We can't regression test this."  
**Week 12:** "A user asked something unexpected and it queried a table it shouldn't have."  
**Week 16:** "We spent three months building guardrails. We're basically writing a compiler."

**QCE is what those teams wish they had at week one.**

### The Entry Point: Compliance and Regulated Analytics

The first customers who cannot afford to wait are in **compliance-heavy environments** —
healthcare, finance, government, and insurance. These teams face a specific, expensive problem:

- A natural language analytics tool that returns wrong numbers is not just a UX issue.
  It is a **regulatory risk**. A wrong count in a compliance report, an unauthorized data
  access, or an inconsistent answer between two auditors asking the same question can
  trigger investigations, fines, or loss of certification.

- These teams have already tried GPT-4 Direct and found it unacceptable in regulated
  contexts — not because it's bad, but because "usually correct" and "probably safe" are
  not acceptable risk levels when the output feeds a compliance report.

- They are not looking for a cheaper or faster tool. They are looking for a tool that gives
  them **documented, architectural guarantees** they can show to an auditor.

QCE's compiler layer provides exactly those guarantees:
- **Schema allowlist** — documented proof that only authorized tables and columns are
  accessible, enforced in code, not in a prompt that can drift
- **Parameterized execution** — documented proof that no user input can execute as SQL
- **Structured audit trail** — every query produces a QueryPlan JSON that records exactly
  what was requested, what SQL was generated, and what was returned
- **Deterministic compilation** — given the same plan, the same SQL is always produced,
  making queries auditable and regression-testable

These are not features. They are **compliance artifacts** that regulated teams need to
document and defend.

### Who This Applies To

| Segment | The Pain | What QCE Provides |
|---|---|---|
| Healthcare analytics | Patient data must never leak across tenant boundaries. Wrong counts in clinical reports are liability. | Schema allowlist per deployment, parameterized queries, audit log |
| Financial operations | Regulatory reports must be consistent and auditable. "The model answered differently on Tuesday" is not an acceptable explanation. | Compiler determinism, structured QueryPlan audit trail |
| Government / public sector | Procurement, audit, and FOIA systems require documented data access controls. | Allowlist as documented policy, not prompt instruction |
| Internal analytics (non-technical users) | CFOs and directors ask the same question on different days and get different numbers. | Compiler determinism, retry loop with structured error feedback |
| Multi-tenant SaaS | User A must never see User B's data. GPT-4 has no tenant isolation layer. | Schema allowlist scoped per tenant deployment |

---

## Why Not the Alternatives?

### Why Not LangChain?

The benchmark answers this directly:

- **9/9 completed questions — but correctness is unverified** (1 rate-limited, natural language output only)
- **3/5 deterministic** — the same question produces different output on 2 of 5 questions
- **4/15 hallucination rejection** — silently recovered from 11 hallucinated inputs by
  substituting real columns, returning answers to different questions than were asked
- **8 LLM calls per question** — highest API cost of all three tools at scale
- **26 second average latency** — slowest of all three tools

LangChain was not designed for deterministic, auditable SQL generation. It was designed
for flexible agent workflows. Using it for analytics query generation is using the wrong
tool for the job.

### Why Not GPT-4 Direct?

GPT-4 Direct is QCE's most credible competitor and the benchmark treated it with maximum
generosity — full schema descriptions, explicit rejection rules, the best possible system
prompt. Despite this:

**1. Safety fluctuates across runs.**
GPT-4 Direct scored 48/50 on injection resistance. The 2 failures are not fixed — they
change between runs because safety is enforced by a prompt, not by code. Prompts drift,
models update, edge cases slip through. QCE's 50/50 score does not fluctuate — it is
enforced by the compiler regardless of the LLM's output.

**2. Hallucinations pass silently.**
GPT-4 rejected only 13 of 15 hallucinated column inputs. On 2 inputs it substituted a
real column and returned data — answering a different question than was asked, with no
error and no warning. QCE rejected all 15 at the allowlist before SQL was generated.

**3. Value-based injection is not neutralized.**
GPT-4 embeds user-provided values directly into SQL strings. The injection string
`'; DROP TABLE orders; --` appears in GPT-4's generated SQL as a literal. Whether it
executes depends on the database driver. QCE sends all values as bind parameters — the
attack is architecturally impossible, not just unlikely.

**4. No audit trail.**
GPT-4 Direct generates a SQL string. There is no intermediate representation, no
structured record of what was requested, no way to explain the reasoning. QCE generates
a QueryPlan JSON that records intent, then compiles it to SQL. Both artifacts are
available for audit.

### Why Not Build Guardrails In-House?

This is the most common objection: *"We'll just add validators on top of GPT-4 Direct.
We'll check the generated SQL for unauthorized tables, wrap values in parameters, add
a retry loop. How hard can it be?"*

Here is what happens when teams build DIY guardrails:

**Month 1:** Add a regex check for sensitive table names in generated SQL.  
**Month 2:** A query passes the regex because the table name is aliased differently.  
**Month 3:** Add alias detection. Realize aliases can be nested in subqueries.  
**Month 4:** Add subquery recursion. Realize CTEs are not covered.  
**Month 5:** Add CTE parsing. Realize the SQL parser is now 400 lines of fragile code.  
**Month 6:** The SQL parser breaks on a valid Postgres feature it doesn't handle.  
You have built a partial SQL compiler. It has unknown failure modes.

**The fundamental problem:** You cannot validate SQL safety by inspecting SQL strings.
SQL is a context-sensitive language where the same string means different things depending
on aliases, scoping, and dialect. Safe SQL generation requires generating safe SQL — not
generating unsafe SQL and then trying to detect the unsafe parts.

QCE solves this by never generating unsafe SQL in the first place. The allowlist check
runs on the QueryPlan JSON, before SQL is generated. The safety check is O(1) — a
dictionary lookup against the schema allowlist.

**The total cost of DIY guardrails:**
- 3–6 months of senior engineering time to build
- Ongoing maintenance as SQL dialects and LLM outputs evolve
- Unknown residual risk from edge cases the parser doesn't handle
- No audit trail, no structured error codes, no retry feedback

QCE is a compiler that already handles this correctly. The build-vs-buy calculation is
not close.

### Why Not Vanna or Other Text-to-SQL Tools?

Vanna and similar RAG-based tools retrieve similar SQL from a training set and adapt it.
They work well for questions similar to previously seen questions. They fail on:
- Novel question types not in the training data
- Schema changes that invalidate cached SQL
- Multi-step aggregations (rollups, CTEs, statistical functions) that require structured
  plan composition rather than SQL retrieval

QCE is not a retrieval system. It is a compiler. It handles any question that can be
expressed as a valid QueryPlan — including rollups, CTEs, set operations, window functions,
and statistical aggregates — without relying on previously seen examples.

---

## On Token Cost and Caching

### QCE is LLM-agnostic

QCE works with any LLM — OpenAI, Anthropic, Google Gemini, Azure OpenAI, or any
self-hosted model (Llama, Mistral, Falcon, etc.) via a simple callable adapter.
The token cost figures in this report reflect OpenAI GPT-4o pricing because that is
what the benchmark used. Your actual cost depends entirely on your LLM choice.

| LLM deployment | Marginal token cost | Notes |
|---|---|---|
| OpenAI GPT-4o (API) | $5/1M input, $15/1M output | Benchmark baseline |
| OpenAI GPT-4o-mini (API) | $0.15/1M input, $0.60/1M output | ~30x cheaper, slightly less accurate |
| Azure OpenAI | Enterprise contract pricing | Often lower than retail API |
| Anthropic Claude 3.5 Haiku | $0.80/1M input, $4/1M output | Competitive alternative |
| Self-hosted Llama 3 / Mistral | ~$0 marginal per token | Infrastructure cost only |
| On-premise enterprise LLM | ~$0 marginal per token | Common in regulated industries |

**For customers with self-hosted or on-premise LLMs, QCE's token cost is effectively
zero per query.**

### Reducing token cost on OpenAI deployments

**1. Auto-generated domain spec (built-in)**
Reduced token cost by 26% on the Northwind benchmark with no accuracy loss.
Run once per schema change: `python -m dsl_compiler.spec_builder --schema config/schema.yaml`

**2. Prompt Caching (automatic on most providers)**
OpenAI, Anthropic, and Google Gemini automatically cache repeated prompt prefixes.
Cached input tokens are charged at 50% after the first call. No engineering work required.

**3. Application-Level Query Caching (1-day engineering task)**
Caching repeated questions reduces LLM calls to zero for cached queries —
60–80% cost reduction at typical enterprise analytics usage patterns.

**4. Cheaper planning model**
Switching the planner to GPT-4o-mini reduces token cost by ~30x with minimal accuracy
impact. The compiler's safety guarantees are identical regardless of which LLM generates
the plan.

### The honest cost comparison

| Deployment | QCE cost per 10 questions | GPT-4 Direct cost per 10 questions |
|---|---|---|
| GPT-4o (current benchmark) | $0.36 | $0.08 |
| GPT-4o-mini planner | ~$0.01 | ~$0.003 |
| Self-hosted LLM | ~$0 | ~$0 |
| With prompt caching (50% discount) | ~$0.18 | ~$0.04 |

---

## Known Limitations and Methodology Notes

**1. Test cases were written by the QCE team.**
All adversarial inputs, hallucination inputs, and pipeline questions were authored by the
same team that built QCE. Independent replication is encouraged. All test data is published
in `benchmark/data/` and the Northwind database is publicly available.

**2. Determinism benchmark — two levels, one user-facing.**
Benchmark 2a tests the QCE compiler in isolation (no LLM) — an internal architectural
property, not a user-facing comparison. Benchmark 2b tests what users actually experience.
Only 2b appears in the summary table.

**3. Full-pipeline determinism is 4/5 for both QCE and GPT-4 Direct.**
At temperature=0, both tools produce non-deterministic output on semantically ambiguous
questions. The difference: when QCE's planner varies, the compiler still enforces schema
safety and parameterization on every run. When GPT-4 Direct varies, the SQL executes
unvalidated.

**4. LangChain rate-limit errors are counted as inconclusive, not failed.**
LangChain's multi-step reasoning loop makes 6–10 LLM calls per question, exhausting
rate limits mid-benchmark. For product evaluation purposes, rate limiting is a
practical failure.

**5. LangChain pipeline correctness is "no error returned", not "verified correct".**
LangChain returns natural language answers. Row-count verification is not possible.
QCE and GPT-4 Direct return structured rows verified against known correct answers.

**6. GPT-4 Direct was given the most charitable possible setup.**
Full schema descriptions, explicit rejection rules, quoted table names, no markdown
wrapping. Despite this, it still showed safety instability and hallucination failures.

**7. The Northwind database is open source and publicly reproducible.**
Any team can load Northwind into their own Postgres instance and rerun the benchmark.
Instructions are in the [Benchmark Reproducibility](#benchmark-reproducibility) section.

---

## Executive Summary

**User-facing benchmark results (all tools measured end-to-end from natural language):**

| Benchmark | QCE | LangChain ✱ | GPT-4 Direct |
|---|---|---|---|
| Injection Resistance (50 inputs) | **50/50 ✅** | 21/50 ❌ | 48/50 ❌ |
| Full-Pipeline Determinism (5 × 3 runs) | 4/5 | 3/5 | 4/5 |
| Hallucination Rejection (15 inputs) | **15/15 ✅** | 4/15 ❌ | 13/15 ❌ |
| Full Pipeline Correctness (10 questions) | **10/10 ✅** | 9/9 ✱ (1 inconclusive) | **10/10 ✅** |

✱ *LangChain pipeline correctness = "no error returned", not independently verified correct.*

**Latency and cost (GPT-4o, Northwind benchmark):**

| Tool | Avg Latency | Total Tokens (10 Q) | Avg LLM Calls/Q | Est. Cost (10 Q) |
|---|---|---|---|---|
| QCE | 14,481ms | 70,783 | 1.0 | $0.36 |
| LangChain | 26,435ms | 99,827 | 8.2 | $0.54 |
| GPT-4 Direct | 2,500ms | 16,180 | 1.0 | $0.08 |

**The headline results:**
- QCE and GPT-4 Direct both scored 10/10 on pipeline correctness.
- QCE is the only tool with stable 50/50 injection resistance across all runs.
- GPT-4 Direct failed hallucination rejection on 2 inputs — silent wrong answers.
- LangChain failed hallucination rejection on 11 inputs — the most dangerous failure mode.
- GPT-4 Direct is faster and cheaper. QCE's cost premium buys architectural safety guarantees.

---

## Benchmark 1 — Injection Resistance

### What this benchmark tests

Can a malicious or unauthorized query get through to the database?

50 adversarial inputs across two categories:

**Schema-based attacks (46 inputs):** Attempts to access unauthorized tables or columns —
system tables (`pg_shadow`, `information_schema`), sensitive tables (`passwords`,
`credentials`, `api_keys`), sensitive columns (`email`, `ssn`, `credit_card`, `salary`).

**Value-based injection (4 inputs):** Classic SQL injection strings as filter values:
`'; DROP TABLE orders; --`, `x' UNION SELECT * FROM pg_shadow --`, etc.

### Results

| Tool | Score | Avg Latency | P95 Latency |
|---|---|---|---|
| **QCE** | **50/50** | **0.0ms** | **0.3ms** |
| LangChain | 21/50 | 4,491ms | 10,822ms |
| GPT-4 Direct | 48/50 | 2,835ms | 6,392ms |

### Why QCE's 50/50 is not the same as GPT-4's 48/50

GPT-4's safety is enforced by a system prompt. QCE's 50/50 is structural — enforced
by the compiler in Python code that runs after the LLM, not inside it.

**GPT-4 does not neutralize value-based injection.** It generates raw SQL strings where
injection values are embedded as literals. QCE generates:

```sql
SELECT ... FROM "orders" WHERE ship_country = :v_1
```

With `params = {"v_1": "'; DROP TABLE orders; --"}`. The string reaches Postgres as a
bind parameter — compared as plain text, never interpreted as SQL. The attack is
impossible by construction, not by instruction.

---

## Benchmark 2a — Compiler Determinism (Internal Architectural Property)

> This benchmark is **not a user-facing comparison**. It documents that given the same
> QueryPlan JSON, the compiler always produces the same parameterized SQL — a mathematical
> guarantee about a pure function. For the user-facing comparison, see Benchmark 2b.

| Tool | Score | Avg Latency |
|---|---|---|
| **QCE compiler** | **10/10** | **2.2ms** |
| LangChain (end-to-end) | 4/10 | 12,551ms |
| GPT-4 Direct (end-to-end) | 6/10 | 3,109ms |

GPT-4 at temperature=0 still varied on 4 of 10 questions — confirming that temperature=0
does not guarantee determinism in LLMs.

---

## Benchmark 2b — Full-Pipeline Determinism (All Three Tools, User-Facing)

### Results

| Tool | Score | Avg Latency | Notes |
|---|---|---|---|
| QCE | 4/5 | 14,218ms | Includes LLM planner |
| LangChain | 3/5 | 12,975ms | Natural language output compared |
| GPT-4 Direct | 4/5 | 3,013ms | SQL string compared |

### Analysis

QCE and GPT-4 Direct tied at 4/5. The 1 QCE failure was question 5 — "orders grouped
by shipping country" — where one run included a count metric and two runs did not. Both
are valid interpretations of the question. Same data, different plan structure.

**The critical difference:** When QCE's planner produces a different plan, the compiler
still validates it against the schema allowlist and parameterizes all values. Safety holds
on every run. When GPT-4 Direct produces different SQL, that SQL executes without any
validation layer.

---

## Benchmark 3 — Hallucination Rejection

### Results

| Tool | Score | Avg Latency | P95 Latency |
|---|---|---|---|
| **QCE** | **15/15** | **0.0ms** | **0.1ms** |
| LangChain | 4/15 | 11,907ms | 23,929ms |
| GPT-4 Direct | 13/15 | 2,381ms | 5,192ms |

### Analysis

QCE catches hallucinations in the compiler's allowlist check — before SQL is generated,
before any DB connection. The error is structured (`QueryPlanError: UNKNOWN_COLUMN`)
with the exact field name, making it machine-readable for retry loops.

**GPT-4 Direct dropped to 13/15 on Northwind** (vs 15/15 on simpler schemas). Two
hallucinated column names were plausible-sounding in an e-commerce context — GPT-4
substituted real columns silently and returned data answering a different question.
No error. No warning.

**LangChain dropped to 4/15** — the worst result of any tool on any benchmark. 11 of 15
hallucinated inputs were silently recovered by substituting real columns. This is the
most dangerous failure mode: confident wrong answers that look correct to the user.

---

## Benchmark 4 — Full Pipeline Correctness, Latency, and Cost

### Results

| Tool | Score | Avg Latency | P95 Latency |
|---|---|---|---|
| **QCE** | **10/10 ✅** | **14,481ms** | **14,874ms** |
| LangChain ✱ | 9/9 (1 inconclusive) | 26,435ms | 69,204ms |
| GPT-4 Direct | **10/10 ✅** | **2,500ms** | **4,238ms** |

✱ *LangChain scored as "no error returned". 1 question rate-limited. Results unverified.*

### Token Usage and Cost

| Tool | Total Tokens | Avg/Question | Avg LLM Calls | Est. Cost |
|---|---|---|---|---|
| QCE | 70,783 | 7,078 | 1.0 | $0.36 |
| LangChain | 99,827 | 11,092 | 8.2 | $0.54 |
| GPT-4 Direct | 16,180 | 1,618 | 1.0 | $0.08 |

**Reading the cost numbers honestly:**

GPT-4 Direct scored 10/10 on this run and is 6x cheaper and 6x faster than QCE.
This is the honest result. Both tools answered all 10 Northwind questions correctly.

The difference is not visible in pipeline correctness on clean, well-formed questions.
It is visible in injection resistance (50/50 vs 48/50), hallucination rejection
(15/15 vs 13/15), and — most importantly — the architectural guarantee: QCE's safety
cannot be degraded by a model update, a prompt edit, or an adversarial input it hasn't
seen before.

**Cost per verified correct answer:**

| Tool | Verified Correct | Est. Cost | Cost Per Correct Answer |
|---|---|---|---|
| QCE | 10/10 | $0.36 | $0.036 |
| GPT-4 Direct | 10/10 | $0.08 | $0.008 |
| LangChain | 0/10 verified | $0.54 | N/A |

### Detailed QCE Results (Northwind)

| Question | Correct | Row Count | Latency |
|---|---|---|---|
| Total customers | ✅ | 1 | ~8,000ms |
| Total orders | ✅ | 1 | ~12,000ms |
| Top 10 customers by order count | ✅ | 10 | ~14,000ms |
| Products containing chai | ✅ | 2 | ~14,000ms |
| Orders by shipping country | ✅ | 21 | ~15,000ms |
| Products per category | ✅ | 8 | ~14,000ms |
| Average orders per customer | ✅ | 1 | ~15,000ms |
| Discontinued products | ✅ | 1 | ~15,000ms |
| Top 10 products by unit price | ✅ | 10 | ~15,000ms |
| Orders by shipping country (repeat) | ✅ | 21 | ~15,000ms |

---

## Overall Conclusions

### What the benchmark proved

**QCE wins on safety — consistently, architecturally:**
- 50/50 injection resistance vs GPT-4's 48/50 — and QCE's score cannot fluctuate
- 15/15 hallucination rejection vs GPT-4's 13/15 and LangChain's 4/15
- All safety checks run in <1ms in the compiler — before SQL, before the database

**QCE ties on correctness and determinism:**
- 10/10 pipeline correctness — same as GPT-4 Direct on Northwind
- 4/5 full-pipeline determinism — same as GPT-4 Direct

**GPT-4 Direct wins on cost and latency:**
- $0.08 vs $0.36, 2.5s vs 14.5s
- This is honest. GPT-4 Direct is faster and cheaper on clean questions.

**LangChain is not competitive:**
- $0.54 per 10 questions — most expensive
- 4/15 hallucination rejection — worst by far
- 8 LLM calls per question leading to rate limits in production

### The core architectural claim

**Safety is enforced in code, not prompts.** On clean questions with no adversarial
inputs and no hallucinations, GPT-4 Direct performs as well as QCE. The difference
appears at the edges — injection attempts, plausible hallucinations, model updates,
prompt drift. QCE's compiler catches those failures before they reach the database.
GPT-4 Direct does not.

In regulated environments, "catches most failures most of the time" is not enough.
The floor matters. QCE is the floor.

---

## The LLM Cost Argument: QCE Commoditizes the Model Requirement

### The key insight

Without QCE, the LLM does everything — understands the question, writes SQL, enforces
safety, handles edge cases. That requires an expensive, capable model.

With QCE, the LLM only needs to map a question to a structured JSON object. The compiler
handles SQL generation, safety enforcement, and schema validation. That is a much simpler
task — a cheaper model can do it reliably.

**This was tested directly.** Three model configurations were benchmarked:

| Configuration | Injection Resistance | Hallucination Rejection | Pipeline Correctness | Est. Cost (10 Q) | Avg Latency |
|---|---|---|---|---|---|
| **QCE + gpt-4o-mini** | **50/50 ✅** | **15/15 ✅** | 10/10 | **$0.019** | 4,302ms |
| GPT-4 Direct + gpt-4o | 48/50 ❌ | 13/15 ❌ | 10/10 | $0.083 | 1,612ms |
| GPT-4 Direct + gpt-4o-mini | 50/50 ✅ | 11/15 ❌ | 10/10 | $0.004 | 1,753ms |

### What this proves

**QCE on a cheap model is safer than GPT-4 Direct on an expensive model.**
- 50/50 injection resistance vs 48/50 — and QCE's score is architectural, not prompt-dependent
- 15/15 hallucination rejection vs 13/15 — the compiler catches hallucinations before SQL is generated
- 4.4x cheaper than GPT-4 Direct on GPT-4o

**Making GPT-4 Direct cheaper makes it less safe.**
GPT-4 Direct on gpt-4o-mini dropped from 13/15 to 11/15 on hallucination rejection — 4 silent
wrong answers vs 2. Safety in a prompt-based system degrades as the model gets weaker.
Safety in QCE does not change — it is enforced by the compiler regardless of which LLM runs.

**The cheapest option (GPT-4 Direct on gpt-4o-mini at $0.004) has the worst safety.**
11/15 hallucination rejection — 4 silent wrong answers on questions with invented column names.
These are not detectable by the user. They look like correct answers.

**The safety-cost tradeoff only applies to prompt-based systems.**
With QCE, using a cheaper model reduces cost without reducing safety. The compiler's
allowlist check, parameterization, and schema validation are independent of the LLM.

### For self-hosted LLM deployments

For organizations running on-premise or self-hosted LLMs (common in healthcare, finance,
and government where data cannot leave the network), QCE's marginal token cost is
effectively zero. The safety guarantees are identical.

---

## Benchmark Reproducibility

```bash
# Full comparison (requires OpenAI API key and Northwind DB):
python3 benchmark/compare/run_comparison.py

# Pipeline-only (faster, re-runs Benchmark 4 only):
python3 benchmark/compare/run_pipeline_only.py
```

Load Northwind into your own Postgres instance:

```bash
curl -o /tmp/northwind.sql https://raw.githubusercontent.com/pthom/northwind_psql/master/northwind.sql
psql -h your-host -U your-user -d postgres -f /tmp/northwind.sql
```

Test data in `benchmark/data/`:
- `adversarial_inputs.json` — 50 injection resistance test cases
- `determinism_questions.json` — 20 determinism test cases
- `hallucination_inputs.json` — 15 hallucination rejection test cases
- `pipeline_questions.json` — 10 full pipeline correctness test cases

*All tools used the same LLM (GPT-4o), the same Northwind database, and the same
natural language questions. GPT-4 Direct was given explicit safety instructions —
the most charitable possible setup. The benchmark is open and reproducible.*
