# QCE — Query Compiler Engine
## Why Your Natural Language Analytics Tool Fails in Production (and How We Fix It)

---

## The Problem Every Analytics Team Hits

Building a natural language query interface looks easy at first.

You connect GPT-4 to your database, ask it a question, and it writes SQL. It works.
You demo it to your team. Everyone is excited.

Then you ship it.

**Week 4:** A report shows the wrong number. No error. No warning. Just a wrong answer
that someone made a decision based on.

**Week 8:** Two analysts ask the same question on different days and get different results.
Your CFO wants to know why.

**Week 12:** A user asks something unexpected and the system queries a table it shouldn't
have. You find out because a downstream alert fires.

**Week 16:** Your engineering team has spent three months building regex validators,
SQL parsers, prompt guardrails, and retry loops. They are essentially writing a compiler
from scratch — a bad one, with unknown failure modes.

This is not a GPT-4 problem. It is an architectural problem.
**You cannot make a probabilistic system safe by prompting it more carefully.**

---

## What QCE Does

QCE is a compiler layer that sits between your LLM and your database.

The LLM's job is to understand your user's question and express it as a structured
**QueryPlan JSON** — a simple, readable description of what data is needed.

QCE's job is to compile that plan into safe, parameterized SQL — enforcing your schema,
your access rules, and your data policies in code, not in a prompt.

```
User question
     │
     ▼
  LLM planner  ──→  QueryPlan JSON  ──→  QCE Compiler  ──→  Parameterized SQL  ──→  Database
                     (structured)         (your rules)        (always safe)
```

**When safety is enforced in code, it cannot be forgotten, bypassed, or degraded by a
model update.**

QCE works with any LLM — OpenAI, Anthropic, Google Gemini, Azure OpenAI, or any
self-hosted model. For organizations running internal LLMs for data privacy reasons,
QCE's marginal token cost is effectively zero.

---

## What the Benchmark Showed

Head-to-head against GPT-4 Direct and LangChain SQLAgent on the Northwind database
(open source, publicly reproducible), same LLM (GPT-4o), same questions.

### Safety — the metric that cannot fluctuate

| Tool | Injection Resistance | Hallucination Rejection | Stable Across Runs? |
|---|---|---|---|
| **QCE** | **50/50** | **15/15** | **Yes — enforced in code** |
| GPT-4 Direct | 48/50 | 13/15 | No — prompt-dependent |
| LangChain | 21/50 | 4/15 | No |

GPT-4's safety scores are not fixed — they change between benchmark runs because safety
is enforced by a prompt, not by code. On hallucination rejection, GPT-4 silently
substituted real columns for 2 hallucinated inputs and returned data answering the
wrong question. No error. No warning.

LangChain silently recovered from 11 of 15 hallucinated inputs — returning confident
wrong answers. That is the most dangerous failure mode.

QCE rejected all injection attempts and all hallucinations in the compiler — before
SQL was generated, before the database was touched. It ran in under 1ms.

### Correctness — what your users actually see

| Tool | Correct Answers | Verified? |
|---|---|---|
| **QCE** | **10/10** | **Yes — row counts verified** |
| GPT-4 Direct | 10/10 | Yes — row counts verified |
| LangChain | 9/9 completed | No — natural language output only |

Both QCE and GPT-4 Direct scored 10/10 on the Northwind pipeline benchmark. This is
the honest result — on clean, well-formed questions, both tools work correctly.

The difference appears at the edges: injection attempts, plausible hallucinations,
model updates, and adversarial inputs. QCE's compiler catches those before they reach
the database. GPT-4 Direct does not.

### Cost and Latency — the honest numbers

| Tool | Avg Latency | Est. Cost (10 questions) |
|---|---|---|
| GPT-4 Direct | **2,500ms** | **$0.08** |
| QCE | 14,481ms | $0.36 (GPT-4o) / ~$0 (self-hosted LLM) |
| LangChain | 26,435ms | $0.54 — results unverified |

GPT-4 Direct is 6x faster and 4.5x cheaper on OpenAI deployments. We are honest
about that. The cost premium buys architectural safety guarantees that cannot be
replicated by prompting.

For organizations running self-hosted or on-premise LLMs — common in regulated
industries where data cannot leave the network — QCE's marginal token cost is zero.
LangChain is the most expensive tool at $0.54 per 10 questions with the worst safety
scores of any tool tested.

---

## QCE Commoditizes the LLM Requirement

This is the argument that changes the cost conversation completely.

Without QCE, the LLM must write safe, correct, parameterized SQL. That requires
a powerful, expensive model. Even then it fails — as the benchmark shows.

With QCE, the LLM only needs to map a question to a structured JSON object.
The compiler handles everything else. That is a much simpler task.

**Benchmarked directly — three model configurations, same questions:**

| Configuration | Safety (Hallucination) | Correctness | Cost (10 Q) |
|---|---|---|---|
| **QCE + gpt-4o-mini** | **15/15 ✅** | 10/10 | **$0.019** |
| GPT-4 Direct + gpt-4o | 13/15 ❌ | 10/10 | $0.083 |
| GPT-4 Direct + gpt-4o-mini | 11/15 ❌ | 10/10 | $0.004 |

**QCE on a cheap model is safer than GPT-4 Direct on an expensive model — and 4x cheaper.**

Making GPT-4 Direct cheaper makes it less safe. GPT-4 Direct on gpt-4o-mini dropped
from 13/15 to 11/15 on hallucination rejection. Safety in a prompt-based system
degrades as the model gets weaker. Safety in QCE does not change — it is enforced
by the compiler, not the model.

For organizations running self-hosted or on-premise LLMs, QCE's marginal cost is
effectively zero — with the same architectural safety guarantees as any other deployment.

---

## Who This Is For

**You operate in a regulated environment.**
Healthcare, finance, government, insurance. Wrong answers and unauthorized data access
are liability, not bugs. QCE's schema allowlist is a documented, auditable policy —
not a prompt that can drift between model updates. Every query produces a structured
record of what was requested, what SQL ran, and what was returned. That is a compliance
artifact you can show to an auditor.

**You have non-technical users querying production data.**
When a CFO gets different numbers on the same question on different days, the problem
is not the LLM — it is the absence of a compiler enforcing consistent behavior.
QCE's compiler is a pure function: same plan, same SQL, always.

**You are building a multi-tenant analytics product.**
User A must never see User B's data. GPT-4 has no tenant isolation layer. QCE's schema
allowlist is scoped per deployment — different tenants get different schemas, enforced
in code, not in a system prompt that every engineer can accidentally edit.

**Your team has already tried LangChain.**
8 LLM calls per question. Rate limits on production workloads. 4/15 hallucination
rejection. $0.54 per 10 questions with unverified correctness. The benchmark confirmed
what most teams already discovered the hard way.

**You tried GPT-4 Direct and it worked in your demo.**
It works in the demo. The failures happen in production — on injection attempts you
didn't expect, on hallucinated columns that look plausible, on model updates that
change behavior you relied on. QCE's compiler is the floor that catches those failures
before they reach your users or your auditors.

---

## Why Not Build Guardrails Yourself?

The most common objection: *"We'll add validators on top of GPT-4. How hard can it be?"*

Month 1: regex check for sensitive table names.
Month 2: the table is aliased — regex misses it.
Month 3: add alias detection. Aliases can be nested in subqueries.
Month 4: add subquery recursion. CTEs are not covered.
Month 5: add CTE parsing. The SQL parser is now 400 lines of fragile code.
Month 6: the parser breaks on a valid Postgres feature. Unknown failure mode.

**You cannot validate SQL safety by inspecting SQL strings.** SQL is context-sensitive —
the same identifier means different things depending on aliases, scoping, and dialect.

QCE solves this by never generating unsafe SQL in the first place. The allowlist check
runs on the QueryPlan JSON before SQL is generated. It is an O(1) dictionary lookup —
it cannot be aliased, scoped around, or broken by a new Postgres feature.

The build cost: 3–6 months of senior engineering time, ongoing maintenance, unknown
residual risk, no audit trail, no structured error codes. QCE already exists.

---

## The Honest Limitations

**QCE costs more per query on OpenAI deployments.**
GPT-4 Direct at $0.08 vs QCE at $0.36 for 10 questions. This gap closes with
domain-scoped spec generation, prompt caching, and a cheaper planning model.
For self-hosted LLMs, the gap is zero.

**QCE is slower than GPT-4 Direct.**
14.5s vs 2.5s average latency on GPT-4o. Both tools make one LLM call per question —
the latency difference is GPT-4o's response time with a larger context, not additional
processing steps. Switching to GPT-4o-mini brings QCE's latency to 2–4 seconds,
comparable to GPT-4 Direct on GPT-4o. For self-hosted LLMs, latency depends entirely
on your infrastructure.

**Full-pipeline determinism is 4/5, same as GPT-4 Direct.**
Both tools have LLM-driven variation on ambiguous questions. The difference: QCE's
varying output is always schema-safe and parameterized. GPT-4's varying SQL executes
without any validation. Plan canonicalization is on our roadmap.

**This benchmark used one database domain.**
The Northwind database is a standard, publicly reproducible e-commerce schema.
The architectural guarantees are domain-independent — the benchmark evidence is
domain-specific. We encourage independent replication on any Postgres database.

---

## The One-Sentence Version

> GPT-4 Direct is not selling you a seatbelt. QCE is.

On clean questions, both tools work. The difference is what happens at the edges —
injection attempts, hallucinated columns, model updates, adversarial inputs.
QCE's compiler catches those before they reach the database.
GPT-4 Direct does not.

In regulated environments, that floor is not optional.

---

## Next Steps

**If you are a potential customer:**
We are onboarding early design partners in compliance-heavy analytics environments —
healthcare operations, financial reporting, government procurement, and internal
analytics platforms where wrong answers or unauthorized data access carry real
consequences.

Early partners get priority onboarding, direct access to the founding team, and
pricing that reflects the early stage. If your natural language query project keeps
failing in production — or a compliance requirement makes raw LLM-generated SQL
unacceptable — we want to talk.

**If you are an investor:**
The technology is built and benchmarked against real alternatives on a publicly
reproducible database. The core thesis — safety enforced in code, not prompts —
is validated. The next step is distribution: finding the first three customers in
regulated analytics who cannot ship without this guarantee.

We are raising a pre-seed round to get there.

**Contact:** [your email or contact page]
