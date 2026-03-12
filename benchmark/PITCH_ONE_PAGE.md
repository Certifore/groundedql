# QCE — Query Compiler Engine
### Safe, auditable natural language queries for any Postgres database.

---

## The problem

Your team built a natural language query interface on top of GPT-4.
It worked in the demo. Then you shipped it.

- A report returned the wrong number. No error. No warning.
- The same question gave different results on different days.
- A user asked something unexpected and it queried a table it shouldn't have.

This is not a GPT-4 problem. It is an architectural problem.
**You cannot make a probabilistic system safe by prompting it more carefully.**

---

## What QCE does

QCE is a compiler layer that sits between your LLM and your database.

```
User question → LLM → QueryPlan JSON → QCE Compiler → Parameterized SQL → Database
```

The LLM maps the question to a structured JSON object.
The compiler enforces your schema, your access rules, and your data policies — in code.

**Safety is structural. It cannot be forgotten, bypassed, or degraded by a model update.**

---

## What the benchmark proved

Tested head-to-head against GPT-4 Direct and LangChain on an open, reproducible database.

| | QCE | GPT-4 Direct | LangChain |
|---|---|---|---|
| Unauthorized schema access blocked | **50/50** | 48/50 | 21/50 |
| Hallucinated columns rejected | **15/15** | 13/15 | 4/15 |
| Correct answers on real questions | **10/10** | 10/10 | unverified |

GPT-4 Direct answered the same questions correctly — on clean inputs.
The difference appears on injection attempts, hallucinated columns, and model updates.
QCE catches those in the compiler. GPT-4 Direct does not.

---

## You don't need an expensive model

QCE on a cheap model (gpt-4o-mini) outperforms GPT-4 Direct on an expensive model on every safety metric — and costs 4x less.

| | QCE + gpt-4o-mini | GPT-4 Direct + gpt-4o |
|---|---|---|
| Hallucinated columns rejected | **15/15** | 13/15 |
| Unauthorized access blocked | **50/50** | 48/50 |
| Cost per 10 questions | **$0.019** | $0.083 |

Making GPT-4 Direct cheaper makes it less safe.
Making QCE cheaper keeps it exactly as safe — safety is in the compiler, not the model.

---

## Who this is for

- **Regulated environments** — healthcare, finance, government. Wrong answers and unauthorized data access are liability. QCE's allowlist is a documented, auditable policy you can show to an auditor.
- **Non-technical users on production data** — CFOs and directors who need consistent numbers, not "the model answered differently on Tuesday."
- **Multi-tenant analytics products** — tenant isolation enforced in code, not a system prompt every engineer can accidentally edit.
- **Teams already burned by LangChain** — 8 LLM calls per question, rate limits, 4/15 hallucination rejection, unverified answers.

---

## Works with any LLM, any Postgres database

OpenAI · Anthropic · Google Gemini · Azure OpenAI · self-hosted Llama / Mistral.
For organizations running on-premise LLMs, marginal cost is zero.
Any Postgres schema — define your allowlist once in a YAML file, QCE enforces it on every query.

---

## Talk to us

We are onboarding early design partners in compliance-heavy analytics environments.
Early partners get direct access to the founding team and pilot pricing.

**If your natural language query project keeps failing in production — or a compliance
requirement makes raw LLM-generated SQL unacceptable — we want to talk.**

**Contact:** [your email or contact page]
