# Migrating To GroundedQL

IntentQL was renamed to GroundedQL before its broader open-source launch because another
product already used the IntentQL name.

GroundedQL `0.3.0` is the first release under the new identity. The compiler architecture,
QueryPlan format, and schema format remain the same.

## Package And Imports

```bash
pip uninstall intentql
pip install groundedql
```

Update Python imports:

```python
# Before
from intentql import QueryAgent

# After
from groundedql import QueryAgent
```

## Command-Line Interface

Replace the `intentql` command with `groundedql`:

```bash
groundedql --help
groundedql init --db "postgresql://user:pass@host/db"
```

## Optional Memory Configuration

The optional memory feature now reads:

- `GROUNDEDQL_DISABLE_MEMORY`
- `GROUNDEDQL_MEMORY_MIN_SIMILARITY`
- `GROUNDEDQL_MEMORY_TASK_FILTER`

Its default local state directory changed from `~/.intentql/intent_memory` to
`~/.groundedql/intent_memory`. Move that directory manually if you want to retain existing
few-shot memory.

## Local Ollama Model

The default local model name changed from `intentql-gemma4` to `groundedql-gemma4`. Set
`OLLAMA_MODEL` explicitly if you want to continue using an existing model under its old
local name.
