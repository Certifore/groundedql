# Contributing To GroundedQL

Thank you for contributing to GroundedQL. Contributions are proposed through pull requests
and become part of the official project only after maintainer review and merge.

## Before Opening A Pull Request

1. Open or review an issue for substantial changes.
2. Keep changes focused and consistent with the existing architecture.
3. Add tests that describe the general capability being changed.
4. Run the relevant checks locally.

```bash
pip install -e ".[dev]"
ruff check groundedql/
python test/test_main.py lint
python test/test_generic_planner_resolution.py
```

## Generality Requirement

GroundedQL uses benchmarks to reveal missing general capabilities. Core logic must not branch
on benchmark case IDs, benchmark domains, known questions, entity names, or expected
answers.

A benchmark-driven change should:

- explain the reusable semantic capability;
- include domain-neutral regression coverage;
- preserve previously correct behavior;
- document ambiguity that cannot be resolved safely.

## Pull Request Review

The lead maintainer reviews and decides whether to merge pull requests into the official
repository. Opening a pull request does not grant merge, maintainer, or release authority.

Review may request changes for correctness, scope, maintainability, security, test coverage,
or alignment with project direction.

## Releases

Contributors should not publish packages or releases using the official GroundedQL project
identity. Official GitHub releases, tags, and PyPI publications are handled only by the lead
maintainer. See [GOVERNANCE.md](GOVERNANCE.md).

## License

By submitting a contribution for inclusion, you agree that it may be distributed under the
project's Apache License 2.0, consistent with the contribution terms in that license.
