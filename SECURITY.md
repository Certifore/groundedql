# Security Policy

## Reporting A Vulnerability

Please do not open a public issue for a suspected security vulnerability.

Use GitHub's private vulnerability reporting feature for the official GroundedQL repository.
Include:

- the affected version or commit;
- reproduction steps or a proof of concept;
- the potential impact;
- any suggested mitigation.

The lead maintainer will review the report, coordinate a fix where appropriate, and decide
when disclosure is safe.

## Scope

Security reports may include vulnerabilities in schema allowlist enforcement, SQL
parameterization, validation, execution boundaries, dependency handling, or official
release artifacts.

GroundedQL cannot secure database credentials, permissions, application code, or SQL executed
outside GroundedQL. Deployments should use least-privilege database roles and independent
operational controls.
