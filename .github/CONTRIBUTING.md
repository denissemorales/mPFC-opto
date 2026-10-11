# Contributing

## Setup

```bash
pip install -e ".[test]"
pre-commit install
```

## Tests

```bash
pytest                # pure-function tests; no database, no Docker
pytest --with-db      # adds the tests that need a MySQL server
```

`--with-db` starts a `datajoint/mysql` container, waits for it to accept
connections, and removes it when the session ends. Pass `--no-teardown` to keep
it for the next run.

Tests that need a server are marked `requires_db` and skipped without the flag,
so a plain `pytest` stays fast and dependency-light.

### Writing tests

Import the module under test **inside a fixture**, never at module scope.
Importing a table module runs its `@schema` decorators, which open a DataJoint
connection — at collection time that happens before any fixture has configured
credentials, so the suite would connect to whatever `dj.config` holds. The
fixtures in `tests/conftest.py` show the pattern.

## Style

`black` and `ruff` at 80 columns, enforced by pre-commit and CI. NumPy-style
docstrings.

## Notebooks

Notebooks are paired with scripts under `notebooks/py_scripts/` via `jupytext`.
Edit either side and let the pre-commit hook sync the other; commit both.
