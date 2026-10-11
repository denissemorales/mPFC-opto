# Change Log

This is a [changelog](https://keepachangelog.com/en/1.0.0/) designed to document
all notable changes to this project.

## [0.1.0] (Unreleased)

### Added

- Packaging: build backend, project metadata, and the full dependency list.
- `pytest` suite with a `--with-db` flag; database-backed tests are opt-in.
- `black` and `ruff` configuration, pre-commit hooks, and CI.
- `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, and mkdocs sources.
- `sleep.pss_utils` and `glm.basis_utils`, holding the computation that used to
  sit beside the table definitions, so it can be used without a database.
- Notebooks are paired with scripts under `notebooks/py_scripts/`.

### Changed

- `glm`, `behavior`, and `sleep` are now subpackages of `mpfc_opto` rather than
  three top-level import names.
- Notebooks import the installed package instead of manipulating `sys.path`.
- Module names no longer carry author initials: `pss_dmr` is `pss`,
  `sleep_table_dmr` is `sleep_table`, `AS_EM_module` is `em_module`, and so on.
  `GLMBasis` is `basis` and `FiltersEM` is `filters_em`. Class and table names
  are unchanged, so no database migration is needed.

### Fixed

- `em_module` reached `filters_em` by a bare import that only resolved when the
  working directory happened to be `notebooks/`.
