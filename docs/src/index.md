# mPFC-opto

Preliminary analysis of mPFC-opto experiments in the Frank Lab at UCSF.

Built on [Spyglass](https://github.com/LorenFrankLab/spyglass). Each analysis is
a DataJoint pipeline, so results are stored in and recomputed from the database
rather than kept in notebook state.

## Install

```bash
pip install -e ".[test]"
```

The `glm` subpackage additionally needs `nemos` and `neurospatial`, which pull
in `jax`. `jax` ships a compiled extension tied to a NumPy ABI, and Spyglass
pins `numpy<2` — so a `jax` built for NumPy 2 will fail to import in a Spyglass
environment. Install a `jax` built against NumPy 1 if `import jax` raises
`ImportError: numpy._core.multiarray failed to import`.

## Tests

```bash
pytest              # pure-function tests; no database, no Docker
pytest --with-db    # adds tests that need a MySQL server
```

`--with-db` starts a `datajoint/mysql` container, waits for it, and removes it
afterwards. Tests needing a server are marked `requires_db` and skipped without
the flag.

## Layout

```
src/mpfc_opto/
├── glm/        # encoding models, basis construction, path progression
├── behavior/   # fork-track and W-track behavior, state-space EM filters
└── sleep/      # sleep scoring, up/down states, spectral slope (PSS)
```

See [pipelines.md](pipelines.md) for the tables each module defines.
