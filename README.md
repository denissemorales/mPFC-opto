# mPFC-opto

Preliminary analysis of mPFC-opto experiments in the Frank Lab at UCSF.

Built on [Spyglass](https://github.com/LorenFrankLab/spyglass): each analysis is
a DataJoint pipeline following the `Params` → `Selection` → `Computed` pattern.

## Install

```bash
pip install -e .            # analysis only
pip install -e ".[test]"    # and the test suite
```

## Project structure

```
mPFC-opto/
├── .github/                 # CI, issue and PR templates, CONTRIBUTING.md
├── .pre-commit-config.yaml  # black + ruff hooks
├── docs/                    # mkdocs sources
├── environment.yml          # conda environment
├── notebooks/               # analysis notebooks
├── pyproject.toml           # packaging, dependencies, tool config
├── src/mpfc_opto/
│   ├── glm/                 # encoding models and path progression
│   ├── behavior/            # fork-track and W-track behavior, EM filters
│   └── sleep/               # sleep scoring, up/down states, spectral slope
└── tests/
```

## Pipelines

| Module                        | Tables                                                          |
| ----------------------------- | --------------------------------------------------------------- |
| `sleep.sleep_table`           | `SleepScoringParams` → `SleepScoringSelection` → `SleepScoring` |
| `sleep.updown_tables`         | `UpDownStateParams` → `UpDownStateSelection` → `UpDownStates`   |
| `sleep.pss`                   | `PSSParams` → `PSSSelection` → `SleepPSS`                       |
| `behavior.forktrack_tables`   | `ForkTrackParams` → `ForkTrackSelection` → `ForkTrackEvents`    |
| `behavior.wtrack_tables`      | `WTrackParams` → `WTrackSelection` → `WTrackEvents`             |
| `glm.path_progression_tables` | `PathProgressSelection` → `PathProgress`                        |
| `glm.glm_tables`              | `GLMSelection` → `GLMStorage`                                   |
| `glm.basis`                   | `GLMBasisParams` → `GLMBasisSelection` → `GLMBasis`             |

## Tests

```bash
pytest              # pure-function tests; no database, no Docker
pytest --with-db    # adds tests that need a MySQL server
```

See [CONTRIBUTING.md](.github/CONTRIBUTING.md).
