# Pipelines

Every analysis follows the DataJoint `Params` → `Selection` → `Computed`
pattern: a lookup table of hyperparameters, a manual table pairing those
parameters with input data, and a computed table that does the work.

## Sleep

| Module                | Tables                                                          |
| --------------------- | --------------------------------------------------------------- |
| `sleep.sleep_table`   | `SleepScoringParams` → `SleepScoringSelection` → `SleepScoring` |
| `sleep.updown_tables` | `UpDownStateParams` → `UpDownStateSelection` → `UpDownStates`   |
| `sleep.pss`           | `PSSParams` → `PSSSelection` → `SleepPSS`                       |

The spectral-slope computation itself lives in `sleep.pss_utils`, which has no
DataJoint dependency and so can be used without a database. `fit_pss_from_psd`
returns `(pss, intercept, slope)` where `pss == -slope`, so that a **larger**
`pss` means a steeper 1/f falloff.

## Behavior

| Module                      | Tables                                                       |
| --------------------------- | ------------------------------------------------------------ |
| `behavior.forktrack_tables` | `ForkTrackParams` → `ForkTrackSelection` → `ForkTrackEvents` |
| `behavior.wtrack_tables`    | `WTrackParams` → `WTrackSelection` → `WTrackEvents`          |

`behavior.filters_em` and `behavior.em_module` implement the Smith et al.
state-space EM estimate of a learning curve. They define no tables and need no
database.

## GLM

| Module                        | Tables                                              |
| ----------------------------- | --------------------------------------------------- |
| `glm.path_progression_tables` | `PathProgressSelection` → `PathProgress`            |
| `glm.glm_tables`              | `GLMSelection` → `GLMStorage`                       |
| `glm.basis`                   | `GLMBasisParams` → `GLMBasisSelection` → `GLMBasis` |

`GLMStorage` bins spiking, position, path progression, turns, and reward into 50
ms bins. `GLMBasis` then expands those covariates into `nemos` bases and stores
the design matrix.

The basis construction lives in `glm.basis_utils`, which like `pss_utils` has no
DataJoint dependency, though it does need `jax` and `nemos`.
