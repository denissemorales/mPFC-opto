"""Analysis pipelines for mPFC-opto experiments.

Nothing is re-exported here on purpose. Importing a table module executes its
`@schema` decorators, which opens a DataJoint connection, so `import mpfc_opto`
must not reach them. Import the module you want:

    from mpfc_opto.sleep.sleep_table import SleepScoring     # needs a database
    from mpfc_opto.sleep.pss_utils import fit_pss_from_psd   # does not

The `*_utils` and `poke_validation` modules hold the computation that needs no
database, and are the parts worth importing directly.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mPFC-opto")
except PackageNotFoundError:  # pragma: no cover - source tree, not installed
    __version__ = "unknown"

__all__ = ["__version__"]
