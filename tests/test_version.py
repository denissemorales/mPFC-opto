"""Check that the installed distribution agrees with `pyproject.toml`."""

from importlib.metadata import version
from pathlib import Path

import pytest

try:  # stdlib from 3.11; this project supports 3.10, where it is not
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on interpreter
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="session")
def declared_version():
    """Version as written in `pyproject.toml`."""
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


@pytest.mark.unit
def test_installed_version_matches_pyproject(declared_version):
    """A stale editable install is the usual cause of a mismatch here."""
    assert version("mPFC-opto") == declared_version


@pytest.mark.unit
def test_dunder_version_matches_pyproject(declared_version):
    import mpfc_opto

    assert mpfc_opto.__version__ == declared_version


@pytest.mark.unit
def test_package_imports():
    """Smoke test: the package and its three subpackages are importable."""
    import importlib

    for name in (
        "mpfc_opto",
        "mpfc_opto.glm",
        "mpfc_opto.behavior",
        "mpfc_opto.sleep",
    ):
        assert importlib.import_module(name) is not None


@pytest.mark.unit
def test_importing_the_package_needs_no_database():
    """`import mpfc_opto` must not reach a table module. If it ever
    re-exports one, the `@schema` decorators run and this blocks on a
    credential prompt instead of failing."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mpfc_opto; print(mpfc_opto.__version__)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "datajoint" not in result.stderr.lower()
