"""Shared fixtures and the `--with-db` opt-in.

Database-backed tests are off by default: a bare `pytest` run needs no Docker
and no MySQL client, so the suite is usable on a laptop and in a container-less
CI job. Pass `--with-db` to start a server and run the rest.

Modules under test are imported *inside* fixtures, never at module scope.
Importing a table module executes its `@schema` decorators, which opens a
DataJoint connection; at collection time that would happen before any fixture
has configured credentials, so the suite would try to connect to whatever
`dj.config` happened to hold.
"""

import os
from pathlib import Path

import pytest

SERVER = None

BASE_DIR = Path(__file__).resolve().parent / "_data"

# Spyglass normally applies these when its own config loads. Under --with-db
# there is no config file, so declaring any spyglass-derived table fails on
# filepath support unless they are set here, before the first spyglass import.
SPYGLASS_ENV = {
    "DJ_SUPPORT_FILEPATH_MANAGEMENT": "TRUE",
    "KACHERY_CLOUD_EPHEMERAL": "TRUE",
    "HDF5_USE_FILE_LOCKING": "FALSE",
}


def pytest_addoption(parser):
    """Add database options.

    Parameters
    ----------
    --with-db (bool): Default False. Start MySQL and run `requires_db` tests.
    --no-teardown (bool): Default False. Leave the container running on exit.
    --container-name (str): Default 'mpfc-opto-pytest'.
    --container-port (int): Default 3307.
    """
    parser.addoption(
        "--with-db",
        action="store_true",
        dest="with_db",
        default=False,
        help="Start a MySQL container and run database-backed tests.",
    )
    parser.addoption(
        "--no-teardown",
        action="store_true",
        dest="no_teardown",
        default=False,
        help="Leave the container running after the session, for reuse.",
    )
    parser.addoption(
        "--container-name",
        action="store",
        dest="container_name",
        default="mpfc-opto-pytest",
        help="Docker container name, so concurrent runs need not collide.",
    )
    parser.addoption(
        "--container-port",
        action="store",
        dest="container_port",
        default=3307,
        type=int,
        help="Host port mapped to MySQL's 3306.",
    )


def pytest_configure(config):
    """Start the server, but only when asked for."""
    global SERVER
    if not config.option.with_db:
        return

    import datajoint as dj

    from .container import DockerMySQLManager

    os.environ.update(SPYGLASS_ENV)
    # Never honor an ambient SPYGLASS_BASE_DIR: pointed at real storage, a
    # destructive test would run against production analysis files.
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["SPYGLASS_BASE_DIR"] = str(BASE_DIR)

    SERVER = DockerMySQLManager(
        container_name=config.option.container_name,
        port=config.option.container_port,
    )
    SERVER.start()
    SERVER.wait()

    custom = dj.config.setdefault("custom", {})
    custom["spyglass_dirs"] = {"base": str(BASE_DIR)}
    # Spyglass checks that an Analysis table's schema prefix matches this, and
    # derives the schema itself from the connected user -- so they must agree.
    custom["database.prefix"] = SERVER.user


def pytest_unconfigure(config):
    if SERVER is not None:
        SERVER.stop(remove=not config.option.no_teardown)


def pytest_collection_modifyitems(config, items):
    """Skip `requires_db` tests unless `--with-db` was passed."""
    if config.option.with_db:
        return
    skip = pytest.mark.skip(reason="needs a database; pass --with-db")
    for item in items:
        if "requires_db" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def server():
    """The running container, or skip if the suite was started without one."""
    if SERVER is None:
        pytest.skip("no database; pass --with-db")
    return SERVER


# --- Modules under test -------------------------------------------------
# Imported lazily, for the reason given in this module's docstring.


@pytest.fixture(scope="session")
def filters_em():
    """`filters_em`: pure numpy, no DataJoint, so it needs no database."""
    from mpfc_opto.behavior import filters_em

    return filters_em


@pytest.fixture(scope="session")
def em_module():
    """`em_module`: matplotlib and filters_em only, still no database."""
    from mpfc_opto.behavior import em_module

    return em_module


@pytest.fixture(scope="session")
def basis_utils():
    """`basis_utils`: no DataJoint, but still needs jax.

    jax ships a compiled extension tied to a NumPy ABI, so an environment
    pinned to numpy<2 for spyglass can hold a jax built for NumPy 2 and leave
    it unimportable. exc_type is required because that surfaces as an
    ImportError, which importorskip treats as an error rather than a skip.
    """
    pytest.importorskip(
        "jax",
        reason="jax unimportable (NumPy ABI mismatch?)",
        exc_type=ImportError,
    )
    pytest.importorskip("nemos", exc_type=ImportError)
    return pytest.importorskip(
        "mpfc_opto.glm.basis_utils", exc_type=ImportError
    )


@pytest.fixture(scope="session")
def glm_basis(server):
    """`glm.basis`, or skip.

    Declares tables, so importing it needs a database. Also needs `nemos`/`jax`,
    by way of `basis_utils`.
    """
    # exc_type is required: a NumPy ABI mismatch surfaces as ImportError, and
    # importorskip treats that as an error rather than a skip by default.
    pytest.importorskip(
        "jax",
        reason="jax unimportable (NumPy ABI mismatch?)",
        exc_type=ImportError,
    )
    pytest.importorskip("nemos", exc_type=ImportError)
    return pytest.importorskip("mpfc_opto.glm.basis", exc_type=ImportError)


@pytest.fixture(scope="session")
def pss_utils():
    """`pss_utils`: numpy and scipy only, so no database."""
    from mpfc_opto.sleep import pss_utils

    return pss_utils


@pytest.fixture(scope="session")
def sleep_pss(server):
    """`pss`. Declares tables, so importing it needs a database."""
    from mpfc_opto.sleep import pss

    return pss
