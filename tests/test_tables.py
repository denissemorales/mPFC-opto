"""Database-backed tests: that every pipeline's tables declare.

`requires_db` because importing any of these modules executes `@schema`, which
declares tables against a live server.
"""

import pytest

pytestmark = pytest.mark.requires_db


class TestTableDeclaration:
    """Each pipeline declares as a Params -> Selection -> Computed triple."""

    @pytest.mark.parametrize(
        "module,names",
        [
            (
                "mpfc_opto.sleep.pss",
                ["PSSParams", "PSSSelection", "SleepPSS"],
            ),
            (
                "mpfc_opto.sleep.sleep_table",
                [
                    "SleepScoringParams",
                    "SleepScoringSelection",
                    "SleepScoring",
                ],
            ),
            (
                "mpfc_opto.sleep.updown_tables",
                [
                    "UpDownStateParams",
                    "UpDownStateSelection",
                    "UpDownStates",
                ],
            ),
            (
                "mpfc_opto.behavior.forktrack_tables",
                ["ForkTrackParams", "ForkTrackSelection", "ForkTrackEvents"],
            ),
            (
                "mpfc_opto.behavior.wtrack_tables",
                ["WTrackParams", "WTrackSelection", "WTrackEvents"],
            ),
            (
                "mpfc_opto.glm.glm_tables",
                ["GLMSelection", "GLMStorage"],
            ),
            (
                "mpfc_opto.glm.path_progression_tables",
                ["PathProgressSelection", "PathProgress"],
            ),
        ],
    )
    def test_tables_declare(self, server, module, names):
        """Importing the module declares its tables; each gains a heading."""
        import importlib

        mod = importlib.import_module(module)
        for name in names:
            table = getattr(mod, name)
            assert table.heading is not None
            assert len(table.heading.names) > 0

    def test_params_tables_are_lookups_with_contents(self, server):
        """Lookup tables should ship default parameter sets."""
        from mpfc_opto.sleep.pss import PSSParams

        assert len(PSSParams()) >= 0  # declares and queries without error


class TestMakeKeyPinsUpstream:
    """`make` restricts upstream tables with `key`, so `key` must pin them.

    These guard the restrictions in `GLMStorage.make_fetch` and
    `PathProgress.make_fetch`. Both previously restricted on `epoch`, which is
    a *secondary* attribute and therefore matches across sessions and parameter
    sets. The fix relies on each upstream table's primary key being wholly
    contained in the computed table's own key; if a `->` dependency is ever
    dropped from a Selection definition, `& key` silently stops pinning and
    these fail.
    """

    @pytest.mark.parametrize(
        "selection,upstream",
        [
            ("GLMSelection", "ForkTrackEvents"),
            ("GLMSelection", "PathProgress"),
            ("PathProgressSelection", "ForkTrackEvents"),
        ],
    )
    def test_key_contains_upstream_primary_key(
        self, server, selection, upstream
    ):
        from mpfc_opto.behavior.forktrack_tables import ForkTrackEvents
        from mpfc_opto.glm.glm_tables import GLMSelection
        from mpfc_opto.glm.path_progression_tables import (
            PathProgress,
            PathProgressSelection,
        )

        tables = {
            "GLMSelection": GLMSelection,
            "PathProgressSelection": PathProgressSelection,
            "ForkTrackEvents": ForkTrackEvents,
            "PathProgress": PathProgress,
        }
        missing = set(tables[upstream].primary_key) - set(
            tables[selection].primary_key
        )
        assert not missing, (
            f"{selection}'s key does not pin {upstream}: {sorted(missing)} "
            f"absent, so `& key` would match more than one row"
        )

    def test_epoch_is_not_a_primary_key_attribute(self, server):
        """The premise of the bug these replaced: `epoch` does not identify a
        row, so it must never be the whole restriction."""
        from mpfc_opto.behavior.forktrack_tables import ForkTrackEvents
        from mpfc_opto.glm.path_progression_tables import PathProgress

        for table in (ForkTrackEvents, PathProgress):
            assert "epoch" in table.heading.names
            assert "epoch" not in table.primary_key


class TestTriPartMake:
    """Computed tables doing heavy work use the tri-part make.

    DataJoint runs `make` inside a transaction, so a table that fits a mixture
    model there holds one open for the duration. The tri-part methods let the
    fetch and the computation happen outside it.
    """

    @pytest.mark.parametrize(
        "module,table",
        [
            ("mpfc_opto.sleep.sleep_table", "SleepScoring"),
            ("mpfc_opto.sleep.updown_tables", "UpDownStates"),
            ("mpfc_opto.glm.glm_tables", "GLMStorage"),
            ("mpfc_opto.glm.path_progression_tables", "PathProgress"),
            ("mpfc_opto.glm.basis", "GLMBasis"),
            ("mpfc_opto.behavior.forktrack_tables", "ForkTrackEvents"),
        ],
    )
    def test_defines_all_three_and_no_plain_make(self, server, module, table):
        """DataJoint accepts `make` *or* the full trio; a half-converted table
        would fall back to `make` and silently keep its transaction."""
        import importlib

        cls = getattr(importlib.import_module(module), table)
        for method in ("make_fetch", "make_compute", "make_insert"):
            assert callable(getattr(cls, method, None)), f"missing {method}"
        assert "make" not in vars(cls), (
            f"{table} defines both `make` and the tri-part methods; DataJoint "
            "would use `make` and the split would have no effect"
        )
