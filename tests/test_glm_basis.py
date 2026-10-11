"""Database-backed tests for the `GLMBasis` tables.

The basis-construction functions this module used to also cover now live in
`basis_utils`, and are tested without a database in `test_basis_utils.py`. What
is left here needs a server, because declaring these tables declares them.
"""

import datajoint as dj
import pytest

pytestmark = pytest.mark.requires_db


class TestTableDeclaration:
    """`GLMBasisParams` -> `GLMBasisSelection` -> `GLMBasis`."""

    @pytest.mark.parametrize(
        "name,kind",
        [
            ("GLMBasisParams", dj.Lookup),
            ("GLMBasisSelection", dj.Manual),
            ("GLMBasis", dj.Computed),
        ],
    )
    def test_each_tier_declares(self, glm_basis, name, kind):
        table = getattr(glm_basis, name)
        assert issubclass(table, kind)
        assert table.heading is not None
        assert len(table.heading.names) > 0

    # Compared via full_table_name rather than by substring: DataJoint derives
    # a table's SQL name by splitting CamelCase, so `GLMBasisParams` becomes
    # `#g_l_m_basis_params` and no readable substring matches.
    def test_selection_depends_on_params(self, glm_basis):
        """A Selection table keys on its Params table."""
        assert (
            glm_basis.GLMBasisParams.full_table_name
            in glm_basis.GLMBasisSelection.parents()
        )

    def test_selection_depends_on_glm_storage(self, glm_basis):
        """And on the covariate table whose output it expands."""
        from mpfc_opto.glm.glm_tables import GLMStorage

        assert (
            GLMStorage.full_table_name in glm_basis.GLMBasisSelection.parents()
        )

    def test_computed_depends_on_selection(self, glm_basis):
        assert (
            glm_basis.GLMBasisSelection.full_table_name
            in glm_basis.GLMBasis.parents()
        )


class TestTriPartMake:
    """`GLMBasis` uses the tri-part make pattern rather than a plain `make`."""

    @pytest.mark.parametrize(
        "method", ["make_fetch", "make_compute", "make_insert"]
    )
    def test_defines_each_stage(self, glm_basis, method):
        assert callable(getattr(glm_basis.GLMBasis, method, None))

    def test_does_not_declare_parallel_make(self, glm_basis):
        """In spyglass `_parallel_make` means the make function itself spawns a
        process pool. This one does not, so the flag must stay unset or
        `populate(processes=N)` routes through NonDaemonPool for nothing."""
        assert "_parallel_make" not in vars(glm_basis.GLMBasis)
        assert glm_basis.GLMBasis._parallel_make is False
