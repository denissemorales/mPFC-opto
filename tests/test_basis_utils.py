"""Tests for the design-matrix basis helpers in `basis_utils`.

`unit`, not `requires_db`: these were split out of `GLMBasis` precisely so they
could be imported, and tested, without a database. They still need `jax`, whose
compiled extension is tied to a NumPy ABI, so the fixture skips if it is
unimportable.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestWrapTo2Pi:
    def test_maps_into_zero_to_two_pi(self, basis_utils):
        x = np.array([-3 * np.pi, -0.1, 0.0, 1.0, 7.0, 100.0])
        out = basis_utils._wrap_to_2pi(x)
        assert np.all(out >= 0)
        assert np.all(out < 2 * np.pi)

    def test_is_identity_inside_the_range(self, basis_utils):
        x = np.array([0.0, 1.0, 3.0, 6.0])
        assert basis_utils._wrap_to_2pi(x) == pytest.approx(x)

    def test_preserves_angle_modulo_two_pi(self, basis_utils):
        x = np.array([-1.0, 0.5, 8.0])
        out = basis_utils._wrap_to_2pi(x)
        assert np.cos(out) == pytest.approx(np.cos(x))
        assert np.sin(out) == pytest.approx(np.sin(x))


class TestNanToZero:
    def test_replaces_nan(self, basis_utils):
        out = np.asarray(basis_utils._nan_to_zero(np.array([1.0, np.nan, 3.0])))
        assert not np.any(np.isnan(out))
        assert out[1] == 0.0

    def test_leaves_finite_values_alone(self, basis_utils):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.asarray(basis_utils._nan_to_zero(x)) == pytest.approx(x)

    def test_returns_a_jax_array(self, basis_utils):
        """Downstream code indexes the result with jax semantics."""
        out = basis_utils._nan_to_zero(np.array([1.0, 2.0]))
        assert type(out).__module__.startswith("jax")


class TestGetUnitColumns:
    def test_selects_only_unit_columns(self, basis_utils):
        import pandas as pd

        df = pd.DataFrame(
            {
                "time": [0.0, 0.05],
                "unit_0": [0, 1],
                "unit_11": [1, 0],
                "speed": [1.0, 2.0],
            }
        )
        assert basis_utils.get_unit_columns(df) == ["unit_0", "unit_11"]

    def test_returns_empty_when_there_are_no_units(self, basis_utils):
        import pandas as pd

        df = pd.DataFrame({"time": [0.0], "speed": [1.0]})
        assert basis_utils.get_unit_columns(df) == []

    def test_preserves_column_order(self, basis_utils):
        """Column order is the design matrix's column order; it must not sort."""
        import pandas as pd

        df = pd.DataFrame({c: [0] for c in ["unit_2", "unit_0", "unit_1"]})
        assert basis_utils.get_unit_columns(df) == [
            "unit_2",
            "unit_0",
            "unit_1",
        ]


class TestSplitBasisByCategory:
    """Masks a basis to each category's rows, then concatenates the pieces.

    Returns `(combined, cat_idx, names, labels)`. `eval_basis` must be a jax
    array: the implementation zeroes rows with `.at[...].set(0.0)`.
    """

    N_BASIS = 3
    N_ROWS = 8

    @pytest.fixture
    def eval_basis(self, basis_utils):
        import jax.numpy as jnp

        flat = np.arange(self.N_ROWS * self.N_BASIS, dtype=float) + 1.0
        return jnp.array(flat.reshape(self.N_ROWS, self.N_BASIS))

    @pytest.fixture
    def alternating(self):
        return np.array(["a", "b", "a", "b", "a", "b", "a", "b"])

    def test_one_block_of_columns_per_category(
        self, basis_utils, eval_basis, alternating
    ):
        combined, cat_idx, names, labels = basis_utils._split_basis_by_category(
            eval_basis, alternating, self.N_BASIS, "x"
        )
        assert np.asarray(combined).shape == (self.N_ROWS, 2 * self.N_BASIS)
        assert len(cat_idx) == 2 * self.N_BASIS
        assert len(names) == 2 * self.N_BASIS
        assert labels == ["a", "b"]

    def test_rows_outside_a_category_are_zeroed(self, basis_utils, eval_basis):
        cats = np.array(["a"] * 4 + ["b"] * 4)
        combined = np.asarray(
            basis_utils._split_basis_by_category(
                eval_basis, cats, self.N_BASIS, "x"
            )[0]
        )
        first = combined[:, : self.N_BASIS]
        second = combined[:, self.N_BASIS :]
        assert np.all(first[4:] == 0)
        assert np.all(second[:4] == 0)
        assert np.any(first[:4] != 0)
        assert np.any(second[4:] != 0)

    def test_blocks_sum_back_to_the_basis(
        self, basis_utils, eval_basis, alternating
    ):
        """Categories partition the rows, so the blocks recover the input."""
        combined = np.asarray(
            basis_utils._split_basis_by_category(
                eval_basis, alternating, self.N_BASIS, "x"
            )[0]
        )
        blocks = [
            combined[:, i * self.N_BASIS : (i + 1) * self.N_BASIS]
            for i in range(2)
        ]
        assert sum(blocks) == pytest.approx(np.asarray(eval_basis))

    def test_nan_rows_contribute_all_zeros(self, basis_utils, eval_basis):
        """A row with no category must not drive any interaction term."""
        cats = np.array(
            [np.nan, "a", "a", "a", "b", "b", "b", "b"], dtype=object
        )
        combined, _, _, labels = basis_utils._split_basis_by_category(
            eval_basis, cats, self.N_BASIS, "x"
        )
        assert labels == ["a", "b"]  # NaN is not a category
        assert np.all(np.asarray(combined)[0] == 0)

    def test_labels_are_sorted(self, basis_utils, eval_basis):
        """Block order follows sorted labels, not order of appearance."""
        cats = np.array(["z"] * 4 + ["a"] * 4)
        _, _, _, labels = basis_utils._split_basis_by_category(
            eval_basis, cats, self.N_BASIS, "x"
        )
        assert labels == ["a", "z"]

    def test_names_encode_prefix_category_and_bump(
        self, basis_utils, eval_basis
    ):
        cats = np.array(["a"] * 4 + ["b"] * 4)
        names = basis_utils._split_basis_by_category(
            eval_basis, cats, self.N_BASIS, "ppt"
        )[2]
        assert names[: self.N_BASIS] == [
            f"ppt_a_bump_{i}" for i in range(self.N_BASIS)
        ]
        assert names[self.N_BASIS :] == [
            f"ppt_b_bump_{i}" for i in range(self.N_BASIS)
        ]

    def test_cat_idx_labels_each_column(self, basis_utils, eval_basis):
        cats = np.array(["a"] * 4 + ["b"] * 4)
        cat_idx = basis_utils._split_basis_by_category(
            eval_basis, cats, self.N_BASIS, "x"
        )[1]
        assert list(cat_idx) == ["a"] * self.N_BASIS + ["b"] * self.N_BASIS

    def test_no_valid_categories_gives_zero_columns(
        self, basis_utils, eval_basis
    ):
        cats = np.array([np.nan] * self.N_ROWS, dtype=object)
        combined, _, names, labels = basis_utils._split_basis_by_category(
            eval_basis, cats, self.N_BASIS, "x"
        )
        assert np.asarray(combined).shape == (self.N_ROWS, 0)
        assert names == [] and labels == []
