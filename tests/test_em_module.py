"""Tests for the EM entry points in `em_module`."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit

RESPONSES = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1]


@pytest.fixture
def axes():
    """A caller-supplied figure/axes pair.

    Required, not optional: see `test_runem_without_axes_raises`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


class TestEMMain:
    @pytest.fixture(scope="class")
    def result(self, em_module, request):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        request.addfinalizer(lambda: plt.close(fig))
        return em_module.EM_main(RESPONSES, fig_ax_list=(fig, ax))

    def test_returns_five_values(self, result):
        assert len(result) == 5

    def test_bounds_bracket_the_estimate(self, result):
        _, _, pll, pul, pmode = result
        assert np.all(pll <= pmode)
        assert np.all(pmode <= pul)

    def test_estimates_are_probabilities(self, result):
        _, _, pll, pul, pmode = result
        for arr in (pll, pul, pmode):
            assert np.all(arr > 0) and np.all(arr < 1)

    def test_learning_curve_ends_above_where_it_started(self, result):
        """The input is mostly failures then mostly successes."""
        _, _, _, _, pmode = result
        assert pmode[-1] > pmode[0]

    def test_returns_the_axes_it_was_given(self, result, em_module):
        fig, ax = result[0], result[1]
        assert fig is not None and ax is not None


class TestRunEM:
    def test_accepts_a_dataframe(self, em_module, axes):
        import pandas as pd

        df = pd.DataFrame({"y": RESPONSES})
        out = em_module.RunEM(df, fig_ax_list=axes)
        assert len(out) == 5

    def test_p_init_shifts_the_curve(self, em_module, axes):
        import pandas as pd

        df = pd.DataFrame({"y": RESPONSES})
        low = em_module.RunEM(df, p_init=0.1, fig_ax_list=axes)[4]
        high = em_module.RunEM(df, p_init=0.9, fig_ax_list=axes)[4]
        assert not np.allclose(low, high)

    def test_builds_its_own_axes_when_none_given(self, em_module):
        """Used to raise NameError: `figsize` was referenced but never a
        parameter, so any caller that did not pass axes crashed."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        fig, ax = em_module.RunEM(pd.DataFrame({"y": RESPONSES}))[:2]
        assert fig is not None and ax is not None
        plt.close(fig)

    def test_figsize_sizes_the_figure_it_builds(self, em_module):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        fig = em_module.RunEM(pd.DataFrame({"y": RESPONSES}), figsize=(7, 2))[0]
        assert tuple(fig.get_size_inches()) == (7.0, 2.0)
        plt.close(fig)
