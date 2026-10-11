"""Tests for the state-space EM filter functions.

These assert the mathematical relations each function is supposed to satisfy,
rather than pinning expected arrays, so the tests stay meaningful if the
implementation is rewritten.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def sigmoid(z):
    return np.exp(z) / (1.0 + np.exp(z))


class TestNewtonSolve:
    """`NewtonSolve` finds x satisfying x = xp + sp * (N - Nmax * p(x))."""

    @pytest.mark.parametrize("n_obs", [0, 1])
    @pytest.mark.parametrize("x_prior", [-0.5, 0.0, 0.5])
    @pytest.mark.parametrize("sigma_prior", [0.05, 0.25, 1.0])
    def test_satisfies_fixed_point(
        self, filters_em, x_prior, sigma_prior, n_obs
    ):
        mu = 0.0
        x = filters_em.NewtonSolve(x_prior, sigma_prior, n_obs, 1, mu)
        expected = x_prior + sigma_prior * (n_obs - 1 * sigmoid(mu + x))
        assert x == pytest.approx(expected, abs=1e-8)

    def test_observation_moves_estimate_upward(self, filters_em):
        """A success should pull the estimate above a failure's."""
        hit = filters_em.NewtonSolve(0.0, 0.5, 1, 1, 0.0)
        miss = filters_em.NewtonSolve(0.0, 0.5, 0, 1, 0.0)
        assert hit > miss

    def test_zero_prior_variance_returns_prior(self, filters_em):
        """With no prior uncertainty the observation cannot move x."""
        assert filters_em.NewtonSolve(0.3, 0.0, 1, 1, 0.0) == pytest.approx(
            0.3, abs=1e-8
        )


class TestFwdFilterEM:
    """The forward pass is a one-step-ahead filter over T observations."""

    @pytest.fixture
    def run(self, filters_em):
        y = np.array([0, 1, 1, 0, 1, 1, 1, 0])
        out = filters_em.FwdFilterEM(y, 1, 0.0, 0.25, 0.25, 0.0)
        return y, out

    def test_returns_T_plus_one_samples(self, run):
        """Index 0 holds the initial condition, so arrays are T+1 long."""
        y, (x_prior, x_post, s2_prior, s2_post, _) = run
        for arr in (x_prior, x_post, s2_prior, s2_post):
            assert len(arr) == len(y) + 1

    def test_initial_conditions_are_placed_at_index_zero(self, run):
        _, (_, x_post, _, s2_post, _) = run
        assert x_post[0] == 0.0
        assert s2_post[0] == 0.25

    def test_prior_mean_is_previous_posterior(self, run):
        y, (x_prior, x_post, _, _, _) = run
        for t in range(1, len(y) + 1):
            assert x_prior[t] == pytest.approx(x_post[t - 1])

    def test_prior_variance_adds_process_noise(self, run):
        y, (_, _, s2_prior, s2_post, _) = run
        for t in range(1, len(y) + 1):
            assert s2_prior[t] == pytest.approx(s2_post[t - 1] + 0.25)

    def test_posterior_variance_shrinks_prior(self, run):
        """Conditioning on an observation cannot increase variance."""
        y, (_, _, s2_prior, s2_post, _) = run
        for t in range(1, len(y) + 1):
            assert s2_post[t] <= s2_prior[t]
            assert s2_post[t] > 0

    def test_all_successes_drive_estimate_up(self, filters_em):
        ones = np.ones(12, dtype=int)
        _, x_post, _, _, _ = filters_em.FwdFilterEM(
            ones, 1, 0.0, 0.25, 0.25, 0.0
        )
        assert np.all(np.diff(x_post[1:]) > 0)


class TestBackwardFilter:
    """The smoother runs the forward estimates back through time."""

    @pytest.fixture
    def run(self, filters_em):
        y = np.array([0, 1, 1, 0, 1, 1, 1, 0])
        x_prior, x_post, s2_prior, s2_post, _ = filters_em.FwdFilterEM(
            y, 1, 0.0, 0.25, 0.25, 0.0
        )
        return (x_post, s2_post), filters_em.BackwardFilter(
            x_post, x_prior, s2_post, s2_prior
        )

    def test_last_sample_is_unchanged(self, run):
        """The final smoothed value has no future to borrow from."""
        (x_post, s2_post), (x_T, s2_T, _) = run
        assert x_T[-1] == pytest.approx(x_post[-1])
        assert s2_T[-1] == pytest.approx(s2_post[-1])

    def test_index_zero_is_left_at_zero(self, run):
        """The loop stops before index 0, so it keeps its initialized value.

        `EM` overwrites both afterwards; a caller using `BackwardFilter`
        directly gets a leading 0 that is not a smoothed estimate.
        """
        _, (x_T, s2_T, A) = run
        assert x_T[0] == 0.0
        assert s2_T[0] == 0.0
        assert A[0] == 0.0

    def test_gain_is_a_variance_ratio(self, run):
        _, (_, _, A) = run
        assert np.all(A[1:-1] > 0)
        assert np.all(A[1:-1] < 1)


class TestMSTEP:
    """The M-step returns the updated process-noise variance."""

    def test_returns_positive_scalar(self, filters_em):
        x = np.array([0.0, 0.1, 0.2, 0.25, 0.3, 0.33])
        s2 = np.full(6, 0.05)
        A = np.full(6, 0.5)
        out = filters_em.MSTEP(x, s2, A)
        assert np.isscalar(out) or np.ndim(out) == 0
        assert out > 0

    def test_flat_state_gives_smaller_variance_than_rising_state(
        self, filters_em
    ):
        s2 = np.full(6, 0.05)
        A = np.full(6, 0.5)
        flat = filters_em.MSTEP(np.zeros(6), s2, A)
        rising = filters_em.MSTEP(np.linspace(0, 1, 6), s2, A)
        assert flat < rising


class TestTransformToProb:
    """Posterior state is mapped to probability with confidence bounds."""

    @pytest.fixture
    def run(self, filters_em):
        meanv = np.array([-0.5, 0.0, 0.25, 0.75])
        sigma2 = np.full(4, 0.09)
        return meanv, filters_em.TransformToProb(meanv, sigma2, 0.0)

    def test_mode_is_the_sigmoid_of_the_mean(self, run):
        meanv, (pmode, _, _, _) = run
        assert pmode == pytest.approx(sigmoid(meanv), abs=1e-12)

    def test_bounds_bracket_the_median(self, run):
        _, (_, p, pll, pul) = run
        assert np.all(pll <= p)
        assert np.all(p <= pul)

    def test_everything_is_a_probability(self, run):
        _, out = run
        for arr in out:
            assert np.all(arr > 0) and np.all(arr < 1)

    def test_is_deterministic(self, filters_em):
        """Repeated calls agree: the simulation seeds its own generator."""
        meanv, sigma2 = np.array([0.0, 0.3]), np.full(2, 0.09)
        first = filters_em.TransformToProb(meanv, sigma2, 0.0)
        second = filters_em.TransformToProb(meanv, sigma2, 0.0)
        for a, b in zip(first, second):
            assert a == pytest.approx(b)

    def test_leaves_the_global_rng_alone(self, filters_em):
        """It used to call np.random.seed(0), resetting the caller's stream."""
        meanv, sigma2 = np.array([0.0, 0.3]), np.full(2, 0.09)
        np.random.seed(12345)
        expected = np.random.normal(size=3)

        np.random.seed(12345)
        filters_em.TransformToProb(meanv, sigma2, 0.0)
        after = np.random.normal(size=3)

        assert after == pytest.approx(expected)

    def test_wider_posterior_widens_the_interval(self, filters_em):
        meanv = np.array([0.0])
        narrow = filters_em.TransformToProb(meanv, np.array([0.01]), 0.0)
        wide = filters_em.TransformToProb(meanv, np.array([1.0]), 0.0)
        assert (wide[3] - wide[2])[0] > (narrow[3] - narrow[2])[0]
