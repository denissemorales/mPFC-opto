"""Tests for the spectral-slope helpers in `pss_utils`.

`unit`, not `requires_db`: these functions were split out of `pss` precisely
so they could be imported, and tested, without a database.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestFitPSSFromPSD:
    """`fit_pss_from_psd` returns (pss, intercept, slope), pss == -slope.

    The sign is inverted on purpose, so that a larger pss means a steeper 1/f
    falloff. A test asserting "slope < 0" would be testing the raw third
    element, not the headline value.
    """

    def test_falling_psd_gives_positive_pss(self, pss_utils):
        freqs = np.linspace(1.0, 100.0, 200)
        pss, _, slope = pss_utils.fit_pss_from_psd(freqs, freqs**-2.0)
        assert pss > 0
        assert slope < 0

    def test_pss_is_the_negated_slope(self, pss_utils):
        freqs = np.linspace(1.0, 100.0, 200)
        pss, _, slope = pss_utils.fit_pss_from_psd(freqs, freqs**-2.0)
        assert pss == pytest.approx(-slope)

    @pytest.mark.parametrize("exponent", [1.0, 2.0, 3.0])
    def test_recovers_a_known_exponent(self, pss_utils, exponent):
        freqs = np.linspace(1.0, 100.0, 400)
        pss, _, _ = pss_utils.fit_pss_from_psd(freqs, freqs**-exponent)
        assert pss == pytest.approx(exponent, abs=1e-6)

    def test_steeper_psd_gives_larger_pss(self, pss_utils):
        freqs = np.linspace(1.0, 100.0, 200)
        shallow, _, _ = pss_utils.fit_pss_from_psd(freqs, freqs**-1.0)
        steep, _, _ = pss_utils.fit_pss_from_psd(freqs, freqs**-4.0)
        assert steep > shallow

    def test_flat_psd_gives_near_zero_pss(self, pss_utils):
        freqs = np.linspace(1.0, 100.0, 200)
        pss, _, _ = pss_utils.fit_pss_from_psd(freqs, np.ones_like(freqs))
        assert pss == pytest.approx(0.0, abs=1e-9)

    def test_f_range_excludes_bins(self, pss_utils):
        """Only bins inside f_range are fitted."""
        freqs = np.linspace(1.0, 100.0, 400)
        psd = freqs**-2.0
        psd[freqs > 90.0] = 1e6  # corrupt outside the default range
        pss, _, _ = pss_utils.fit_pss_from_psd(freqs, psd)
        assert pss == pytest.approx(2.0, abs=1e-6)

    def test_too_few_valid_bins_raises(self, pss_utils):
        freqs = np.array([5.0, 6.0, 7.0])
        with pytest.raises(ValueError, match="Not enough valid frequency bins"):
            pss_utils.fit_pss_from_psd(freqs, freqs**-2.0)

    def test_nonpositive_psd_bins_are_dropped(self, pss_utils):
        """log10 of a zero or negative PSD is undefined, so those are masked."""
        freqs = np.linspace(1.0, 100.0, 200)
        psd = freqs**-2.0
        psd[:50] = 0.0
        pss, _, _ = pss_utils.fit_pss_from_psd(freqs, psd)
        assert np.isfinite(pss)


class TestComputePSSWindows:
    """Sliding-window PSS over a wideband trace."""

    FS = 1000.0

    @pytest.fixture
    def signal(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal(10_000)

    def test_one_pss_value_per_window(self, pss_utils, signal):
        t_centers, pss_vals = pss_utils.compute_pss_windows(signal, self.FS)[:2]
        assert len(t_centers) == len(pss_vals)
        assert len(pss_vals) > 1

    def test_window_count_follows_window_and_step(self, pss_utils, signal):
        """10 s of data, 2 s windows, 1 s steps -> 9 window starts."""
        t_centers = pss_utils.compute_pss_windows(
            signal, self.FS, window_s=2.0, step_s=1.0
        )[0]
        assert len(t_centers) == 9

    def test_centers_are_increasing_and_relative(self, pss_utils, signal):
        t_centers = pss_utils.compute_pss_windows(signal, self.FS)[0]
        assert np.all(np.diff(t_centers) > 0)
        assert t_centers[0] >= 0

    def test_window_shorter_than_16_samples_raises(self, pss_utils, signal):
        with pytest.raises(ValueError, match="window_s is too small"):
            pss_utils.compute_pss_windows(signal, self.FS, window_s=0.001)

    def test_nonpositive_step_raises(self, pss_utils, signal):
        with pytest.raises(ValueError, match="step_s must be positive"):
            pss_utils.compute_pss_windows(signal, self.FS, step_s=0.0)

    def test_signal_shorter_than_one_window_raises(self, pss_utils):
        with pytest.raises(ValueError, match="shorter than one window"):
            pss_utils.compute_pss_windows(np.zeros(100), self.FS, window_s=2.0)


class TestComputePSSWindowsWithTimestamps:
    """The absolute-time variant, used by `SleepPSS`."""

    FS = 1000.0

    @pytest.fixture
    def signal(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal(10_000)

    def test_centers_come_from_the_given_timestamps(self, pss_utils, signal):
        """Centers are sampled from `timestamps`, not computed from fs, so the
        result shares the sleep-scoring timebase."""
        t0 = 1_700_000_000.0
        timestamps = t0 + np.arange(len(signal)) / self.FS
        centers = pss_utils.compute_pss_windows_with_timestamps(
            signal, timestamps, self.FS
        )[0]
        assert np.all(centers >= t0)
        assert np.all(np.diff(centers) > 0)

    def test_agrees_with_the_relative_variant(self, pss_utils, signal):
        """Given timestamps starting at 0, both variants should agree on pss."""
        timestamps = np.arange(len(signal)) / self.FS
        rel = pss_utils.compute_pss_windows(signal, self.FS)[1]
        absolute = pss_utils.compute_pss_windows_with_timestamps(
            signal, timestamps, self.FS
        )[1]
        assert rel == pytest.approx(absolute)

    def test_length_mismatch_raises(self, pss_utils, signal):
        with pytest.raises(ValueError, match="same length"):
            pss_utils.compute_pss_windows_with_timestamps(
                signal, np.arange(len(signal) - 1) / self.FS, self.FS
            )
