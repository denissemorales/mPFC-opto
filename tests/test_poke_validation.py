"""Tests for the shared poke-validation helpers.

`unit`, not `requires_db`: `poke_validation` imports no DataJoint, which is the
point of having extracted it from the two table modules.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit

WELLS = {"left": (0.0, 0.0), "right": (100.0, 0.0)}


@pytest.fixture(scope="session")
def pv():
    from mpfc_opto.behavior import poke_validation

    return poke_validation


class TestInterpolatePosition:
    def test_returns_one_xy_pair_per_query(self, pv):
        t = np.arange(0.0, 10.0, 0.5)
        out = pv.interpolate_position(t, t * 2, t * 3, np.array([1.0, 2.0]))
        assert out.shape == (2, 2)

    def test_interpolates_linearly(self, pv):
        t = np.array([0.0, 1.0, 2.0])
        out = pv.interpolate_position(
            t,
            np.array([0.0, 10.0, 20.0]),
            np.array([0.0, -5.0, -10.0]),
            np.array([0.5, 1.5]),
        )
        assert out[:, 0] == pytest.approx([5.0, 15.0])
        assert out[:, 1] == pytest.approx([-2.5, -7.5])

    def test_hits_samples_exactly(self, pv):
        t = np.array([0.0, 1.0, 2.0])
        x, y = np.array([3.0, 4.0, 5.0]), np.array([6.0, 7.0, 8.0])
        out = pv.interpolate_position(t, x, y, t)
        assert out[:, 0] == pytest.approx(x)
        assert out[:, 1] == pytest.approx(y)

    def test_unsorted_sample_times_raise(self, pv):
        """np.interp returns nonsense rather than complaining, so this must."""
        t = np.array([0.0, 2.0, 1.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            pv.interpolate_position(t, t, t, np.array([0.5]))

    def test_duplicate_sample_times_raise(self, pv):
        t = np.array([0.0, 1.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            pv.interpolate_position(t, t, t, np.array([0.5]))


class TestValidatePokeEvents:
    """A poke is valid if the animal was near the well it reports."""

    FS = 100.0

    @pytest.fixture
    def position(self):
        """Animal sits at the left well, then walks to the right well."""
        t = np.arange(0.0, 20.0, 1.0 / self.FS)
        x = np.clip((t - 5.0) * 10.0, 0.0, 100.0)
        return t, x, np.zeros_like(t)

    def _run(self, pv, position, times, names, **kw):
        t, x, y = position
        return pv.validate_poke_events(
            well_positions=WELLS,
            poke_times=np.asarray(times, dtype=float),
            poke_names=np.asarray(names),
            position_times=t,
            position_x=x,
            position_y=y,
            plot=False,
            **kw,
        )

    def test_poke_at_the_reported_well_is_valid(self, pv, position):
        report = self._run(pv, position, [1.0], ["left"])
        assert report["summary"]["valid_pokes"] == 1
        assert report["summary"]["invalid_pokes"] == 0

    def test_poke_far_from_the_reported_well_is_invalid(self, pv, position):
        # At t=1 s the animal is at the left well, not the right one.
        report = self._run(pv, position, [1.0], ["right"])
        assert report["summary"]["valid_pokes"] == 0
        assert report["summary"]["invalid_pokes"] == 1

    def test_distance_threshold_is_honoured(self, pv, position):
        # At t=6 s the animal is ~10 units from the left well.
        near = self._run(pv, position, [6.0], ["left"], distance_threshold=25.0)
        far = self._run(pv, position, [6.0], ["left"], distance_threshold=5.0)
        assert near["summary"]["valid_pokes"] == 1
        assert far["summary"]["valid_pokes"] == 0

    def test_unknown_well_name_is_invalid(self, pv, position):
        """A name absent from well_positions has no distance, so it cannot pass."""
        report = self._run(pv, position, [1.0], ["nowhere"])
        assert report["summary"]["valid_pokes"] == 0

    def test_percent_valid_matches_the_counts(self, pv, position):
        report = self._run(
            pv,
            position,
            [1.0, 1.5, 2.0, 19.0],
            ["left", "left", "right", "right"],
        )
        s = report["summary"]
        expected = (
            100 * s["valid_pokes"] / (s["valid_pokes"] + s["invalid_pokes"])
        )
        assert s["percent_valid"] == pytest.approx(expected)

    def test_no_pokes_gives_zero_percent_not_a_zero_division(
        self, pv, position
    ):
        report = self._run(pv, position, [], [])
        assert report["summary"]["percent_valid"] == 0

    def test_report_exposes_the_offending_rows(self, pv, position):
        report = self._run(pv, position, [1.0, 1.0], ["left", "right"])
        assert len(report["valid_pokes"]) == 1
        assert len(report["invalid_pokes"]) == 1
        assert set(report["invalid_pokes"]["well_name"]) == {"right"}
