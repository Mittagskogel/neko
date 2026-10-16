"""Tests for variance_stats.py."""

import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

# Import the module under test as a standalone module. Running pytest from the
# ``examples/tgv`` directory allows this import to resolve without installing
# the script as a package.
import variance_stats as vs


class FakeElement:
    """Minimal stand-in for a pymech element."""

    def __init__(self, vel, scal):
        self.vel = vel
        self.scal = scal


class FakeHexaData:
    """Minimal stand-in for ``pymech.core.HexaData``."""

    def __init__(self, time, lr1, var, elem):
        self.time = time
        self.lr1 = lr1
        self.var = var
        self.elem = elem


def _make_data(nelv=2, order=3, time=0.0, n_vel=3, n_scal=3):
    """Create a FakeHexaData object with constant field values."""
    lx = ly = lz = order
    # Create simple velocity and scalar arrays.  By default velocity is
    # (u, v, w) = (1, 2, 3) and scalars are (10, 20, 30) everywhere.
    elem = []
    for _ in range(nelv):
        vel = np.ones((3, lz, ly, lx))
        vel[0] *= 1.0
        vel[1] *= 2.0
        vel[2] *= 3.0
        scal = np.ones((n_scal, lz, ly, lx))
        for i in range(n_scal):
            scal[i] *= 10.0 * (i + 1)
        elem.append(FakeElement(vel, scal))

    var = [3, n_vel, 1, 1, n_scal]
    return FakeHexaData(time, (lx, ly, lz), var, elem)


class TestGllWeights1d:
    """Tests for ``gll_weights_1d``."""

    def test_order_zero_is_one(self):
        w = vs.gll_weights_1d(0)
        assert w.shape == (1,)
        assert w[0] == pytest.approx(1.0)

    def test_sum_to_two(self):
        for order in [1, 2, 4, 7, 12]:
            w = vs.gll_weights_1d(order)
            assert w.sum() == pytest.approx(2.0)

    def test_lobatto_endpoints_are_smallest(self):
        # The endpoint weights are the smallest for GLL rules.
        w = vs.gll_weights_1d(7)
        assert w[0] < w[1]
        assert w[-1] < w[-2]

    def test_symmetry(self):
        w = vs.gll_weights_1d(7)
        np.testing.assert_allclose(w, w[::-1])


class TestParseNek5000Meta:
    """Tests for ``parse_nek5000_meta``."""

    def test_parses_valid_meta(self, tmp_path: Path):
        meta = tmp_path / "field0.nek5000"
        meta.write_text(
            "filetemplate: field%01d.f%05d\n"
            "firsttimestep: 0\n"
            "numtimesteps: 5\n"
        )
        template, first, nsteps = vs.parse_nek5000_meta(meta)
        assert template == "field%01d.f%05d"
        assert first == 0
        assert nsteps == 5

    def test_missing_field_raises(self, tmp_path: Path):
        meta = tmp_path / "field0.nek5000"
        meta.write_text("filetemplate: field%01d.f%05d\n")
        with pytest.raises(ValueError, match="Could not parse meta file"):
            vs.parse_nek5000_meta(meta)


class TestBuildFieldFilename:
    """Tests for ``build_field_filename``."""

    def test_builds_correct_name(self, tmp_path: Path):
        meta = tmp_path / "field0.nek5000"
        fname = vs.build_field_filename(meta, "field%01d.f%05d", 3)
        assert fname == tmp_path / "field0.f00003"


class TestWeightedStats:
    """Tests for ``_weighted_stats`` and derived variance helpers."""

    def test_constant_field(self):
        # Uniform weights and constant values: mean = value, variance = 0.
        weights = np.ones((2, 3, 4))
        values = np.ones((5, 2, 3, 4)) * 7.0
        mean, var = vs._weighted_stats(weights, values)
        assert mean == pytest.approx(7.0)
        assert var == pytest.approx(0.0)

    def test_uniform_weights_reproduce_numpy_mean(self):
        weights = np.ones((2, 3, 4))
        rng = np.random.default_rng(0)
        values = rng.random((3, 2, 3, 4))
        mean, _ = vs._weighted_stats(weights, values)
        assert mean == pytest.approx(values.mean())

    def test_scalar_variance(self):
        weights = np.ones((3, 3, 3))
        # Values alternate between 0 and 1 with equal weight -> mean=0.5, var=0.25
        values = np.zeros((2, 3, 3, 3))
        values[0] = 1.0
        mean, var = vs.scalar_variance(weights, values)
        assert mean == pytest.approx(0.5)
        assert var == pytest.approx(0.25)

    def test_vector_variance_constant_vector(self):
        # Constant vector (1, 2, 3) -> mean magnitude = sqrt(14), variance = 0.
        weights = np.ones((3, 3, 3))
        shape = (2, 3, 3, 3)
        c1 = np.ones(shape) * 1.0
        c2 = np.ones(shape) * 2.0
        c3 = np.ones(shape) * 3.0
        mean_mag, var = vs.vector_variance(weights, c1, c2, c3)
        assert mean_mag == pytest.approx(np.sqrt(14.0))
        assert var == pytest.approx(0.0)

    def test_vector_variance_two_component_vector(self):
        # Vector (1, 0) everywhere -> mean magnitude = 1, variance = 0.
        weights = np.ones((2, 2, 2))
        shape = (1, 2, 2, 2)
        c1 = np.ones(shape)
        c2 = np.zeros(shape)
        c3 = np.zeros(shape)
        mean_mag, var = vs.vector_variance(weights, c1, c2, c3)
        assert mean_mag == pytest.approx(1.0)
        assert var == pytest.approx(0.0)


class TestResolveFieldSpec:
    """Tests for argument resolution."""

    def test_default_is_vorticity_vector(self):
        spec = vs._resolve_field_spec(None, None, None, None)
        assert isinstance(spec, vs.VectorSpec)
        assert spec.names == ("omega_x", "omega_y", "omega_z")

    def test_velocity_preset(self):
        spec = vs._resolve_field_spec("velocity", None, None, None)
        assert isinstance(spec, vs.VectorSpec)
        assert spec.names == ("u", "v", "w")

    def test_custom_fields(self):
        spec = vs._resolve_field_spec(None, "a,b,c", None, None)
        assert isinstance(spec, vs.VectorSpec)
        assert spec.names == ("a", "b", "c")

    def test_scalar_spec(self):
        spec = vs._resolve_field_spec(None, None, "omega_x", 0)
        assert isinstance(spec, vs.ScalarSpec)
        assert spec.name == "omega_x"
        assert spec.index == 0

    def test_mutual_exclusion(self):
        with pytest.raises(ValueError, match="Only one of"):
            vs._resolve_field_spec("velocity", "a,b,c", None, None)

    def test_invalid_vector_preset(self):
        with pytest.raises(ValueError, match="Unknown vector preset"):
            vs._resolve_field_spec("pressure", None, None, None)


class TestReadVectorField:
    """Tests for ``read_vector_field`` with mocked ``readnek``."""

    def test_reads_velocity_storage(self, monkeypatch):
        data = _make_data(n_vel=3, n_scal=0)
        monkeypatch.setattr(vs, "readnek", lambda _path: data)

        time, weights, c1, c2, c3 = vs.read_vector_field(
            Path("dummy.f00001"), ("u", "v", "w")
        )
        assert time == 0.0
        assert weights.shape == (3, 3, 3)
        assert c1.shape == c2.shape == c3.shape == (2, 3, 3, 3)
        assert c1.mean() == pytest.approx(1.0)
        assert c2.mean() == pytest.approx(2.0)
        assert c3.mean() == pytest.approx(3.0)

    def test_reads_scalar_storage(self, monkeypatch):
        data = _make_data(n_vel=0, n_scal=3)
        monkeypatch.setattr(vs, "readnek", lambda _path: data)

        time, weights, c1, c2, c3 = vs.read_vector_field(
            Path("dummy.f00001"), ("omega_x", "omega_y", "omega_z")
        )
        assert c1.mean() == pytest.approx(10.0)
        assert c2.mean() == pytest.approx(20.0)
        assert c3.mean() == pytest.approx(30.0)


class TestReadScalarField:
    """Tests for ``read_scalar_field`` with mocked ``readnek``."""

    def test_reads_velocity_component(self, monkeypatch):
        data = _make_data(n_vel=3, n_scal=3)
        monkeypatch.setattr(vs, "readnek", lambda _path: data)

        time, weights, values = vs.read_scalar_field(
            Path("dummy.f00001"), "v"
        )
        assert values.mean() == pytest.approx(2.0)

    def test_reads_named_scalar(self, monkeypatch):
        data = _make_data(n_vel=0, n_scal=3)
        monkeypatch.setattr(vs, "readnek", lambda _path: data)

        time, weights, values = vs.read_scalar_field(
            Path("dummy.f00001"), "omega_y"
        )
        assert values.mean() == pytest.approx(20.0)

    def test_explicit_scalar_index_overrides_name(self, monkeypatch):
        data = _make_data(n_vel=3, n_scal=3)
        monkeypatch.setattr(vs, "readnek", lambda _path: data)

        # Even though 'u' is a velocity name, scalar_index forces scalar slot 2.
        time, weights, values = vs.read_scalar_field(
            Path("dummy.f00001"), "u", scalar_index=2
        )
        assert values.mean() == pytest.approx(30.0)


class TestMain:
    """End-to-end tests for ``main`` using mocked file I/O."""

    def _write_meta(self, tmp_path: Path, name: str, nsteps: int = 3) -> Path:
        meta = tmp_path / f"{name}.nek5000"
        meta.write_text(
            f"filetemplate: {name}%01d.f%05d\n"
            f"firsttimestep: 0\n"
            f"numtimesteps: {nsteps}\n"
        )
        return meta

    def test_single_series_vector(self, tmp_path: Path, monkeypatch):
        meta = self._write_meta(tmp_path, "field")
        data = _make_data(n_vel=3, n_scal=3)
        monkeypatch.setattr(vs, "readnek", lambda _path: data)

        stdout = StringIO()
        with patch.object(sys, "stdout", stdout):
            ret = vs.main([str(meta), "--no-progress"])

        assert ret == 0
        lines = stdout.getvalue().strip().split("\n")
        assert lines[0] == "# time mean_magnitude variance"
        # Each of the 3 timesteps should produce one data row.
        assert len(lines) == 4

    def test_single_series_scalar_velocity(self, tmp_path: Path, monkeypatch):
        meta = self._write_meta(tmp_path, "field")
        data = _make_data(n_vel=3, n_scal=0)
        monkeypatch.setattr(vs, "readnek", lambda _path: data)

        stdout = StringIO()
        with patch.object(sys, "stdout", stdout):
            ret = vs.main([str(meta), "--scalar", "u", "--no-progress"])

        assert ret == 0
        lines = stdout.getvalue().strip().split("\n")
        assert lines[0] == "# time mean variance"
        assert len(lines) == 4

    def test_two_series_scalar_difference(self, tmp_path: Path, monkeypatch):
        meta1 = self._write_meta(tmp_path, "field1")
        meta2 = self._write_meta(tmp_path, "field2")
        # Identical data -> difference variance is exactly zero.
        data = _make_data(n_vel=0, n_scal=3)
        monkeypatch.setattr(vs, "readnek", lambda _path: data)

        stdout = StringIO()
        with patch.object(sys, "stdout", stdout):
            ret = vs.main(
                [str(meta1), "--second", str(meta2), "--scalar", "omega_x", "--no-progress"]
            )

        assert ret == 0
        lines = stdout.getvalue().strip().split("\n")
        assert lines[0] == "# time mean variance mean_diff variance_diff"
        # Last column (variance of the difference) should be 0.
        parts = lines[1].split()
        assert float(parts[-1]) == pytest.approx(0.0)
