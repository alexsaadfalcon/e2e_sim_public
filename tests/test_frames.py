"""Tests for e2e/frames.py -- the frame-shape contract accessor module."""

import pytest
torch = pytest.importorskip("torch")

from e2e.frames import (
    FrameContractError,
    FrameDims,
    dims,
    require_no_mimo,
    require_single_chirp,
    chirp0,
    to_aperture_grid,
)


def _frame(n_rx=1024, n_tx=1, n_chirp=1, n_freqs=16, device="cpu"):
    return torch.zeros(n_rx, n_tx, n_chirp, n_freqs, dtype=torch.complex64, device=device)


# --------------------------------------------------------------------------- dims

def test_dims_happy_path():
    s_pars = _frame(n_rx=1024, n_tx=1, n_chirp=1, n_freqs=32)
    d = dims(s_pars)
    assert d == FrameDims(1024, 1, 1, 32)
    assert d.n_rx == 1024 and d.n_tx == 1 and d.n_chirp == 1 and d.n_freqs == 32


def test_dims_wrong_ndim_raises():
    s_pars = torch.zeros(4, 4, 4)  # 3-D, missing a dim
    with pytest.raises(ValueError, match=r"4 dims.*\[n_rx, n_tx, n_chirp, n_freqs\].*got shape \(4, 4, 4\)"):
        dims(s_pars)


def test_dims_extra_ndim_raises():
    s_pars = torch.zeros(2, 2, 2, 2, 2)  # 5-D
    with pytest.raises(ValueError, match=r"5 dims"):
        dims(s_pars)


# --------------------------------------------------------------------------- require_no_mimo / require_single_chirp

def test_require_no_mimo_passes_for_single_tx():
    s_pars = _frame(n_tx=1)
    require_no_mimo(s_pars, who="TestBlock")  # should not raise


def test_require_no_mimo_raises_and_names_caller_and_shape():
    s_pars = _frame(n_tx=2)
    with pytest.raises(ValueError, match=r"TestBlock: MIMO not supported yet.*got shape \(1024, 2, 1, 16\)"):
        require_no_mimo(s_pars, who="TestBlock")


def test_require_single_chirp_passes_for_one_chirp():
    s_pars = _frame(n_chirp=1)
    require_single_chirp(s_pars, who="TestBlock")  # should not raise


def test_require_single_chirp_raises_and_names_caller_and_shape():
    s_pars = _frame(n_chirp=2)
    with pytest.raises(ValueError, match=r"TestBlock: multiple chirps not supported yet.*got shape \(1024, 1, 2, 16\)"):
        require_single_chirp(s_pars, who="TestBlock")


# --------------------------------------------------------------------------- chirp0

def test_chirp0_shape_and_values():
    s_pars = torch.arange(1 * 1 * 2 * 3, dtype=torch.float32).to(torch.complex64).reshape(1, 1, 2, 3)
    out = chirp0(s_pars)
    assert out.shape == (1, 1, 3)
    assert torch.equal(out, s_pars[:, :, 0, :])


def test_chirp0_is_a_view_not_a_copy():
    s_pars = _frame(n_rx=2, n_tx=1, n_chirp=1, n_freqs=3)
    out = chirp0(s_pars)
    out[0, 0, 0] = 5 + 5j
    assert s_pars[0, 0, 0, 0] == 5 + 5j  # mutation visible in the source -> it's a view


def test_chirp0_wrong_ndim_raises():
    with pytest.raises(ValueError, match=r"4 dims"):
        chirp0(torch.zeros(4, 4, 4))


def test_chirp0_on_aperture_grid_layout():
    """chirp0 also slices dim 2 of an aperture-grid tensor [rx_x, rx_y, n_chirp, n_freqs]
    (e.g. produced by to_aperture_grid), not just raw [n_rx, n_tx, n_chirp, n_freqs] frames."""
    s_pars = _frame(n_rx=4, n_tx=1, n_chirp=1, n_freqs=3)
    grid = to_aperture_grid(s_pars, (2, 2))  # [2, 2, 1, 3]
    out = chirp0(grid)
    assert out.shape == (2, 2, 3)
    assert torch.equal(out, grid[:, :, 0, :])


# --------------------------------------------------------------------------- to_aperture_grid

def test_to_aperture_grid_happy_path():
    s_pars = _frame(n_rx=1024, n_tx=1, n_chirp=1, n_freqs=16)
    out = to_aperture_grid(s_pars, (32, 32))
    assert out.shape == (32, 32, 1, 16)


def test_to_aperture_grid_is_a_view_not_a_copy():
    s_pars = _frame(n_rx=4, n_tx=1, n_chirp=1, n_freqs=3)
    out = to_aperture_grid(s_pars, (2, 2))
    out[0, 0, 0, 0] = 7 + 7j
    assert s_pars[0, 0, 0, 0] == 7 + 7j  # mutation visible in the source -> it's a view


def test_to_aperture_grid_mismatched_shape_raises():
    s_pars = _frame(n_rx=1024, n_tx=1, n_chirp=1, n_freqs=16)
    with pytest.raises(ValueError, match=r"\(31, 32\).*992.*n_rx=1024"):
        to_aperture_grid(s_pars, (31, 32))


def test_to_aperture_grid_wrong_ndim_raises():
    with pytest.raises(ValueError, match=r"4 dims"):
        to_aperture_grid(torch.zeros(4, 4, 4), (2, 2))


def test_contract_violations_raise_the_named_error_type():
    """All shape-contract guards raise FrameContractError (a ValueError subclass), the
    type webapp/pipeline_runner.py maps to its 'Pipeline constraint failed' message."""
    assert issubclass(FrameContractError, ValueError)
    with pytest.raises(FrameContractError):
        dims(torch.zeros(4, 4, 4))
    with pytest.raises(FrameContractError):
        require_no_mimo(_frame(n_tx=2), who="TestBlock")
    with pytest.raises(FrameContractError):
        require_single_chirp(_frame(n_chirp=2), who="TestBlock")
    with pytest.raises(FrameContractError):
        to_aperture_grid(_frame(n_rx=1024), (31, 32))


def test_to_aperture_grid_recovers_sionna_column_first_ordering():
    """Sionna's PlanarArray numbers antennas column-first: flat RX index
    = col * num_rows + row (the row index varies fastest). A row-major reshape
    therefore needs the slow (column) axis first — to_aperture_grid((num_cols,
    num_rows)) must land element (col, row) at grid[col, row], so grid dim 0 is
    azimuth (columns) and dim 1 elevation (rows)."""
    num_rows, num_cols = 2, 3
    s_pars = _frame(n_rx=num_rows * num_cols, n_freqs=1)
    for col in range(num_cols):
        for row in range(num_rows):
            s_pars[col * num_rows + row, 0, 0, 0] = complex(col, row)
    grid = to_aperture_grid(s_pars, (num_cols, num_rows))
    assert grid.shape == (num_cols, num_rows, 1, 1)
    for col in range(num_cols):
        for row in range(num_rows):
            assert grid[col, row, 0, 0] == complex(col, row)


# --------------------------------------------------------------------------- device handling

def test_device_follows_input(torch_device):
    s_pars = _frame(n_rx=1024, n_tx=1, n_chirp=1, n_freqs=16, device=torch_device)
    assert chirp0(s_pars).device.type == torch.device(torch_device).type
    assert to_aperture_grid(s_pars, (32, 32)).device.type == torch.device(torch_device).type
