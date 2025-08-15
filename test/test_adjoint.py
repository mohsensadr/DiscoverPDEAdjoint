import numpy as np
import pytest
from src.adjoint import forward_m

def test_forward_m_shape():
    numPDE = 2
    Nx = 5
    Nt = 3
    f0 = np.ones((numPDE, Nx))
    dx = [1.0]
    dt = 0.1
    ps = [(1, 0), (0, 1)]
    ds = [(0, 0)]
    params = np.zeros((numPDE, len(ds), len(ps)))

    f = forward_m(params, ps, ds, f0, dx, dt, Nt)
    assert f.shape == (numPDE, Nt, Nx)
    assert isinstance(f, np.ndarray)

def test_forward_m_zero_params_constant_solution():
    numPDE = 1
    Nx = 4
    Nt = 5
    f0 = np.random.rand(numPDE, Nx)
    dx = [1.0]
    dt = 0.1
    ps = [(1,)]
    ds = [(0,)]
    params = np.zeros((numPDE, len(ds), len(ps)))

    f = forward_m(params, ps, ds, f0, dx, dt, Nt)
    # Repeat f0 along time axis to match f
    expected = np.repeat(f0[:, None, :], Nt, axis=1)
    np.testing.assert_allclose(f, expected, atol=1e-12)

def test_forward_m_small_dt_changes_little():
    numPDE = 1
    Nx = 4
    Nt = 2
    f0 = np.ones((numPDE, Nx))
    dx = [1.0]
    dt = 1e-8
    ps = [(1,)]
    ds = [(0,)]
    params = np.ones((numPDE, len(ds), len(ps)))

    f = forward_m(params, ps, ds, f0, dx, dt, Nt)
    # Change should be very small
    diff = np.abs(f[:,1,:] - f[:,0,:])
    assert np.all(diff < 1e-7)

def test_forward_m_handles_derivatives():
    numPDE = 1
    Nx = 5
    Nt = 2
    f0 = np.arange(Nx).reshape((numPDE, Nx)).astype(float)
    dx = [1.0]
    dt = 0.1
    ps = [(1,)]
    ds = [(1,)]  # first derivative
    params = np.ones((numPDE, len(ds), len(ps)))

    f = forward_m(params, ps, ds, f0, dx, dt, Nt)
    # Just check it runs and shape is correct
    assert f.shape == (numPDE, Nt, Nx)

if __name__ == "__main__":
    pytest.main()

